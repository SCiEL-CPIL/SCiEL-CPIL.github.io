"""Improved WalkSAT with incremental clause tracking, min-conflicts heuristic, and deadline guard."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 288.0  # safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Build variable-to-clause index for efficient conflict counting
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append(ci)

    def check_all_satisfied(assignment):
        """Check if all clauses are satisfied."""
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def walk_one(assignment, rng, max_steps, p_flip, start_time, deadline):
        """Run WalkSAT with incremental tracking from current assignment."""
        # Build initial state: for each clause, track how many literals are satisfied
        # sat_count[c] = number of satisfied literals in clause c
        sat_count = [0] * num_clauses
        for ci, clause in enumerate(clause_data):
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat_count[ci] += 1

        # unsat_set: set of clause indices that are unsatisfied
        unsat_set = set(ci for ci in range(num_clauses) if sat_count[ci] == 0)

        # For min-conflicts: track for each variable, how many unsat clauses it appears in
        # conflict_count[var] = number of unsat clauses containing var
        conflict_count = [0] * n
        for ci in unsat_set:
            for var_idx, sign in clause_data[ci]:
                conflict_count[var_idx] += 1

        for step in range(max_steps):
            # Periodic deadline check (every 100 steps to reduce overhead)
            if step % 100 == 0 and time.time() > deadline:
                return assignment, False

            if not unsat_set:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clause_data[ci]

            # For each literal in the clause, compute a score
            # Use min-conflicts: prefer flipping variable with highest conflict count
            # Add randomness controlled by p_flip
            scores = []
            for var_idx, sign in clause:
                base = conflict_count[var_idx] + 1
                noise = rng.random() * p_flip * 10.0
                scores.append(base + noise)

            # Pick literal with weighted probability
            total_w = sum(scores)
            r = rng.random() * total_w
            cumulative = 0
            chosen_idx = 0
            for idx, w in enumerate(scores):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = idx
                    break

            var_idx, sign = clause[chosen_idx]

            # Flip the variable and update tracking
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val

            # Update sat_count and unsat_set for all clauses containing var_idx
            for cj in var_to_clauses[var_idx]:
                cj_clause = clause_data[cj]
                # Find sign of this variable in clause cj
                cj_sign = None
                for v, s in cj_clause:
                    if v == var_idx:
                        cj_sign = s
                        break
                if cj_sign is None:
                    continue

                # Determine if this literal was satisfied before and after the flip
                was_sat = (old_val and cj_sign == 1) or (not old_val and cj_sign == -1)
                now_sat = (assignment[var_idx] and cj_sign == 1) or (not assignment[var_idx] and cj_sign == -1)

                if was_sat and not now_sat:
                    # Literal went from satisfied to unsatisfied
                    sat_count[cj] -= 1
                    if sat_count[cj] == 0:
                        # Clause became unsatisfied
                        unsat_set.add(cj)
                        for v, s in cj_clause:
                            conflict_count[v] += 1
                elif not was_sat and now_sat:
                    # Literal went from unsatisfied to satisfied
                    sat_count[cj] += 1
                    if sat_count[cj] == 1:
                        # Clause became satisfied (was unsat, now has exactly 1 sat literal)
                        unsat_set.discard(cj)
                        for v, s in cj_clause:
                            conflict_count[v] -= 1

        return assignment, False

    # Multi-restart strategy
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    max_steps_per_restart = 30000
    num_restarts = 2000  # max restarts, will stop at deadline

    # Cycle through different p_flip values
    p_flip_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        p_flip = p_flip_values[restart % len(p_flip_values)]

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart, p_flip, start_time, deadline)

        if solved:
            return assignment

        # Track best
        uc = count_unsat(assignment)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # If we have a best assignment with few unsat clauses, try a final polish
    if best_assignment is not None and best_unsat_count < num_clauses:
        # Try a short, focused search from the best assignment
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            # Try a few more walks from the best state
            for _ in range(5):
                if time.time() > deadline:
                    break
                assignment = best_assignment[:]
                assignment, solved = walk_one(assignment, rng, 5000, 0.3, start_time, deadline)
                if solved:
                    return assignment
                uc = count_unsat(assignment)
                if uc < best_unsat_count:
                    best_unsat_count = uc
                    best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
