"""Combined WalkSAT solver for planted 3-SAT (150 vars) with incremental tracking,
adaptive p, and deadline guard."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Build variable-to-clauses index for incremental updates
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append(ci)

    def is_clause_satisfied(clause, assignment):
        """Check if a single clause is satisfied."""
        for var_idx, sign in clause:
            if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                return True
        return False

    def walk_one(assignment, rng, max_steps, p_flip):
        """Run incremental WalkSAT from current assignment for up to max_steps."""
        # Initialize unsatisfied clause set
        unsat_clauses = set()
        for ci, clause in enumerate(clause_data):
            if not is_clause_satisfied(clause, assignment):
                unsat_clauses.add(ci)

        if not unsat_clauses:
            return assignment, True

        # Maintain variable unsat count incrementally
        var_unsat_count = [0] * n
        for ci in unsat_clauses:
            for var_idx, sign in clause_data[ci]:
                var_unsat_count[var_idx] += 1

        unsat_list = list(unsat_clauses)

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if not unsat_list:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_list)
            clause = clause_data[ci]

            # Weighted selection: combine var_unsat_count with noise
            # With probability p_flip, use pure random; otherwise use weighted
            if rng.random() < p_flip:
                chosen_idx = rng.randint(0, len(clause) - 1)
            else:
                weights = []
                for var_idx, sign in clause:
                    weights.append(var_unsat_count[var_idx] + 1)

                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                chosen_idx = 0
                for idx, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen_idx = idx
                        break

            var_idx, sign = clause[chosen_idx]
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val

            # Incrementally update unsat_clauses and var_unsat_count
            # Clauses that could change are those containing var_idx
            for affected_ci in var_to_clauses[var_idx]:
                was_unsat = affected_ci in unsat_clauses
                now_sat = is_clause_satisfied(clause_data[affected_ci], assignment)

                if was_unsat and now_sat:
                    # Clause became satisfied: remove from unsat, decrement counts
                    unsat_clauses.discard(affected_ci)
                    unsat_list.remove(affected_ci)
                    for v, s in clause_data[affected_ci]:
                        var_unsat_count[v] -= 1
                elif not was_unsat and not now_sat:
                    # Clause became unsatisfied: add to unsat, increment counts
                    unsat_clauses.add(affected_ci)
                    unsat_list.append(affected_ci)
                    for v, s in clause_data[affected_ci]:
                        var_unsat_count[v] += 1

        return assignment, False

    # Multi-restart strategy
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters
    max_steps_per_restart = 20000
    num_restarts = 2000  # max restarts, will stop at deadline

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Cycle through different p_flip values for diversity
        p_flip = 0.2 + 0.3 * (restart % 5) / 4.0  # 0.2, 0.275, 0.35, 0.425, 0.5

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart, p_flip)

        if solved:
            return assignment

        # Track best: count unsat clauses
        uc = 0
        for ci, clause in enumerate(clause_data):
            if not is_clause_satisfied(clause, assignment):
                uc += 1
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
