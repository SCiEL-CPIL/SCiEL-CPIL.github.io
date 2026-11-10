"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """Optimized WalkSAT with incremental tracking, phase saving, and adaptive restarts."""
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0

    # Pre-extract clause literals as (var_idx, sign)
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Build variable-to-clauses index
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append(i)

    def is_clause_satisfied(clause, assignment):
        for var_idx, sign in clause:
            if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for clause in clause_data:
            if not is_clause_satisfied(clause, assignment):
                count += 1
        return count

    def walk_one(assignment, rng, max_steps, noise=0.5):
        """WalkSAT with incremental unsat tracking."""
        # Initialize unsatisfied clause set
        unsat_set = set()
        for i, clause in enumerate(clause_data):
            if not is_clause_satisfied(clause, assignment):
                unsat_set.add(i)

        if not unsat_set:
            return assignment, True

        # Phase saving: track which phase satisfied more clauses
        phase_count = [0] * n
        for i, clause in enumerate(clause_data):
            if is_clause_satisfied(clause, assignment):
                for var_idx, sign in clause:
                    if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                        phase_count[var_idx] += 1

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if not unsat_set:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = next(iter(unsat_set))
            clause = clause_data[ci]

            # Score each literal in the clause
            best_score = -1
            best_idx = 0
            for idx, (var_idx, sign) in enumerate(clause):
                # Count how many unsat clauses this variable appears in
                # and how many would be satisfied if we flip
                score = 0
                for cj in var_to_clauses[var_idx]:
                    if cj in unsat_set:
                        # Check if flipping var_idx would satisfy clause cj
                        cj_data = clause_data[cj]
                        would_satisfy = False
                        for v, s in cj_data:
                            if v == var_idx:
                                # After flip, check if this literal satisfies
                                new_val = not assignment[var_idx]
                                if (new_val and s == 1) or (not new_val and s == -1):
                                    would_satisfy = True
                                    break
                            else:
                                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                                    would_satisfy = True
                                    break
                        if would_satisfy:
                            score += 1
                        else:
                            score -= 1  # flipping might unsatisfy
                    else:
                        # Clause is currently satisfied; check if flip would unsatisfy
                        cj_data = clause_data[cj]
                        would_unsatisfy = True
                        for v, s in cj_data:
                            if v == var_idx:
                                new_val = not assignment[var_idx]
                                if not ((new_val and s == 1) or (not new_val and s == -1)):
                                    would_unsatisfy = False
                                    break
                            else:
                                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                                    would_unsatisfy = False
                                    break
                        if would_unsatisfy:
                            score -= 1

                if score > best_score:
                    best_score = score
                    best_idx = idx

            # With probability noise, pick a random literal instead
            if rng.random() < noise:
                best_idx = rng.randint(0, len(clause) - 1)

            var_idx, sign = clause[best_idx]
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val

            # Update phase counts
            if new_val:
                phase_count[var_idx] += 1
            else:
                phase_count[var_idx] -= 1

            # Update unsat set: check clauses containing var_idx
            for cj in var_to_clauses[var_idx]:
                clause_j = clause_data[cj]
                was_unsat = cj in unsat_set
                now_sat = is_clause_satisfied(clause_j, assignment)
                if was_unsat and now_sat:
                    unsat_set.remove(cj)
                elif not was_unsat and not now_sat:
                    unsat_set.add(cj)

            if not unsat_set:
                return assignment, True

        return assignment, False

    # Multi-restart with adaptive parameters
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Phase-saved initial assignment
    phase_saved = [False] * n
    for var_idx in range(n):
        # Use phase that appears in more satisfied clauses (approximate)
        pos_count = 0
        neg_count = 0
        for cj in var_to_clauses[var_idx]:
            clause_j = clause_data[cj]
            for v, s in clause_j:
                if v == var_idx:
                    if s == 1:
                        pos_count += 1
                    else:
                        neg_count += 1
        phase_saved[var_idx] = pos_count > neg_count

    # Try phase-saved first
    assignment = phase_saved[:]
    assignment, solved = walk_one(assignment, rng, 50000, noise=0.5)
    if solved:
        return assignment
    uc = count_unsat(assignment)
    if uc < best_unsat_count:
        best_unsat_count = uc
        best_assignment = assignment[:]

    # Multi-restart
    restart_num = 0
    while time.time() < deadline:
        restart_num += 1
        if time.time() > deadline:
            break

        # Adaptive noise: start high, decrease
        base_noise = max(0.1, 0.5 - restart_num * 0.02)
        noise = base_noise + rng.random() * 0.1

        # Adaptive steps: increase with restarts
        steps = min(50000, 5000 + restart_num * 1000)

        # Random initial assignment with phase bias
        assignment = []
        for var_idx in range(n):
            if rng.random() < 0.3:
                assignment.append(phase_saved[var_idx])
            else:
                assignment.append(rng.random() < 0.5)

        assignment, solved = walk_one(assignment, rng, steps, noise=noise)

        if solved:
            return assignment

        uc = count_unsat(assignment)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

        # Early termination if stuck at same count for many restarts
        if best_unsat_count > 0 and restart_num > 20:
            # Try a few more then give up
            if restart_num > 50 and best_unsat_count > 5:
                break

    if best_assignment is not None:
        return best_assignment
    return [False] * n

# EVOLVE-BLOCK-END
