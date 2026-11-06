"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    deadline = time.time() + 280.0

    # Precompute clause variable sets for fast evaluation
    clause_vars = [[abs(l) - 1 for l in c] for c in clauses]

    # WalkSAT with adaptive noise and clause weighting
    assignment = [rng.random() < 0.5 for _ in range(n)]

    def is_satisfied(c_idx):
        c = clauses[c_idx]
        for i in range(len(c)):
            v = c[i]
            if assignment[abs(v) - 1] == (v > 0):
                return True
        return False

    def num_unsat(c_idx):
        c = clauses[c_idx]
        cnt = 0
        for i in range(len(c)):
            v = c[i]
            if assignment[abs(v) - 1] != (v > 0):
                cnt += 1
        return cnt

    def count_unsat_clauses():
        cnt = 0
        for ci in range(len(clauses)):
            if not is_satisfied(ci):
                cnt += 1
        return cnt

    # Try multiple restarts with increasing effort
    max_restarts = 50
    base_steps = 5000
    noise = 0.15

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Reset assignment with slight bias
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Track clause weights for weighting selection
        clause_weights = [1.0] * len(clauses)

        steps = base_steps + restart * 2000
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break

            # Find unsatisfied clauses
            unsat_indices = []
            for ci in range(len(clauses)):
                if not is_satisfied(ci):
                    unsat_indices.append(ci)

            if not unsat_indices:
                return assignment

            # Weighted selection of a clause
            total_w = sum(clause_weights[ci] for ci in unsat_indices)
            r = rng.random() * total_w
            cum = 0.0
            chosen_ci = unsat_indices[0]
            for ci in unsat_indices:
                cum += clause_weights[ci]
                if r <= cum:
                    chosen_ci = ci
                    break

            # Choose literal to flip
            clause = clauses[chosen_ci]
            # Count how many literals are false (all are false since clause is unsat)
            # Pick a literal to flip
            # With some probability, pick a literal that appears in few other unsat clauses
            lit_indices = [abs(v) - 1 for v in clause]
            if rng.random() < noise:
                # Random flip
                var_idx = rng.choice(lit_indices)
            else:
                # Heuristic: pick variable that appears in fewest other unsat clauses
                best_var = lit_indices[0]
                best_score = float('inf')
                for vi in lit_indices:
                    score = 0
                    for other_ci in unsat_indices:
                        if other_ci != chosen_ci:
                            if vi in clause_vars[other_ci]:
                                score += 1
                    if score < best_score:
                        best_score = score
                        best_var = vi
                var_idx = best_var

            assignment[var_idx] = not assignment[var_idx]

            # Update clause weights
            # Decrease weight of satisfied clauses, increase unsat
            for ci in unsat_indices:
                if ci == chosen_ci:
                    # This clause is now potentially satisfied
                    if is_satisfied(ci):
                        clause_weights[ci] = max(0.5, clause_weights[ci] * 0.9)
                    else:
                        clause_weights[ci] = min(10.0, clause_weights[ci] * 1.1)
                else:
                    # Check if flipping made this clause more or less satisfied
                    # Simple: just decay weights slowly
                    clause_weights[ci] = min(10.0, clause_weights[ci] * 1.001)

            # Occasionally decay all weights
            if step % 1000 == 0:
                for ci in range(len(clauses)):
                    clause_weights[ci] = max(0.5, clause_weights[ci] * 0.99)

    # Final check
    for ci in range(len(clauses)):
        if not is_satisfied(ci):
            break
    else:
        return assignment

    return assignment
# EVOLVE-BLOCK-END
