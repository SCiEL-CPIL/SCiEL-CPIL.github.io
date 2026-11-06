"""Improved planted 3-SAT solver with deadline guard and refined WalkSAT."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # 280s safety margin within 300s limit

    # Precompute clause literal indices and signs for speed
    clause_idx = []
    clause_sign = []
    for c in clauses:
        idx = [abs(l) - 1 for l in c]
        sign = [1 if l > 0 else -1 for l in c]
        clause_idx.append(idx)
        clause_sign.append(sign)

    m = len(clauses)

    # Count how many times each variable appears in each clause (for scoring)
    # We'll use a simpler approach: just track unsat count per clause

    def make_random_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]

    def count_unsat(assignment):
        """Return list of unsatisfied clause indices."""
        unsat = []
        for i in range(m):
            satisfied = False
            for j in range(3):
                if assignment[clause_idx[i][j]] == (clause_sign[i][j] > 0):
                    satisfied = True
                    break
            if not satisfied:
                unsat.append(i)
        return unsat

    def walk_sat(assignment, rng, max_steps):
        """Run WalkSAT with a fixed step budget. Returns (assignment, solved)."""
        unsat = count_unsat(assignment)
        if not unsat:
            return assignment, True

        for step in range(max_steps):
            if time.time() > deadline:
                break

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat)

            # Score each literal in the clause by number of other clauses it would satisfy
            best_var = -1
            best_score = -1
            best_flip = False

            for j in range(3):
                var = clause_idx[ci][j]
                sign = clause_sign[ci][j]
                current_val = assignment[var]
                new_val = not current_val

                # Would flipping this variable satisfy this clause?
                # The literal is satisfied if assignment[var] == (sign > 0)
                literal_satisfied = (new_val == (sign > 0))

                # Count how many other unsat clauses this flip would satisfy
                # and how many satisfied clauses it would break
                score = 0
                for k in range(m):
                    if k == ci:
                        continue
                    # Check if variable is in clause k
                    for jj in range(3):
                        if clause_idx[k][jj] == var:
                            k_sign = clause_sign[k][jj]
                            # Currently satisfied?
                            cur_sat = (assignment[var] == (k_sign > 0))
                            # After flip?
                            new_sat = (new_val == (k_sign > 0))
                            if not cur_sat and new_sat:
                                score += 1
                            elif cur_sat and not new_sat:
                                score -= 1
                            break  # Variable appears at most once per clause in 3-SAT

                if literal_satisfied:
                    score += 100  # Strong preference for satisfying the chosen clause

                if score > best_score:
                    best_score = score
                    best_var = var
                    best_flip = new_val

            # With some probability, pick a random literal instead (escape local minima)
            if rng.random() < 0.3:
                j = rng.randint(0, 2)
                best_var = clause_idx[ci][j]
                best_flip = not assignment[best_var]

            assignment[best_var] = best_flip

            # Recompute unsat clauses (incremental would be faster but simpler here)
            unsat = count_unsat(assignment)
            if not unsat:
                return assignment, True

        return assignment, False

    # Multi-restart WalkSAT
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Try several restarts with increasing budgets
    restarts = [
        (2000, 0),
        (5000, 1),
        (10000, 2),
        (20000, 3),
        (50000, 4),
    ]

    for max_steps, seed in restarts:
        if time.time() > deadline:
            break
        rng2 = random.Random(seed)
        assignment = make_random_assignment(rng2)
        assignment, solved = walk_sat(assignment, rng2, max_steps)

        if solved:
            return assignment

        unsat_count = len(count_unsat(assignment))
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

    # Final: use best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: random assignment
    rng3 = random.Random(12345)
    return make_random_assignment(rng3)

# EVOLVE-BLOCK-END
