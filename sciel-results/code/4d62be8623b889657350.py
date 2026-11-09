"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    # Planted 3-SAT solver with WalkSAT + restarts and a deadline guard.
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: return best result before 330s wall-clock.
    start_time = time.time()
    DEADLINE = 320.0  # safety margin

    # Preprocess: convert clauses to 0-based signed ints for speed.
    # Each clause is a list of 3 signed 1-based literals.
    # Convert to (var_idx, sign) where sign is +1 or -1.
    clause_vars = np.empty((m, 3), dtype=np.int32)  # 0-based variable index
    clause_sigs = np.empty((m, 3), dtype=np.int8)   # +1 or -1
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_sigs[i, j] = 1 if lit > 0 else -1

    # Precompute: for each variable, which clauses contain it (as positive or negative).
    # This helps with efficient evaluation.
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_vars[i, j]].append(i)

    # Build a fast clause satisfaction checker using numpy.
    # assignment is a numpy array of booleans (True = variable is True).
    # A clause is satisfied if any literal is true.
    # literal is true iff (assignment[var] == (sig > 0))

    def is_all_satisfied(assignment):
        # assignment: np.array of bool, length n
        # For each clause, check if any literal is satisfied.
        # lit_val[i, j] = assignment[clause_vars[i,j]] * (sig==1) or (1-assignment[...]) * (sig==-1)
        # Actually: literal is satisfied if assignment[var] matches the sign.
        # assignment[var] is True/False. sig is +1/-1.
        # Satisfied if (sig == 1 and assignment[var]) or (sig == -1 and not assignment[var])
        # = (sig * (2*assignment[var] - 1)) > 0
        # Let's compute: for each (i,j), val = assignment[clause_vars[i,j]]
        # satisfied if (sig==1 and val) or (sig==-1 and not val)
        # = (val == (sig > 0))
        # Using numpy:
        vals = assignment[clause_vars]  # shape (m, 3)
        sigs = clause_sigs  # shape (m, 3)
        # satisfied_lit = vals == (sigs > 0)
        # clause_satisfied = satisfied_lit.any(axis=1)
        # return not clause_satisfied.any()
        # Faster: compute number of unsatisfied clauses
        unsat_mask = ~(vals == (sigs > 0)).any(axis=1)
        return int(unsat_mask.sum())

    def count_unsat(assignment):
        vals = assignment[clause_vars]
        sigs = clause_sigs
        unsat_mask = ~(vals == (sigs > 0)).any(axis=1)
        return int(unsat_mask.sum())

    # WalkSAT with restarts
    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1

    # Parameters
    flip_prob = 0.5  # probability of flipping a random literal in the clause (vs. the one that helps most)
    max_flips_per_run = 5000
    max_restarts = 200

    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break

        # Initialize random assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)

        # Try to solve
        for flip in range(max_flips_per_run):
            if time.time() - start_time > DEADLINE:
                break

            unsat_count = count_unsat(assignment)
            if unsat_count == 0:
                return assignment.tolist()

            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()

            # Find an unsatisfied clause
            vals = assignment[clause_vars]
            sigs = clause_sigs
            unsat_mask = ~(vals == (sigs > 0)).any(axis=1)
            unsat_indices = np.where(unsat_mask)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()

            # Pick a random unsatisfied clause
            ci = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]

            # Choose which literal to flip
            if rng.random() < flip_prob:
                # Random literal in the clause
                li = rng.randint(0, 2)
            else:
                # Pick the literal that, if flipped, would satisfy the most clauses
                # For each of the 3 literals in this clause, compute how many clauses it would satisfy
                best_li = 0
                best_gain = -1
                for li in range(3):
                    var = clause_vars[ci, li]
                    sig = clause_sigs[ci, li]
                    # If we flip this literal, it becomes satisfied.
                    # How many other clauses contain this variable?
                    # This is expensive; approximate by just picking randomly among the 3
                    # Actually for planted SAT, random is often fine
                    best_li = li
                    break  # just pick first for speed; or random
                li = li if rng.random() < 0.3 else rng.randint(0, 2)

            var = clause_vars[ci, li]
            assignment[var] = not assignment[var]

        if time.time() - start_time > DEADLINE:
            break

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()

    # Fallback: all True
    return [True] * n
# EVOLVE-BLOCK-END
