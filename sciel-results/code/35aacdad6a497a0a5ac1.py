"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # Deadline guard: 325s safety margin out of 330s limit
    deadline = time.time() + 325.0

    # Precompute clause literal arrays
    # lits: shape (m, 3), signed 1-based literals
    lits = np.array(clauses, dtype=np.int32)
    # var_idx: 0-based variable indices
    var_idx = np.abs(lits) - 1
    # sign: +1 if literal is positive, -1 if negative
    sign = np.where(lits > 0, 1, -1)

    # Assignment as int array: 0 or 1
    # A clause is satisfied if any literal is true:
    #   literal true iff (sign * (2*assignment[var] - 1)) > 0
    #   i.e., assignment[var] == 1 if sign==1, assignment[var] == 0 if sign==-1

    rng = np.random.RandomState(42)

    def is_satisfied(assignment):
        # assignment: (n,) int array of 0/1
        # For each literal: val = sign * (2*assignment[var_idx] - 1)
        # Satisfied if val > 0 for any literal in clause
        vals = sign * (2 * assignment[var_idx] - 1)  # (m, 3)
        return (vals > 0).any(axis=1)  # (m,) bool

    def count_unsat(assignment):
        return m - is_satisfied(assignment).sum()

    best_assignment = None
    best_unsat = m + 1

    for restart in range(1000):
        # Check deadline
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = (rng.rand(n) < 0.5).astype(np.int32)

        # Check if already satisfied
        if count_unsat(assignment) == 0:
            return assignment.tolist()

        unsat_count = count_unsat(assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()

        # Adaptive noise parameters
        stuck = 0
        noise = 0.0  # probability of random flip
        min_noise = 0.0
        max_noise = 0.8

        for step in range(200000):
            # Check deadline
            if step % 1000 == 0:
                if time.time() > deadline:
                    break

            # Compute unsat clauses
            sat = is_satisfied(assignment)
            unsat_mask = ~sat
            unsat_idx = np.where(unsat_mask)[0]

            if len(unsat_idx) == 0:
                return assignment.tolist()

            # Track best
            cur_unsat = len(unsat_idx)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment.copy()

            # Adaptive noise: increase when stuck
            if cur_unsat <= stuck:
                stuck = cur_unsat
            else:
                noise = min(noise + 0.02, max_noise)

            # Select an unsatisfied clause
            clause_idx = unsat_idx[rng.randint(len(unsat_idx))]
            clause_vars = var_idx[clause_idx]
            clause_signs = sign[clause_idx]

            # Determine which literals in this clause are false
            # A literal is false if sign * (2*assignment[var] - 1) <= 0
            vals = clause_signs * (2 * assignment[clause_vars] - 1)
            false_mask = vals <= 0

            # With probability `noise`, pick a random variable from the clause
            # Otherwise, pick from false literals with weighting (standard WalkSAT)
            if rng.rand() < noise:
                # Random variable from clause
                var = clause_vars[rng.randint(3)]
            else:
                # Weighted selection from false literals
                if not false_mask.any():
                    var = clause_vars[rng.randint(3)]
                else:
                    false_vars = clause_vars[false_mask]
                    # Uniform among false literals
                    var = false_vars[rng.randint(len(false_vars))]

            # Flip the variable
            assignment[var] = 1 - assignment[var]

            # Periodically check deadline
            if step % 5000 == 0:
                if time.time() > deadline:
                    break

        # If we're about to exceed deadline, stop
        if time.time() > deadline:
            break

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
