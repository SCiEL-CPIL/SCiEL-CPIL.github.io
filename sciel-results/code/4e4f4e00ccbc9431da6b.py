# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: kill after 320s wall-clock (safety margin for 330s limit)
    deadline = time.time() + 320.0

    if m == 0:
        return [False] * n

    # Pre-compute clause data as numpy arrays for vectorized operations
    # Each clause: (abs_indices, signs) where signs[i] indicates if literal is positive
    # We'll use a representation that makes evaluation fast.
    # For each clause, store:
    #   vars: array of variable indices (0-based)
    #   signs: array of +1/-1 (literal polarity)
    # A clause is satisfied if any literal is true:
    #   literal value = assignment[var] if sign==1, else not assignment[var]

    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8)  # +1 or -1

    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1

    # Build incidence: for each variable, which clauses contain it (with sign)
    # var_to_clauses[v] = list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            s = int(clause_signs[i, j])
            var_to_clauses[v].append((i, s))

    # Pre-compute for each variable: list of (clause_idx, sign)
    var_clause_lists = var_to_clauses

    # Assignment as numpy bool array
    assignment = np.random.rand(n) < 0.5

    def is_satisfied_clause(ci):
        """Check if clause ci is satisfied."""
        for j in range(3):
            v = int(clause_vars[ci, j])
            s = int(clause_signs[ci, j])
            if s == 1:
                if assignment[v]:
                    return True
            else:
                if not assignment[v]:
                    return True
        return False

    def count_unsatisfied():
        """Count unsatisfied clauses (vectorized)."""
        # For each clause, check if any literal is true
        # literal_value[i,j] = assignment[clause_vars[i,j]] if sign==1 else ~assignment[clause_vars[i,j]]
        av = assignment[clause_vars]  # shape (m, 3)
        # Where sign is -1, flip
        flipped = np.where(clause_signs == -1, ~av, av)
        # Clause satisfied if any literal true
        sat = flipped.any(axis=1)
        return int((~sat).sum())

    def get_unsatisfied_clauses():
        """Return indices of unsatisfied clauses."""
        av = assignment[clause_vars]
        flipped = np.where(clause_signs == -1, ~av, av)
        sat = flipped.any(axis=1)
        return np.where(~sat)[0]

    # Track best assignment (fewest unsatisfied)
    best_unsat = count_unsatisfied()
    best_assignment = assignment.copy()

    if best_unsat == 0:
        return best_assignment.tolist()

    # WalkSAT with vectorized clause evaluation
    # Strategy: pick an unsatisfied clause, flip a variable in it
    # Use a weighted random choice: prefer variables that appear in more unsat clauses

    rng = np.random.RandomState(42)

    # Parameters
    max_restarts = 50
    steps_per_restart = 5000
    # We'll do multiple restarts, each with a random initial assignment

    # First, try with the current assignment
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        if restart > 0:
            # New random assignment
            assignment = np.random.rand(n) < 0.5
            best_unsat = count_unsatisfied()
            if best_unsat == 0:
                return assignment.tolist()
            best_assignment = assignment.copy()

        # WalkSAT loop for this restart
        for step in range(steps_per_restart):
            if time.time() > deadline - 5.0:
                break

            unsat_indices = get_unsatisfied_clauses()
            n_unsat = len(unsat_indices)

            if n_unsat == 0:
                return assignment.tolist()

            if n_unsat < best_unsat:
                best_unsat = n_unsat
                best_assignment = assignment.copy()

            # Pick a random unsatisfied clause
            ci = int(unsat_indices[rng.randint(n_unsat)])

            # Pick a variable from this clause to flip
            # Weight by how many unsatisfied clauses contain it (simple heuristic)
            # For simplicity, pick uniformly from the 3 literals in the clause
            j = rng.randint(3)
            v = int(clause_vars[ci, j])
            assignment[v] = not assignment[v]

    # Final check
    if best_unsat == 0:
        return best_assignment.tolist()

    # Return best found
    return best_assignment.tolist()
# EVOLVE-BLOCK-END
