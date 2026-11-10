# EVOLVE-BLOCK-START
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Precompute clause literal arrays as 0-based signed indices.
    # lit_val[i] = (sign, var_idx) where sign in {+1, -1}, var_idx in [0, n)
    # We store: clause_vars (m,3) of var indices, clause_signs (m,3) of +1/-1.
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1

    rng = np.random.RandomState(0)

    # Initial assignment: random
    assignment = rng.rand(n) < 0.5  # bool array, True = True

    # Compute per-clause satisfaction: a clause is satisfied if any literal is true.
    # lit_true[i,j] = (assignment[var] == (sign==1))
    def compute_clause_sat(assignment):
        # For each literal position, is it satisfied?
        # assignment is bool array length n
        # var indices: clause_vars (m,3)
        # sign: clause_signs (m,3)
        # A literal is satisfied if assignment[var] == (sign == 1)
        a = assignment[clause_vars]  # (m,3) bool: the variable's value
        s = (clause_signs == 1)      # (m,3) bool: True if positive literal
        lit_sat = (a == s)           # (m,3) bool
        clause_sat = lit_sat.any(axis=1)  # (m,) bool
        return clause_sat

    clause_sat = compute_clause_sat(assignment)
    if clause_sat.all():
        return assignment.tolist()

    # Per-variable impact score: number of unsatisfied clauses that contain this variable.
    # We'll maintain this incrementally.
    # impact[v] = count of unsat clauses containing variable v
    # Build a mapping: for each clause, its 3 variables.
    # impact = zeros(n)
    # For each unsat clause, increment impact for each of its 3 variables.

    # Precompute: for each variable, list of (clause_idx, literal_pos) — but that's large.
    # Instead, recompute impact from scratch each iteration? That's O(m*3) per iteration.
    # Better: maintain incrementally.

    # Let's maintain:
    # - clause_sat: (m,) bool
    # - For each variable, we need to know which clauses it appears in.
    # Build var_to_clauses: dict var -> list of clause indices
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            var_to_clauses[v].append(i)

    # impact[v] = number of unsatisfied clauses containing v
    impact = np.zeros(n, dtype=np.int32)
    for i in range(m):
        if not clause_sat[i]:
            for j in range(3):
                impact[int(clause_vars[i, j])] += 1

    # Flip budget
    max_flips = 100000
    # Parameters for noise
    noise = 0.1

    for _ in range(max_flips):
        if clause_sat.all():
            break

        # Find variable with max impact, with random tie-breaking / noise
        # Add small random noise to break ties and escape local minima
        noisy_impact = impact + rng.rand(n) * noise * max(1, impact.max())
        var_to_flip = int(np.argmax(noisy_impact))

        if impact[var_to_flip] == 0:
            # No unsat clause contains this variable; pick random variable
            var_to_flip = int(rng.randint(0, n - 1))

        # Decide new value for var_to_flip:
        # Count how many currently-unsatisfied clauses would be satisfied
        # if we set var_to_flip to True vs False.
        # For each unsat clause containing var_to_flip:
        #   the literal sign tells us what value satisfies it.
        # If we set var=True: satisfied if sign==+1
        # If we set var=False: satisfied if sign==-1
        clauses_of_var = var_to_clauses[var_to_flip]
        # Filter to unsat clauses
        unsat_clauses_of_var = [ci for ci in clauses_of_var if not clause_sat[ci]]
        if not unsat_clauses_of_var:
            # Random flip
            new_val = not assignment[var_to_flip]
        else:
            sat_if_true = 0
            sat_if_false = 0
            for ci in unsat_clauses_of_var:
                for j in range(3):
                    if int(clause_vars[ci, j]) == var_to_flip:
                        if clause_signs[ci, j] == 1:
                            sat_if_true += 1
                        else:
                            sat_if_false += 1
            # Choose the value that satisfies more, with random tie-break
            if sat_if_true > sat_if_false:
                new_val = True
            elif sat_if_false > sat_if_true:
                new_val = False
            else:
                new_val = bool(rng.rand() < 0.5)

        if new_val == assignment[var_to_flip]:
            continue  # no change, skip

        # Flip the variable
        assignment[var_to_flip] = new_val

        # Update clause_sat and impact incrementally
        # For each clause containing var_to_flip, recompute its satisfaction
        for ci in clauses_of_var:
            old_sat = clause_sat[ci]
            # Recompute this clause
            is_sat = False
            for j in range(3):
                v = int(clause_vars[ci, j])
                s = clause_signs[ci, j] == 1
                if assignment[v] == s:
                    is_sat = True
                    break
            clause_sat[ci] = is_sat
            if old_sat and not is_sat:
                # Clause became unsat: increment impact for its 3 vars
                for j in range(3):
                    impact[int(clause_vars[ci, j])] += 1
            elif (not old_sat) and is_sat:
                # Clause became sat: decrement impact for its 3 vars
                for j in range(3):
                    impact[int(clause_vars[ci, j])] -= 1

    return assignment.tolist()
# EVOLVE-BLOCK-END
