# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 320.0

    # Build clause arrays.
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1

    # Build var -> list of (clause_idx, lit_pos) for incremental updates.
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clauses[v].append((i, j))

    best_assignment = None
    best_unsat = m

    def make_evaluator(assignment_signs):
        # Returns a function that computes per-clause satisfaction status.
        lit_true = (assignment_signs[clause_vars] == clause_signs)  # (m,3)
        sat = lit_true.any(axis=1)  # (m,)
        return sat

    restart = 0
    while time.time() < deadline:
        restart += 1
        rng = random.Random(restart * 104729 + 7)

        # Random initial assignment.
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int8)
        assignment_signs = np.where(assignment, 1, -1).astype(np.int8)

        sat = make_evaluator(assignment_signs)
        unsat_count = int((~sat).sum())

        if unsat_count == 0:
            return list(assignment)

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()

        # Search parameters: adapt slightly per restart.
        p_random = min(0.5, 0.2 + 0.02 * restart)
        max_steps = 80000

        step = 0
        while step < max_steps:
            if time.time() >= deadline:
                break

            if unsat_count == 0:
                return list(assignment)

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()

            # Pick an unsatisfied clause.
            unsat_indices = np.where(~sat)[0]
            clause_idx = rng.choice(unsat_indices.tolist())

            # Choose literal to flip.
            if rng.random() < p_random:
                lit_pos = rng.randint(0, 2)
            else:
                # Greedy: evaluate each of the 3 literals.
                best_score = unsat_count + 100
                best_lit = 0
                for lit_pos_candidate in range(3):
                    v = clause_vars[clause_idx, lit_pos_candidate]
                    # Flip v.
                    old = assignment_signs[v]
                    assignment_signs[v] = -old

                    # Update affected clauses.
                    delta = 0
                    for (ci, cj) in var_clauses[v]:
                        was_sat = sat[ci]
                        # Recompute clause ci satisfaction.
                        new_lit_true = (assignment_signs[clause_vars[ci]] == clause_signs[ci])
                        now_sat = new_lit_true.any()
                        if was_sat and not now_sat:
                            delta += 1
                        elif not was_sat and now_sat:
                            delta -= 1
                        sat[ci] = now_sat

                    new_unsat = unsat_count + delta

                    # Undo flip.
                    assignment_signs[v] = old
                    for (ci, cj) in var_clauses[v]:
                        new_lit_true = (assignment_signs[clause_vars[ci]] == clause_signs[ci])
                        sat[ci] = new_lit_true.any()

                    if new_unsat < best_score:
                        best_score = new_unsat
                        best_lit = lit_pos_candidate
                lit_pos = best_lit

            # Perform the flip.
            v = clause_vars[clause_idx, lit_pos]
            old = assignment_signs[v]
            assignment_signs[v] = -old

            # Update affected clauses.
            delta = 0
            for (ci, cj) in var_clauses[v]:
                was_sat = sat[ci]
                new_lit_true = (assignment_signs[clause_vars[ci]] == clause_signs[ci])
                now_sat = new_lit_true.any()
                if was_sat and not now_sat:
                    delta += 1
                elif not was_sat and now_sat:
                    delta -= 1
                sat[ci] = now_sat

            unsat_count += delta
            step += 1

    if best_assignment is not None:
        return list(best_assignment)
    rng = random.Random(42)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
