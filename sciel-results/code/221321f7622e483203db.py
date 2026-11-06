import time
import random
import sys
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin

    if m == 0:
        return [True] * n

    # Build clause structure
    clause_vars = []   # clause_vars[ci] = list of var indices (0-based)
    clause_signs = []  # clause_signs[ci] = list of +1/-1
    for clause in clauses:
        vars_in = []
        signs_in = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_in.append(v)
            signs_in.append(s)
        clause_vars.append(vars_in)
        clause_signs.append(signs_in)

    # Variable occurrence: var_occ[v] = list of (clause_idx, sign)
    var_occ = [[] for _ in range(n)]
    for ci in range(m):
        for k in range(len(clause_vars[ci])):
            var_occ[clause_vars[ci][k]].append((ci, clause_signs[ci][k]))

    # --- Preprocessing: unit propagation + pure literal elimination ---
    assignment = [None] * n
    clause_sat = [0] * m  # number of satisfied literals per clause

    def compute_clause_sat():
        for ci in range(m):
            sc = 0
            for k in range(len(clause_vars[ci])):
                v = clause_vars[ci][k]
                s = clause_signs[ci][k]
                if assignment[v] is not None:
                    if (s == 1 and assignment[v]) or (s == -1 and not assignment[v]):
                        sc += 1
            clause_sat[ci] = sc

    def propagate():
        changed = True
        while changed:
            changed = False
            # Pure literal elimination
            for v in range(n):
                if assignment[v] is not None:
                    continue
                has_pos = False
                has_neg = False
                for (ci, s) in var_occ[v]:
                    if clause_sat[ci] > 0:
                        continue
                    if s == 1:
                        has_pos = True
                    else:
                        has_neg = True
                if has_pos and not has_neg:
                    assignment[v] = True
                    changed = True
                elif has_neg and not has_pos:
                    assignment[v] = False
                    changed = True
            # Unit propagation
            for ci in range(m):
                if clause_sat[ci] > 0:
                    continue
                unassigned = []
                for k in range(len(clause_vars[ci])):
                    v = clause_vars[ci][k]
                    if assignment[v] is None:
                        unassigned.append((v, clause_signs[ci][k]))
                if len(unassigned) == 1:
                    v, s = unassigned[0]
                    assignment[v] = (s == 1)
                    changed = True
            if changed:
                compute_clause_sat()

    compute_clause_sat()
    propagate()

    # Check if solved
    if all(clause_sat[ci] > 0 for ci in range(m)):
        return [a if a is not None else True for a in assignment]

    # Initialize unassigned variables randomly
    rng = random.Random(42)
    for v in range(n):
        if assignment[v] is None:
            assignment[v] = rng.random() < 0.5

    # Convert to numpy for fast operations
    assignment_np = np.array([1 if a else 0 for a in assignment], dtype=np.int8)

    def recompute_clause_sat():
        for ci in range(m):
            sc = 0
            for k in range(len(clause_vars[ci])):
                v = clause_vars[ci][k]
                s = clause_signs[ci][k]
                if (s == 1 and assignment_np[v] == 1) or (s == -1 and assignment_np[v] == 0):
                    sc += 1
            clause_sat[ci] = sc

    recompute_clause_sat()
    unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]

    if not unsat_clauses:
        return [bool(a) for a in assignment]

    best_unsat = len(unsat_clauses)
    best_assignment = assignment[:]

    max_restarts = 2000

    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break

        # Adaptive step budget based on remaining time
        remaining = deadline - time.time()
        if remaining < 10:
            break
        steps_per_restart = max(200, int(remaining * 150))

        # Initialize or perturb
        if restart == 0:
            pass  # already initialized
        elif restart % 7 == 0:
            # Perturb from best
            assignment_np = np.array([1 if a else 0 for a in best_assignment], dtype=np.int8)
            for _ in range(min(n // 4, 12)):
                v = rng.randint(0, n - 1)
                assignment_np[v] = 1 - assignment_np[v]
            assignment = [bool(a) for a in assignment_np]
            recompute_clause_sat()
        else:
            # Random restart
            assignment_np = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int8)
            assignment = [bool(a) for a in assignment_np]
            recompute_clause_sat()

        unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]
        unsat_count = len(unsat_clauses)

        if unsat_count == 0:
            return [bool(a) for a in assignment]

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        # Adaptive greedy probability
        p_greedy = 0.45 + 0.25 * (1.0 - unsat_count / max(m, 1))

        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break

            if not unsat_clauses:
                return [bool(a) for a in assignment]

            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment[:]

            # Pick a random unsat clause
            c_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            cd_vars = clause_vars[c_idx]
            cd_signs = clause_signs[c_idx]

            # Compute gains for each variable in the clause
            best_gain = -100
            best_vars = []
            for k in range(len(cd_vars)):
                v = cd_vars[k]
                s = cd_signs[k]
                gain = 0
                for (ci2, s2) in var_occ[v]:
                    if s2 == 1:
                        lit_true = (assignment_np[v] == 1)
                    else:
                        lit_true = (assignment_np[v] == 0)
                    if lit_true:
                        if clause_sat[ci2] == 1:
                            gain -= 1
                    else:
                        if clause_sat[ci2] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_vars = [v]
                elif gain == best_gain:
                    best_vars.append(v)

            # Choose variable to flip
            if rng.random() < p_greedy:
                v_to_flip = best_vars[rng.randint(0, len(best_vars) - 1)]
            else:
                v_to_flip = cd_vars[rng.randint(0, len(cd_vars) - 1)]

            # Flip the variable
            assignment_np[v_to_flip] = 1 - assignment_np[v_to_flip]
            assignment[v_to_flip] = bool(assignment_np[v_to_flip])

            # Update clause_sat for all clauses containing v_to_flip
            newly_unsat = []
            for (ci2, s2) in var_occ[v_to_flip]:
                old_sat = clause_sat[ci2]
                # Recompute this clause's sat count
                sc = 0
                for k in range(len(clause_vars[ci2])):
                    v2 = clause_vars[ci2][k]
                    s3 = clause_signs[ci2][k]
                    if (s3 == 1 and assignment_np[v2] == 1) or (s3 == -1 and assignment_np[v2] == 0):
                        sc += 1
                clause_sat[ci2] = sc
                if sc == 0 and old_sat > 0:
                    newly_unsat.append(ci2)

            # Rebuild unsat_clauses: remove those now satisfied, add newly unsat
            unsat_clauses = [ci for ci in unsat_clauses if clause_sat[ci] == 0]
            unsat_clauses.extend(newly_unsat)

    return best_assignment
