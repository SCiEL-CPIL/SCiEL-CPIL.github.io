import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    start_time = time.time()
    deadline = start_time + 300.0
    safety_margin = 2.0

    # Pre-compute variable-clause incidence
    var_to_clauses = [[] for _ in range(n)]
    clause_vars = []
    clause_pols = []

    for i, c in enumerate(clauses):
        cvars = []
        cpols = []
        for lit in c:
            v = abs(lit) - 1
            p = lit > 0
            var_to_clauses[v].append((i, p))
            cvars.append(v)
            cpols.append(p)
        clause_vars.append(cvars)
        clause_pols.append(cpols)

    # Use numpy for fast operations
    # sat_count[i] = number of satisfying literals in clause i
    # assignment[v] = 1 if True, 0 if False

    def init_state(assignment_arr):
        # assignment_arr is numpy array of 0s and 1s
        sat_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            for j, v in enumerate(clause_vars[i]):
                p = clause_pols[i][j]
                if (assignment_arr[v] == 1) == p:
                    sat_count[i] += 1
        return sat_count

    def count_unsat(sat_count):
        return int(np.sum(sat_count == 0))

    # Main solver using WalkSAT-style with adaptive features
    rng = random.Random()

    best_unsat = m + 1
    best_assignment = None

    # Parameters
    noise_prob = 0.25
    max_steps_per_restart = 50000
    weight_boost = 1.05
    weight_decay = 0.98
    max_weight = 50.0
    min_weight = 0.1

    while time.time() < deadline - safety_margin:
        # Random initialization
        assignment = np.array([1 if rng.random() < 0.5 else 0 for _ in range(n)], dtype=np.int32)
        sat_count = init_state(assignment)
        
        current_unsat = count_unsat(sat_count)
        if current_unsat == 0:
            return [bool(x) for x in assignment]
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Clause weights
        clause_weights = np.ones(m, dtype=np.float64)

        steps = 0
        stuck_counter = 0
        local_best_unsat = current_unsat
        local_best_assignment = assignment.copy()
        stagnation_counter = 0

        while steps < max_steps_per_restart:
            if time.time() > deadline - safety_margin:
                break

            if current_unsat == 0:
                return [bool(x) for x in assignment]

            # Pick an unsatisfied clause
            # Find unsat clauses
            unsat_indices = np.where(sat_count == 0)[0]
            if len(unsat_indices) == 0:
                return [bool(x) for x in assignment]

            # Weighted random selection
            weights = clause_weights[unsat_indices]
            total_w = weights.sum()
            if total_w <= 0:
                ci = unsat_indices[rng.randrange(len(unsat_indices))]
            else:
                r = rng.random() * total_w
                cumulative = 0
                ci = unsat_indices[-1]
                for idx in range(len(unsat_indices)):
                    cumulative += weights[idx]
                    if r <= cumulative:
                        ci = unsat_indices[idx]
                        break

            # Evaluate variables in this clause
            cvars = clause_vars[ci]
            cpols = clause_pols[ci]

            best_v = -1
            best_score = -10**9

            for v in cvars:
                # Calculate gain and breakage
                gain = 0.0
                breakage = 0.0
                for uci, p in var_to_clauses[v]:
                    w = clause_weights[uci]
                    if sat_count[uci] == 0:
                        # Clause is unsat. If flipping v satisfies it, gain++
                        if (1 - assignment[v]) == (1 if p else 0):
                            gain += w
                    else:
                        # Clause is sat. If flipping v breaks it, breakage++
                        if sat_count[uci] == 1:
                            if assignment[v] == (1 if p else 0):
                                breakage += w

                score = gain - breakage
                if score > best_score:
                    best_score = score
                    best_v = v

            # Noise: with probability noise_prob, pick random var from clause
            if rng.random() < noise_prob:
                v = cvars[rng.randrange(len(cvars))]
            else:
                v = best_v

            # Flip variable v
            assignment[v] = 1 - assignment[v]

            # Update sat_count for all clauses containing v
            for uci, p in var_to_clauses[v]:
                old_count = sat_count[uci]
                if (assignment[v] == 1) == p:
                    # v is now satisfying
                    if old_count == 0:
                        sat_count[uci] = 1
                        current_unsat -= 1
                    else:
                        sat_count[uci] = old_count + 1
                else:
                    # v is now not satisfying
                    if old_count == 1:
                        sat_count[uci] = 0
                        current_unsat += 1
                    else:
                        sat_count[uci] = old_count - 1

            steps += 1

            # Periodic checks and weight updates
            if steps % 100 == 0:
                if current_unsat < local_best_unsat:
                    local_best_unsat = current_unsat
                    local_best_assignment = assignment.copy()
                    stuck_counter = 0
                    stagnation_counter = 0
                else:
                    stuck_counter += 100
                    stagnation_counter += 100

                # Boost weights of unsatisfied clauses when stagnating
                if stagnation_counter > 500:
                    for i in range(m):
                        if sat_count[i] == 0:
                            clause_weights[i] = min(clause_weights[i] * weight_boost, max_weight)
                        else:
                            clause_weights[i] = max(clause_weights[i] * weight_decay, min_weight)

                if stuck_counter > 10000:
                    break  # Restart

        # End of local search
        if local_best_unsat < best_unsat:
            best_unsat = local_best_unsat
            best_assignment = local_best_assignment.copy()

    if best_assignment is not None:
        return [bool(x) for x in best_assignment]
    return [bool(x) for x in assignment]
