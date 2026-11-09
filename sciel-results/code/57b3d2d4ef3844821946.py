import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    # Precompute clause data
    clause_data = []  # list of lists of (var_idx, polarity)
    var_clauses = [[] for _ in range(n)]

    for i, c in enumerate(clauses):
        lit_data = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            lit_data.append((v, p))
            var_clauses[v].append((i, p))
        clause_data.append(lit_data)

    m = len(clauses)

    # Unit propagation preprocessing
    assignment = [None] * n
    clause_active = [True] * m
    unit_queue = []

    # Find unit clauses
    for i, c in enumerate(clauses):
        cnt = 0
        for l in c:
            if assignment[abs(l) - 1] is None:
                cnt += 1
        if cnt == 1:
            unit_queue.append(i)

    # Propagate units
    while unit_queue:
        ci = unit_queue.pop()
        if not clause_active[ci]:
            continue
        # Find the unassigned literal
        for l in clauses[ci]:
            v = abs(l) - 1
            if assignment[v] is None:
                assignment[v] = l > 0
                # Propagate to all clauses containing v
                for cj, p in var_clauses[v]:
                    if not clause_active[cj]:
                        continue
                    # Check if this clause is now satisfied or unit
                    cnt = 0
                    for l2 in clauses[cj]:
                        v2 = abs(l2) - 1
                        if assignment[v2] is None:
                            cnt += 1
                            # Check if this literal conflicts
                            if assignment[v] is not None and v2 == v:
                                if p != assignment[v]:
                                    clause_active[cj] = False
                                    break
                    if not clause_active[cj]:
                        continue
                    if cnt == 1:
                        unit_queue.append(cj)
                    break
                break

    # Check for contradiction
    for i in range(m):
        if not clause_active[i]:
            sat = False
            for l in clauses[i]:
                v = abs(l) - 1
                if assignment[v] is not None and assignment[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                pass

    # Build incidence matrix for spectral clustering
    incidence = np.zeros((n, n), dtype=np.int32)
    for c in clauses:
        vars_in_clause = [abs(l) - 1 for l in c]
        for i in range(len(vars_in_clause)):
            for j in range(i + 1, len(vars_in_clause)):
                v1, v2 = vars_in_clause[i], vars_in_clause[j]
                incidence[v1][v2] += 1
                incidence[v2][v1] += 1

    # Find connected components
    sparse_incidence = csr_matrix(incidence)
    n_components, component_labels = connected_components(sparse_incidence, directed=False)

    # Group variables by component
    comp_vars = [[] for _ in range(n_components)]
    for v in range(n):
        comp_vars[component_labels[v]].append(v)

    # Assign clauses to components
    comp_clauses = [[] for _ in range(n_components)]
    for i, c in enumerate(clauses):
        if not clause_active[i]:
            continue
        comps = set()
        for l in c:
            v = abs(l) - 1
            comps.add(component_labels[v])
        if len(comps) == 1:
            comp = list(comps)[0]
            comp_clauses[comp].append(i)
        else:
            for comp in comps:
                comp_clauses[comp].append(i)

    # Initialize assignment with random values for unassigned variables
    rng_init = random.Random(42)
    for v in range(n):
        if assignment[v] is None:
            assignment[v] = rng_init.random() < 0.5

    # Solve each component
    total_vars = sum(len(v) for v in comp_vars)
    for comp in range(n_components):
        if time.time() > deadline - 10:
            break

        comp_vars_list = comp_vars[comp]
        comp_clauses_list = comp_clauses[comp]

        if not comp_clauses_list:
            continue

        var_to_local = {v: i for i, v in enumerate(comp_vars_list)}
        local_n = len(comp_vars_list)
        local_m = len(comp_clauses_list)

        # Build local clause data
        local_clause_data = []
        local_var_clauses = [[] for _ in range(local_n)]

        for c_idx in comp_clauses_list:
            lit_data = []
            for v, p in clause_data[c_idx]:
                if v in var_to_local:
                    local_v = var_to_local[v]
                    lit_data.append((local_v, p))
                    local_var_clauses[local_v].append((len(local_clause_data), p))
            if lit_data:
                local_clause_data.append(lit_data)

        local_m = len(local_clause_data)
        if local_m == 0:
            continue

        # WalkSAT on this component with multiple strategies
        best_local_unsat = float('inf')
        best_local_assignment = None

        # Time budget for this component
        comp_time_budget = (deadline - time.time() - 5) * (len(comp_vars_list) / total_vars) if total_vars > 0 else 0
        comp_deadline = time.time() + comp_time_budget

        # Multiple restart strategies with different noise levels
        noise_levels = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        max_restarts = min(300, max(30, local_m * 2))
        steps_per_restart = min(150, max(20, local_m))

        for restart in range(max_restarts):
            if time.time() > comp_deadline:
                break

            rng = random.Random(restart * 1000 + comp * 7 + 13)

            # Choose noise level based on restart number
            noise_idx = restart % len(noise_levels)
            noise_prob = noise_levels[noise_idx]

            # Initialize local assignment
            local_assignment = [assignment[var_to_local[v]] if v in var_to_local else (rng.random() < 0.5) for v in comp_vars_list]

            # Compute initial state
            local_sat_count = [0] * local_m
            local_is_unsat = [False] * local_m
            local_unsat_list = []

            for c_idx in range(local_m):
                cnt = 0
                for v, p in local_clause_data[c_idx]:
                    if local_assignment[v] == p:
                        cnt += 1
                local_sat_count[c_idx] = cnt
                if cnt == 0:
                    local_is_unsat[c_idx] = True
                    local_unsat_list.append(c_idx)

            if not local_unsat_list:
                for v, local_v in var_to_local.items():
                    assignment[v] = local_assignment[local_v]
                best_local_unsat = 0
                best_local_assignment = local_assignment[:]
                break

            if len(local_unsat_list) < best_local_unsat:
                best_local_unsat = len(local_unsat_list)
                best_local_assignment = local_assignment[:]

            # Walk
            for step in range(steps_per_restart):
                if time.time() > comp_deadline:
                    break

                if not local_unsat_list:
                    for v, local_v in var_to_local.items():
                        assignment[v] = local_assignment[local_v]
                    best_local_unsat = 0
                    best_local_assignment = local_assignment[:]
                    break

                # Pick random unsat clause
                c_idx = local_unsat_list[rng.randrange(len(local_unsat_list))]
                if not local_is_unsat[c_idx]:
                    # Rebuild unsat list
                    local_unsat_list = [i for i in range(local_m) if local_is_unsat[i]]
                    if not local_unsat_list:
                        break
                    c_idx = local_unsat_list[rng.randrange(len(local_unsat_list))]

                # Evaluate flips
                best_var = None
                best_score = float('inf')

                for v, p in local_clause_data[c_idx]:
                    broken = 0
                    fixed = 0

                    for c2_idx, p2 in local_var_clauses[v]:
                        was_sat = (local_assignment[v] == p2)
                        if was_sat:
                            if local_sat_count[c2_idx] == 1:
                                broken += 1
                        else:
                            if local_sat_count[c2_idx] == 0:
                                fixed += 1

                    score = broken - fixed
                    if score < best_score:
                        best_score = score
                        best_var = v

                # Noise
                if rng.random() < noise_prob:
                    v_to_flip = local_clause_data[c_idx][rng.randrange(len(local_clause_data[c_idx]))][0]
                else:
                    v_to_flip = best_var

                # Apply flip
                old_val = local_assignment[v_to_flip]
                local_assignment[v_to_flip] = not old_val

                # Update state
                for c2_idx, p2 in local_var_clauses[v_to_flip]:
                    was_sat = (old_val == p2)
                    if was_sat:
                        local_sat_count[c2_idx] -= 1
                        if local_sat_count[c2_idx] == 0 and not local_is_unsat[c2_idx]:
                            local_is_unsat[c2_idx] = True
                            local_unsat_list.append(c2_idx)
                    else:
                        local_sat_count[c2_idx] += 1
                        if local_sat_count[c2_idx] == 1 and local_is_unsat[c2_idx]:
                            local_is_unsat[c2_idx] = False

                # Track best
                current_unsat = sum(1 for i in local_unsat_list if local_is_unsat[i])
                if current_unsat < best_local_unsat:
                    best_local_unsat = current_unsat
                    best_local_assignment = local_assignment[:]

        # Apply best local assignment
        if best_local_assignment is not None:
            for v, local_v in var_to_local.items():
                assignment[v] = best_local_assignment[local_v]

    # Final verification and global WalkSAT if needed
    unsatisfied = []
    for i, c in enumerate(clauses):
        if not clause_active[i]:
            continue
        cnt = 0
        for l in c:
            v = abs(l) - 1
            p = l > 0
            if assignment[v] == p:
                cnt += 1
        if cnt == 0:
            unsatisfied.append(i)

    if not unsatisfied:
        return assignment

    # Global WalkSAT to fix remaining issues
    remaining_time = deadline - time.time()
    if remaining_time < 5:
        return assignment

    global_sat_count = [0] * m
    global_is_unsat = [False] * m
    global_unsat_list = []

    for c_idx in range(m):
        if not clause_active[c_idx]:
            continue
        cnt = 0
        for v, p in clause_data[c_idx]:
            if assignment[v] == p:
                cnt += 1
        global_sat_count[c_idx] = cnt
        if cnt == 0:
            global_is_unsat[c_idx] = True
            global_unsat_list.append(c_idx)

    if not global_unsat_list:
        return assignment

    max_global_restarts = min(150, int(remaining_time / 0.1))
    steps_per_global = min(80, max(10, m // 5))

    for restart in range(max_global_restarts):
        if time.time() > deadline - 2:
            break

        rng = random.Random(restart * 1000 + 999)
        noise_prob = 0.05 + 0.4 * (restart % 6) / 5

        # Small perturbation
        for v in range(n):
            if rng.random() < 0.05:
                assignment[v] = not assignment[v]

        # Recompute
        global_sat_count = [0] * m
        global_is_unsat = [False] * m
        global_unsat_list = []

        for c_idx in range(m):
            if not clause_active[c_idx]:
                continue
            cnt = 0
            for v, p in clause_data[c_idx]:
                if assignment[v] == p:
                    cnt += 1
            global_sat_count[c_idx] = cnt
            if cnt == 0:
                global_is_unsat[c_idx] = True
                global_unsat_list.append(c_idx)

        if not global_unsat_list:
            return assignment

        for step in range(steps_per_global):
            if time.time() > deadline - 2:
                break

            if not global_unsat_list:
                return assignment

            c_idx = global_unsat_list[rng.randrange(len(global_unsat_list))]
            if not global_is_unsat[c_idx]:
                global_unsat_list = [i for i in range(m) if global_is_unsat[i]]
                if not global_unsat_list:
                    return assignment
                c_idx = global_unsat_list[rng.randrange(len(global_unsat_list))]

            best_var = None
            best_score = float('inf')

            for v, p in clause_data[c_idx]:
                broken = 0
                fixed = 0

                for c2_idx, p2 in var_clauses[v]:
                    was_sat = (assignment[v] == p2)
                    if was_sat:
                        if global_sat_count[c2_idx] == 1:
                            broken += 1
                    else:
                        if global_sat_count[c2_idx] == 0:
                            fixed += 1

                score = broken - fixed
                if score < best_score:
                    best_score = score
                    best_var = v

            if rng.random() < noise_prob:
                v_to_flip = clause_data[c_idx][rng.randrange(len(clause_data[c_idx]))][0]
            else:
                v_to_flip = best_var

            old_val = assignment[v_to_flip]
            assignment[v_to_flip] = not old_val

            for c2_idx, p2 in var_clauses[v_to_flip]:
                was_sat = (old_val == p2)
                if was_sat:
                    global_sat_count[c2_idx] -= 1
                    if global_sat_count[c2_idx] == 0 and not global_is_unsat[c2_idx]:
                        global_is_unsat[c2_idx] = True
                        global_unsat_list.append(c2_idx)
                else:
                    global_sat_count[c2_idx] += 1
                    if global_sat_count[c2_idx] == 1 and global_is_unsat[c2_idx]:
                        global_is_unsat[c2_idx] = False

    return assignment
