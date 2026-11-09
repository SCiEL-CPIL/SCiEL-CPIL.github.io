import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
    # Preprocessing: build clause data and variable-clause incidence
    clause_data = []  # list of lists of (var_idx, polarity)
    var_clauses = [[] for _ in range(n)]  # var -> list of (clause_idx, polarity)
    
    for i, c in enumerate(clauses):
        lit_data = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            lit_data.append((v, p))
            var_clauses[v].append((i, p))
        clause_data.append(lit_data)
    
    m = len(clauses)
    
    # Build variable incidence matrix for spectral clustering
    # A[v1][v2] = number of clauses containing both v1 and v2
    incidence = np.zeros((n, n), dtype=np.int32)
    for c in clauses:
        vars_in_clause = [abs(l) - 1 for l in c]
        for i in range(len(vars_in_clause)):
            for j in range(i + 1, len(vars_in_clause)):
                v1, v2 = vars_in_clause[i], vars_in_clause[j]
                incidence[v1][v2] += 1
                incidence[v2][v1] += 1
    
    # Find connected components using sparse graph
    sparse_incidence = csr_matrix(incidence)
    n_components, component_labels = connected_components(sparse_incidence, directed=False)
    
    # Group variables and clauses by component
    comp_vars = [[] for _ in range(n_components)]
    for v in range(n):
        comp_vars[component_labels[v]].append(v)
    
    # Assign clauses to components (a clause belongs to all components it touches)
    # For solving, we'll solve each component independently
    # If a clause spans multiple components, we handle it specially
    
    # Check if problem is already decomposed
    comp_clauses = [[] for _ in range(n_components)]
    for i, c in enumerate(clauses):
        comps = set()
        for l in c:
            v = abs(l) - 1
            comps.add(component_labels[v])
        if len(comps) == 1:
            comp = list(comps)[0]
            comp_clauses[comp].append(i)
        else:
            # Cross-component clause: assign to the component with most variables
            # This is a simplification; in practice, such clauses may indicate
            # the components are actually coupled
            for comp in comps:
                comp_clauses[comp].append(i)
    
    # Solve each component independently
    assignment = [False] * n
    
    # Time budget per component proportional to its size
    total_vars = sum(len(v) for v in comp_vars)
    time_budgets = []
    for comp in range(n_components):
        budget = (deadline - start_time - 5) * (len(comp_vars[comp]) / total_vars) if total_vars > 0 else 0
        time_budgets.append(budget)
    
    for comp in range(n_components):
        if time.time() > deadline - 5:
            break
        
        comp_vars_list = comp_vars[comp]
        comp_clauses_list = comp_clauses[comp]
        
        if not comp_clauses_list:
            continue
        
        # Local indices
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
            if lit_data:  # Only include clauses that have variables in this component
                local_clause_data.append(lit_data)
        
        local_m = len(local_clause_data)
        
        if local_m == 0:
            continue
        
        # WalkSAT on this component
        best_local_unsat = float('inf')
        best_local_assignment = None
        
        max_restarts = min(500, max(50, local_m * 2))
        steps_per_restart = min(200, max(20, local_m))
        
        comp_deadline = time.time() + time_budgets[comp]
        
        for restart in range(max_restarts):
            if time.time() > comp_deadline:
                break
            
            rng = random.Random(restart * 1000 + comp * 7 + 13)
            
            local_assignment = [rng.random() < 0.5 for _ in range(local_n)]
            
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
                # Found solution
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
                c_idx = None
                tries = 0
                while c_idx is None:
                    idx = local_unsat_list[rng.randrange(len(local_unsat_list))]
                    if local_is_unsat[idx]:
                        c_idx = idx
                    else:
                        tries += 1
                        if tries > len(local_unsat_list):
                            local_unsat_list = [i for i in local_unsat_list if local_is_unsat[i]]
                            if not local_unsat_list:
                                break
                            idx = local_unsat_list[rng.randrange(len(local_unsat_list))]
                            if local_is_unsat[idx]:
                                c_idx = idx
                            else:
                                c_idx = idx
                
                if c_idx is None:
                    break
                
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
                if rng.random() < 0.15:
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
    
    # Final check: verify if the assignment satisfies all clauses
    # If not, do a global WalkSAT pass on remaining unsatisfied clauses
    unsatisfied = []
    for i, c in enumerate(clauses):
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
    
    # Build global structures
    global_sat_count = [0] * m
    global_is_unsat = [False] * m
    global_unsat_list = []
    
    for c_idx in range(m):
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
    
    max_global_restarts = min(200, int(remaining_time / 0.1))
    steps_per_global = min(100, max(10, m // 5))
    
    for restart in range(max_global_restarts):
        if time.time() > deadline - 2:
            break
        
        rng = random.Random(restart * 1000 + 999)
        
        # Start from current assignment, add small random perturbation
        for v in range(n):
            if rng.random() < 0.1:
                assignment[v] = not assignment[v]
        
        # Recompute
        global_sat_count = [0] * m
        global_is_unsat = [False] * m
        global_unsat_list = []
        
        for c_idx in range(m):
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
            
            c_idx = None
            tries = 0
            while c_idx is None:
                idx = global_unsat_list[rng.randrange(len(global_unsat_list))]
                if global_is_unsat[idx]:
                    c_idx = idx
                else:
                    tries += 1
                    if tries > len(global_unsat_list):
                        global_unsat_list = [i for i in global_unsat_list if global_is_unsat[i]]
                        if not global_unsat_list:
                            return assignment
                        idx = global_unsat_list[rng.randrange(len(global_unsat_list))]
                        if global_is_unsat[idx]:
                            c_idx = idx
                        else:
                            c_idx = idx
            
            if c_idx is None:
                break
            
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
            
            if rng.random() < 0.1:
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
