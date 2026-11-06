import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Precompute: for each clause, store literals as (var_idx, sign) where sign=+1 or -1
    # Also build var_to_clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    clause_vars = []  # list of lists of (var_idx, sign)
    for i, c in enumerate(clauses):
        var_list = []
        for l in c:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            var_list.append((v, s))
            var_to_clauses[v].append(i)
        clause_vars.append(var_list)
    
    # For fast gain computation: precompute for each variable, the list of (clause_idx, sign)
    var_clause_sigs = [[] for _ in range(n)]
    for v in range(n):
        for ci in var_to_clauses[v]:
            for (vv, ss) in clause_vars[ci]:
                if vv == v:
                    var_clause_sigs[v].append((ci, ss))
                    break
    
    # Precompute clause literal arrays for vectorized satisfaction checking
    clause_lit_vars = np.zeros((m, 3), dtype=np.int32)
    clause_lit_sigs = np.zeros((m, 3), dtype=np.int8)
    clause_lens = np.zeros(m, dtype=np.int32)
    
    for i in range(m):
        for j, (v, s) in enumerate(clause_vars[i]):
            clause_lit_vars[i, j] = v
            clause_lit_sigs[i, j] = s
        clause_lens[i] = len(clause_vars[i])
    
    # Precompute for each clause, the other literals (for checking if a clause has another satisfied literal)
    # For clause i and position j, store the other two (var, sign) pairs
    clause_other_lits = []
    for i in range(m):
        others = []
        for j in range(3):
            o1 = (clause_lit_vars[i, (j+1) % 3], clause_lit_sigs[i, (j+1) % 3])
            o2 = (clause_lit_vars[i, (j+2) % 3], clause_lit_sigs[i, (j+2) % 3])
            others.append((o1, o2))
        clause_other_lits.append(others)
    
    rng = np.random.RandomState(42)
    
    best_assign = None
    best_unsat = float('inf')
    
    # Clause weights for weighted WalkSAT
    clause_weights = np.ones(m, dtype=np.float64)
    
    max_restarts = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize random assignment as int array: 0 or 1
        assign = rng.randint(0, 2, size=n).astype(np.int8)
        
        # Compute clause satisfaction using vectorized operations
        all_vars = clause_lit_vars.ravel()
        all_sigs = clause_lit_sigs.ravel()
        
        assign_vals = assign[all_vars]
        lit_sat_all = (all_sigs * (2 * assign_vals.astype(np.int32) - 1)) > 0
        
        lit_sat_2d = lit_sat_all.reshape(m, 3)
        clause_sat = lit_sat_2d.any(axis=1)
        
        unsat_count = int(np.sum(~clause_sat))
        
        if unsat_count == 0:
            return assign.tolist()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assign.copy()
        
        # Adaptive walk probability
        walk_prob = 0.3
        no_improve_steps = 0
        last_unsat = unsat_count
        step = 0
        max_steps = 30000
        
        while step < max_steps:
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assign.tolist()
            
            # Select weighted random unsatisfied clause
            unsat_mask = ~clause_sat
            if not unsat_mask.any():
                return assign.tolist()
            
            # Weighted selection based on clause weights
            unsat_indices = np.where(unsat_mask)[0]
            w = clause_weights[unsat_indices]
            w = w / w.sum()
            c_idx = int(unsat_indices[rng.choice(len(unsat_indices), p=w)])
            
            c_vars = clause_vars[c_idx]
            
            # Choose variable to flip
            if rng.random() < walk_prob:
                # Random walk step
                var_idx, sign = c_vars[int(rng.randint(0, len(c_vars)))]
            else:
                # Weighted greedy: pick var in c that minimizes weighted gain
                best_gain = float('inf')
                best_var = None
                
                for (v, s) in c_vars:
                    old_val = int(assign[v])
                    new_val = 1 - old_val
                    
                    gain = 0.0
                    for (ci, cs) in var_clause_sigs[v]:
                        if clause_sat[ci]:
                            # Currently satisfied: becomes unsat only if v's lit was the only sat lit
                            old_lit_sat = cs * (2 * old_val - 1) > 0
                            new_lit_sat = cs * (2 * new_val - 1) > 0
                            if old_lit_sat and not new_lit_sat:
                                # Check if any other lit in ci is satisfied
                                others_sat = False
                                for (ov, os) in clause_vars[ci]:
                                    if ov != v:
                                        if os * (2 * int(assign[ov]) - 1) > 0:
                                            others_sat = True
                                            break
                                if not others_sat:
                                    gain += clause_weights[ci]
                        else:
                            # Currently unsat: becomes sat if v's new lit is satisfied
                            new_lit_sat = cs * (2 * new_val - 1) > 0
                            if new_lit_sat:
                                gain -= clause_weights[ci]
                    
                    if gain < best_gain:
                        best_gain = gain
                        best_var = v
                
                if best_var is not None:
                    var_idx = best_var
                else:
                    var_idx, sign = c_vars[int(rng.randint(0, len(c_vars)))]
            
            # Perform flip
            old_val = int(assign[var_idx])
            assign[var_idx] = 1 - old_val
            
            # Update clause statuses for clauses containing var_idx
            for ci in var_to_clauses[var_idx]:
                old_sat = clause_sat[ci]
                c_len = clause_lens[ci]
                c_vars_arr = clause_lit_vars[ci, :c_len]
                c_sigs_arr = clause_lit_sigs[ci, :c_len]
                
                c_assign_vals = assign[c_vars_arr]
                c_lit_sat = (c_sigs_arr * (2 * c_assign_vals.astype(np.int32) - 1)) > 0
                new_sat = c_lit_sat.any()
                
                if old_sat and not new_sat:
                    unsat_count += 1
                elif not old_sat and new_sat:
                    unsat_count -= 1
                clause_sat[ci] = new_sat
            
            step += 1
            
            # Update clause weights: boost unsatisfied clauses
            # Decay all weights slightly, boost unsatisfied ones
            clause_weights *= 0.999
            unsat_mask = ~clause_sat
            clause_weights[unsat_mask] += 1.0
            
            # Adaptive walk probability
            if unsat_count < last_unsat:
                last_unsat = unsat_count
                no_improve_steps = 0
                walk_prob = max(0.1, walk_prob * 0.95)
            else:
                no_improve_steps += 1
                if no_improve_steps % 500 == 0:
                    walk_prob = min(0.8, walk_prob * 1.1)
            
            if no_improve_steps > 8000:
                break
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign.copy()
        
        # End of inner loop
    
    if best_assign is not None:
        return best_assign.tolist()
    return [False] * n
