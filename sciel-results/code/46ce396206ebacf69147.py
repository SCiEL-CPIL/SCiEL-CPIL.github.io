import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 40s margin
    
    # Preprocessing
    clause_data = []
    var_to_clauses = [[] for _ in range(n)]
    
    for c_idx, clause in enumerate(clauses):
        c_data = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            c_data.append((v, s))
            var_to_clauses[v].append(c_idx)
        clause_data.append(c_data)
        
    var_clause_signs = [[] for _ in range(n)]
    for c_idx, c_data in enumerate(clause_data):
        for v, s in c_data:
            var_clause_signs[v].append((c_idx, s))

    def compute_sat(x):
        sat_counts = np.zeros(m, dtype=int)
        for c_idx, c_data in enumerate(clause_data):
            count = 0
            for v, s in c_data:
                if (x[v] and s == 1) or (not x[v] and s == -1):
                    count += 1
            sat_counts[c_idx] = count
        return sat_counts

    rng = random.Random(42)
    
    max_restarts = 500
    base_steps_per_restart = 5000
    p_start = 0.8
    p_end = 0.5
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        frac = min(r / max_restarts, 1.0)
        p = p_start + (p_end - p_start) * frac
        
        x = [rng.random() < 0.5 for _ in range(n)]
        sat_counts = compute_sat(x)
        
        if np.all(sat_counts > 0):
            return x
            
        unsat_clauses_mask = (sat_counts == 0)
        unsat_var_count = np.zeros(n, dtype=int)
        for v in range(n):
            count = 0
            for c_idx, s in var_clause_signs[v]:
                if unsat_clauses_mask[c_idx]:
                    count += 1
            unsat_var_count[v] = count
            
        steps_per_restart = int(base_steps_per_restart * (1.0 + 0.5 * frac))
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return x
                
            unsat_indices = np.where(sat_counts == 0)[0]
            if len(unsat_indices) == 0:
                return x
                
            c_idx = rng.choice(unsat_indices)
            c_data = clause_data[c_idx]
            
            max_count = -1
            best_candidates = []
            for v, s in c_data:
                cnt = unsat_var_count[v]
                if cnt > max_count:
                    max_count = cnt
                    best_candidates = [v]
                elif cnt == max_count:
                    best_candidates.append(v)
            
            if rng.random() < p:
                v = rng.choice(best_candidates)
            else:
                v = rng.choice([cv for cv, cs in c_data])
            
            old_val = x[v]
            new_val = not old_val
            x[v] = new_val
            
            for c_idx2, s in var_clause_signs[v]:
                was_sat = (old_val and s == 1) or (not old_val and s == -1)
                is_sat = (new_val and s == 1) or (not new_val and s == -1)
                
                if was_sat and not is_sat:
                    sat_counts[c_idx2] -= 1
                    if sat_counts[c_idx2] == 0:
                        for v2, s2 in clause_data[c_idx2]:
                            unsat_var_count[v2] += 1
                elif not was_sat and is_sat:
                    sat_counts[c_idx2] += 1
                    if sat_counts[c_idx2] == 1:
                        for v2, s2 in clause_data[c_idx2]:
                            unsat_var_count[v2] -= 1
            
            # Polish: if <=2 unsat clauses, try targeted fixes
            if len(unsat_indices) <= 2 and step > steps_per_restart * 0.3:
                candidate_vars = set()
                for uc in unsat_indices:
                    for v3, s3 in clause_data[uc]:
                        candidate_vars.add(v3)
                
                for v2 in candidate_vars:
                    old_v2 = x[v2]
                    new_v2 = not old_v2
                    would_solve = True
                    for uc in unsat_indices:
                        uc_data = clause_data[uc]
                        sat_in_clause = False
                        for v3, s3 in uc_data:
                            if v3 == v2:
                                if (new_v2 and s3 == 1) or (not new_v2 and s3 == -1):
                                    sat_in_clause = True
                                    break
                            else:
                                if (x[v3] and s3 == 1) or (not x[v3] and s3 == -1):
                                    sat_in_clause = True
                                    break
                        if not sat_in_clause:
                            would_solve = False
                            break
                    if would_solve:
                        x[v2] = new_v2
                        sat_counts = compute_sat(x)
                        if np.all(sat_counts > 0):
                            return x
                        unsat_clauses_mask = (sat_counts == 0)
                        unsat_var_count = np.zeros(n, dtype=int)
                        for v3 in range(n):
                            count = 0
                            for c_idx3, s3 in var_clause_signs[v3]:
                                if unsat_clauses_mask[c_idx3]:
                                    count += 1
                            unsat_var_count[v3] = count
                        break

    return x
