import time
import random
import numpy as np

N = 192
MAX_SIZE = 48

def _precompute_conflicts():
    clauses = []
    var_to_clauses = [[] for _ in range(N)]
    
    for a in range(N - 3):
        va = a + 1
        for b in range(a + 1, N - 2):
            vb = b + 1
            s_ab = va + vb
            for c in range(b + 1, N - 1):
                vc = c + 1
                s_abc = s_ab + vc
                for d in range(c + 1, N):
                    vd = d + 1
                    if (vb + vc + vd) % va == 0 or \
                       (va + vc + vd) % vb == 0 or \
                       (s_ab + vd) % vc == 0 or \
                       s_abc % vd == 0:
                        clause_id = len(clauses)
                        clauses.append((a, b, c, d))
                        var_to_clauses[a].append(clause_id)
                        var_to_clauses[b].append(clause_id)
                        var_to_clauses[c].append(clause_id)
                        var_to_clauses[d].append(clause_id)
                        
    return clauses, var_to_clauses

def _local_search(clauses, var_to_clauses, deadline):
    best_size = 0
    best_set = []
    num_vars = N
    num_clauses = len(clauses)
    
    if num_clauses == 0:
        return list(range(1, N+1))[:MAX_SIZE]

    while time.time() < deadline:
        init_size = random.randint(30, 48)
        x = np.zeros(num_vars, dtype=bool)
        indices = np.random.choice(num_vars, size=init_size, replace=False)
        x[indices] = True
        
        true_count = np.zeros(num_clauses, dtype=np.int32)
        for i in range(num_clauses):
            a, b, c, d = clauses[i]
            true_count[i] = x[a] + x[b] + x[c] + x[d]
            
        unsat_count = np.sum(true_count == 4)
        current_size = np.sum(x)
        
        if unsat_count == 0 and current_size > best_size:
            best_size = current_size
            best_set = [i+1 for i in range(num_vars) if x[i]]
            
        steps = 0
        max_steps = 2000
        
        while steps < max_steps and time.time() < deadline:
            if unsat_count == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = [i+1 for i in range(num_vars) if x[i]]
                    
                improved = False
                zeros = np.where(x == False)[0]
                if len(zeros) > 0:
                    random.shuffle(zeros)
                    for v in zeros[:5]:
                        can_flip = True
                        for c_idx in var_to_clauses[v]:
                            if true_count[c_idx] == 3:
                                can_flip = False
                                break
                        if can_flip:
                            x[v] = True
                            current_size += 1
                            for c_idx in var_to_clauses[v]:
                                true_count[c_idx] += 1
                            improved = True
                            break
                            
                if not improved:
                    break
            else:
                unsat_indices = np.where(true_count == 4)[0]
                if len(unsat_indices) == 0:
                    break
                
                c_idx = unsat_indices[random.randint(0, len(unsat_indices)-1)]
                a, b, c, d = clauses[c_idx]
                vars_in_clause = [a, b, c, d]
                
                best_v = -1
                best_delta = float('inf')
                
                for v in vars_in_clause:
                    if x[v]:
                        delta = 0
                        for vc in var_to_clauses[v]:
                            if true_count[vc] == 4:
                                delta -= 1
                    else:
                        delta = 0
                        for vc in var_to_clauses[v]:
                            if true_count[vc] == 3:
                                delta += 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_v = v
                        
                if best_v == -1:
                    break
                    
                v = best_v
                if x[v]:
                    x[v] = False
                    current_size -= 1
                    for vc in var_to_clauses[v]:
                        true_count[vc] -= 1
                        if true_count[vc] == 3:
                            unsat_count -= 1
                else:
                    x[v] = True
                    current_size += 1
                    for vc in var_to_clauses[v]:
                        true_count[vc] += 1
                        if true_count[vc] == 4:
                            unsat_count += 1
                            
                steps += 1
                
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    clauses, var_to_clauses = _precompute_conflicts()
    
    best_set = _local_search(clauses, var_to_clauses, deadline)
    
    if not best_set:
        best_set = list(range(N, N - 10, -1))
        
    best_set = list(set(best_set))
    best_set = [x for x in best_set if 1 <= x <= N]
    if len(best_set) > MAX_SIZE:
        best_set = best_set[:MAX_SIZE]
        
    return best_set
