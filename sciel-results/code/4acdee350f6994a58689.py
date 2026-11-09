import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

def construct():
    N = 51975
    # Find all divisors of N greater than 1
    divisors = []
    for i in range(3, N + 1, 2):
        if N % i == 0:
            divisors.append(i)
    
    divisors.sort()
    
    # 1. Greedy Initialization to get a baseline
    classes = []
    covered = np.zeros(N, dtype=np.int8)
    score = N
    used_moduli = set()
    available = set(divisors)
    
    while available:
        best_m = None
        best_r = None
        best_gain = -1
        
        for m in available:
            min_nonzero = N
            best_r_for_m = 0
            for r in range(m):
                nz = np.count_nonzero(covered[r::m])
                if nz < min_nonzero:
                    min_nonzero = nz
                    best_r_for_m = r
                    if min_nonzero == 0:
                        break
            
            total_len = (N - 1 - best_r_for_m) // m + 1
            gain = total_len - min_nonzero
            
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = best_r_for_m
        
        if best_gain <= 0:
            break
            
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        available.remove(best_m)
        
        covered[best_r::best_m] += 1
        score -= best_gain

    best_score = score
    best_classes = [c[:] for c in classes]
    
    # 2. ILP Optimization on a subset of small moduli
    K_ILP = 25
    candidate_moduli = divisors[:K_ILP]
    
    # Map modulus to index in candidate list
    m_to_idx = {m: i for i, m in enumerate(candidate_moduli)}
    
    # Create variable mapping: (m_idx, r) -> var_idx
    var_map = {}
    var_count = 0
    m_vars = [] # list of lists of var indices for each modulus
    
    for m_idx, m in enumerate(candidate_moduli):
        m_vars.append([])
        for r in range(m):
            var_map[(m_idx, r)] = var_count
            m_vars[m_idx].append(var_count)
            var_count += 1
    
    n_x = var_count
    n_y = N
    n_vars = n_x + n_y
    
    c = np.zeros(n_vars)
    c[n_x:] = -1.0
    
    # Build sparse matrix for constraints
    rows = []
    cols = []
    vals = []
    
    # 1. Coverage Constraints: y_k - sum(x_v) <= 0
    for (m_idx, r), v_idx in var_map.items():
        m = candidate_moduli[m_idx]
        # Use numpy to generate indices for speed?
        # idx_list = np.arange(r, N, m)
        # for k in idx_list: ...
        
        # Python loop is okay for ~4-5M iterations
        for k in range(r, N, m):
            rows.append(k)
            cols.append(v_idx)
            vals.append(-1)
            
            rows.append(k)
            cols.append(n_x + k)
            vals.append(1)
            
    # 2. Modulus Constraints: sum(x_v) <= 1
    n_cov_rows = N
    for m_idx in range(len(candidate_moduli)):
        row_id = n_cov_rows + m_idx
        for v_idx in m_vars[m_idx]:
            rows.append(row_id)
            cols.append(v_idx)
            vals.append(1)
            
    A = coo_matrix((vals, (rows, cols)), shape=(n_cov_rows + len(candidate_moduli), n_vars)).tocsr()
    
    b = np.zeros(n_cov_rows + len(candidate_moduli))
    b[n_cov_rows:] = 1.0
    
    constraints = [LinearConstraint(A, -np.inf, b)]
    
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    integrality = np.ones(n_vars)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0:
            x_sol = res.x[:n_x]
            selected_classes = []
            for m_idx, m in enumerate(candidate_moduli):
                for r, v_idx in enumerate(m_vars[m_idx]):
                    if x_sol[v_idx] > 0.5:
                        selected_classes.append([m, r])
            
            curr_covered = np.zeros(N, dtype=np.int8)
            for m, r in selected_classes:
                curr_covered[r::m] += 1
            curr_score = np.count_nonzero(curr_covered == 0)
            
            if curr_score < best_score:
                best_score = curr_score
                best_classes = [c[:] for c in selected_classes]
                
    except Exception:
        pass
        
    # 3. Local Search
    curr_classes = [c[:] for c in best_classes]
    curr_covered = np.zeros(N, dtype=np.int8)
    for m, r in curr_classes:
        curr_covered[r::m] += 1
    curr_score = np.count_nonzero(curr_covered == 0)
    
    start_time = time.time()
    deadline = start_time + 110
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for i in range(len(curr_classes)):
            if time.time() > deadline:
                break
                
            m, r = curr_classes[i]
            
            best_delta = 0
            best_new_r = r
            
            one_count_current = np.count_nonzero(curr_covered[r::m] == 1)
            
            min_delta = 0
            best_r_new = r
            
            for r_new in range(m):
                if r_new == r:
                    continue
                gained = np.count_nonzero(curr_covered[r_new::m] == 0)
                delta = one_count_current - gained
                if delta < min_delta:
                    min_delta = delta
                    best_r_new = r_new
            
            if min_delta < 0:
                curr_covered[r::m] -= 1
                curr_covered[best_r_new::m] += 1
                curr_classes[i][1] = best_r_new
                curr_score += min_delta
                improved = True
                
                if curr_score < best_score:
                    best_score = curr_score
                    best_classes = [c[:] for c in curr_classes]
                    
    # 4. Add unused moduli
    if time.time() < deadline - 5:
        used_moduli = set([c[0] for c in curr_classes])
        unused = [m for m in divisors if m not in used_moduli]
        
        for _ in range(10):
            if time.time() > deadline:
                break
            if not unused:
                break
            
            best_m_new = None
            best_r_new = None
            best_gain = -1
            
            for m_new in unused:
                min_nonzero = N
                best_r_for_m = 0
                for r_new in range(m_new):
                    nz = np.count_nonzero(curr_covered[r_new::m_new])
                    if nz < min_nonzero:
                        min_nonzero = nz
                        best_r_for_m = r_new
                        if min_nonzero == 0:
                            break
                
                total_len = (N - 1 - best_r_for_m) // m_new + 1
                gain = total_len - min_nonzero
                
                if gain > best_gain:
                    best_gain = gain
                    best_m_new = m_new
                    best_r_new = best_r_for_m
            
            if best_gain > 0:
                curr_covered[best_r_new::best_m_new] += 1
                curr_classes.append([best_m_new, best_r_new])
                curr_score -= best_gain
                if curr_score < best_score:
                    best_score = curr_score
                    best_classes = [c[:] for c in curr_classes]
            else:
                break

    return best_classes
