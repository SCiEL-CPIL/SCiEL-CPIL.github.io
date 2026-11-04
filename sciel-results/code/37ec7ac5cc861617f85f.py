"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    # --- Greedy construction for a good initial solution
    flow_sums = a.sum(axis=1) + a.sum(axis=0)
    order = np.argsort(-flow_sums, kind="stable")
    p = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    for idx in range(n):
        f = order[idx]
        best_loc = -1
        best_cost = np.iinfo(np.int64).max
        for loc in range(n):
            if used[loc]:
                continue
            inc = 0
            for k in range(idx):
                g = order[k]
                loc_g = p[g]
                inc += a[f, g] * b[loc, loc_g] + a[g, f] * b[loc_g, loc]
            if inc < best_cost:
                best_cost = inc
                best_loc = loc
        p[f] = best_loc
        used[best_loc] = True

    def total_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    current_cost = total_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    def local_search(p, cost):
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Evaluate all pairs to find best improving move
            for i in range(n):
                u = p[i]
                for j in range(i + 1, n):
                    v = p[j]
                    
                    # Compute delta
                    diff_row = a[i, :] - a[j, :]
                    diff_col = a[:, i] - a[:, j]
                    
                    bv_p = b[v, p]
                    bu_p = b[u, p]
                    bp_v = b[p, v]
                    bp_u = b[p, u]
                    
                    s1 = np.dot(diff_row, bv_p - bu_p)
                    s2 = np.dot(diff_col, bp_v - bp_u)
                    
                    # Subtract k=i and k=j terms
                    s1 -= diff_row[i] * (b[v, u] - b[u, u])
                    s2 -= diff_col[i] * (b[u, v] - b[u, u])
                    
                    s1 -= diff_row[j] * (b[v, v] - b[u, v])
                    s2 -= diff_col[j] * (b[v, v] - b[v, u])
                    
                    direct = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
                    
                    delta = s1 + s2 + direct
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                # Apply move
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
                
        return p, cost

    rng = np.random.default_rng(42)

    def perturb(p, k):
        p_new = p.copy()
        for _ in range(k):
            i = rng.integers(n)
            j = rng.integers(n)
            if i != j:
                p_new[i], p_new[j] = p_new[j], p_new[i]
        return p_new

    t0 = time.time()
    time_limit = 110.0
    
    T = 5000.0
    T_min = 0.5
    
    while time.time() - t0 < time_limit:
        # 1. Local Search (Best Improvement 2-opt)
        p_ls, cost_ls = local_search(p, current_cost)
        
        # 2. Update Best
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
        # 3. Perturbation
        # Fixed perturbation strength
        k_perturb = 5
        p_new = perturb(p_ls, k_perturb)
        cost_new = total_cost(p_new)
        
        # 4. Acceptance (SA-like)
        delta = cost_new - cost_ls
        if delta < 0:
            accept = True
        else:
            if T > 1e-9:
                ratio = math.exp(-delta / T)
                accept = rng.random() < ratio
            else:
                accept = False
        
        if accept:
            p = p_new
            current_cost = cost_new
        else:
            p = p_ls
            current_cost = cost_ls
            
        # 5. Cooling
        T *= 0.995
        if T < T_min:
            T = T_min

    return best_p.tolist()
# EVOLVE-BLOCK-END
