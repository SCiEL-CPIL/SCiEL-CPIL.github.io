import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Greedy Heuristic
    # Sort facilities by flow sum descending
    flow_sums = a.sum(axis=1)
    fac_order = np.argsort(-flow_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p.fill(-1)
    assigned = np.zeros(n, dtype=bool)
    
    # Precompute b diagonal for self terms
    b_diag = np.diag(b)
    
    for idx in fac_order:
        if np.sum(assigned) == 0:
            # First facility: minimize a[idx, idx] * b[v, v]
            costs = a[idx, idx] * b_diag
        else:
            assigned_idx = np.where(assigned)[0]
            # V1[v] = sum_j a[idx, j] * b[v, p[j]]
            # b[:, p[assigned_idx]] is (n, k)
            # a[idx, assigned_idx] is (k,)
            # V1 = b[:, p[assigned_idx]] @ a[idx, assigned_idx]
            V1 = b[:, p[assigned_idx]] @ a[idx, assigned_idx]
            
            # V2[v] = sum_j a[j, idx] * b[p[j], v]
            # b[p[assigned_idx], :] is (k, n)
            # a[assigned_idx, idx] is (k,)
            # V2 = (b[p[assigned_idx], :] @ a[assigned_idx, idx])
            V2 = b[p[assigned_idx], :] @ a[assigned_idx, idx]
            
            # Cost to place idx at v:
            # Sum_{j != idx} (a[idx,j]b[v,p[j]] + a[j,idx]b[p[j],v]) + a[idx,idx]b[v,v]
            # Note: V1 and V2 include j=idx? No, assigned_idx does not contain idx yet.
            # So V1 and V2 are sums over j != idx.
            # We just add the self term.
            costs = V1 + V2 + a[idx, idx] * b_diag
            
        best_v = np.argmin(costs)
        p[idx] = best_v
        assigned[idx] = True

    # Calculate initial cost
    B_perm = b[np.ix_(p, p)]
    cost = np.sum(a * B_perm)
    
    best_p = p.copy()
    best_cost = cost
    
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead
    
    # Iterated Local Search
    while time.time() - start_time < time_limit:
        improved = True
        # Local Search Phase: 2-opt and Relocation
        while improved:
            improved = False
            
            # 1. Try Swaps (2-opt)
            for i in range(n):
                for k in range(i + 1, n):
                    u = p[i]
                    v = p[k]
                    
                    diff_a = a[i] - a[k]
                    diff_b = b[v, p] - b[u, p]
                    S = np.dot(diff_a, diff_b)
                    
                    S -= (a[i, i] - a[k, i]) * (b[v, u] - b[u, u])
                    S -= (a[i, k] - a[k, k]) * (b[v, v] - b[u, v])
                    
                    delta = S + (a[i, k] * (b[v, u] - b[u, v]) + 
                                 a[k, i] * (b[u, v] - b[v, u]) + 
                                 a[i, i] * (b[v, v] - b[u, u]) + 
                                 a[k, k] * (b[u, u] - b[v, v]))
                    
                    if delta < 0:
                        p[i], p[k] = p[k], p[i]
                        cost += delta
                        improved = True
                        if cost < best_cost:
                            best_cost = cost
                            best_p = p.copy()
            
            # 2. If no swap improved, try Relocations (Or-opt 1)
            if not improved:
                for i in range(n):
                    u = p[i]
                    # Compute costs for all v
                    # V1[v] = sum_j a[i, j] * b[v, p[j]]
                    # V2[v] = sum_j a[j, i] * b[p[j], v]
                    # We need to exclude j=i from the sums?
                    # If we use full vectors:
                    # V1_full = b[:, p] @ a[i, :]
                    # V2_full = b[p, :].T @ a[:, i]
                    # V1_full includes j=i: a[i,i] * b[v, p[i]] = a[i,i] * b[v, u]
                    # V2_full includes j=i: a[i,i] * b[p[i], v] = a[i,i] * b[u, v]
                    
                    V1 = b[:, p] @ a[i, :]
                    V2 = b[p, :].T @ a[:, i]
                    
                    # New cost if i moves to v:
                    # Sum_{j != i} (a[i,j]b[v,p[j]] + a[j,i]b[p[j],v]) + a[i,i]b[v,v]
                    # = (V1[v] - a[i,i]b[v,u]) + (V2[v] - a[i,i]b[u,v]) + a[i,i]b[v,v]
                    
                    # Old cost contribution of i (to be subtracted from total? No, we compare new total vs old total)
                    # Old cost = cost
                    # New cost = cost - Old_i + New_i
                    # Old_i = Sum_{j != i} (a[i,j]b[u,p[j]] + a[j,i]b[p[j],u]) + a[i,i]b[u,u]
                    # Old_i = (V1[u] - a[i,i]b[u,u]) + (V2[u] - a[i,i]b[u,u]) + a[i,i]b[u,u]
                    # Old_i = V1[u] + V2[u] - a[i,i]b[u,u]
                    
                    # New_i(v) = V1[v] - a[i,i]b[v,u] + V2[v] - a[i,i]b[u,v] + a[i,i]b[v,v]
                    
                    # Delta(v) = New_i(v) - Old_i
                    # Delta(v) = V1[v] + V2[v] - a[i,i](b[v,u] + b[u,v] - b[v,v]) - (V1[u] + V2[u] - a[i,i]b[u,u])
                    
                    base = V1 + V2
                    correction = a[i, i] * (b[v, u] + b[u, v] - b[v, v]) # Vector op?
                    # b[:, u] is column u. b[u, :] is row u.
                    # correction[v] = a[i,i] * (b[v,u] + b[u,v] - b[v,v])
                    corr_vec = a[i, i] * (b[:, u] + b[u, :] - b_diag)
                    
                    new_costs = base - corr_vec
                    old_cost_i = V1[u] + V2[u] - a[i, i] * b[u, u]
                    
                    # Find best v
                    best_v = np.argmin(new_costs)
                    best_new_cost_i = new_costs[best_v]
                    
                    if best_new_cost_i < old_cost_i:
                        # Apply move
                        delta = best_new_cost_i - old_cost_i
                        p[i] = best_v
                        cost += delta
                        improved = True
                        if cost < best_cost:
                            best_cost = cost
                            best_p = p.copy()

        # Perturbation Phase
        for _ in range(5):
            i = random.randint(0, n - 1)
            k = random.randint(0, n - 1)
            if i != k:
                p[i], p[k] = p[k], p[i]
        
        # Recalculate cost
        B_perm = b[np.ix_(p, p)]
        cost = np.sum(a * B_perm)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    return best_p.tolist()
