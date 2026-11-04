"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Greedy Rank Matching
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_sum, kind="stable")
    # Sort locations by ascending distance sum
    central_order = np.argsort(b_sum, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # Calculate initial cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # 2. Precomputation for Fast Delta Calculation
    # We maintain 'p' (facility -> location) and 'inv_p' (location -> facility).
    # We maintain 'RC' (Row Cost) and 'CC' (Col Cost) for the current assignment.
    # RC[i] = sum_{k != i} a[i, k] * b[p[i], p[k]]
    # CC[i] = sum_{k != i} a[k, i] * b[p[k], p[i]]
    # Note: For symmetric A and B, RC[i] == CC[i].
    # Let's assume symmetry for speed, but we can verify or use general if needed.
    # QAPLIB instances are symmetric.
    
    def compute_rc_cc(p):
        rc = np.zeros(n, dtype=np.int64)
        cc = np.zeros(n, dtype=np.int64)
        # Vectorized approach is tricky for individual elements, so we loop or use matrix ops.
        # b_perm = b[p][:, p]
        # rc[i] = dot(a[i, :], b_perm[i, :]) - a[i, i]*b[p[i], p[i]]
        # This is O(n^2) to compute from scratch.
        b_perm = b[p][:, p]
        # rc = a.dot(b_perm) - np.diag(a) * np.diag(b_perm)
        # cc = a.T.dot(b_perm) - np.diag(a) * np.diag(b_perm)
        # Since A, B symmetric, a.T = a, so rc == cc.
        # Let's just compute rc and use it for both.
        rc = np.sum(a * b_perm, axis=1) - np.diag(a) * np.diag(b_perm)
        return rc
        
    rc = compute_rc_cc(p)
    
    # 3. Simulated Annealing with Fast Delta
    T_start = 5000.0
    T_end = 1.0
    alpha = 0.9995
    
    current_T = T_start
    iter_count = 0
    
    # Helper to update RC after swap i, j
    # When swapping i and j (locations u, v):
    # 1. Update p[i], p[j]
    # 2. Update RC for i and j
    # 3. Update RC for all other k
    
    # Delta for swap i, j:
    # delta = (RC_new[i] + RC_new[j]) - (RC_old[i] + RC_old[j])
    # This is not quite right because RC depends on p.
    # Let's use the standard O(n) delta formula derived earlier, but optimized.
    # delta = 2 * sum_{k != i, j} (a[i, k] - a[j, k]) * (b[p[k], v] - b[p[k], u]) 
    #         + 2 * (a[i, j] - a[j, j]) * (b[v, u] - b[u, v])  # Wait, cross terms
    # Actually, the formula:
    # delta = 2 * dot(a[i, :] - a[j, :], b[:, v] - b[:, u]) - 2 * (a[i, j] + a[j, i]) * (b[v, u] - b[u, v]) ?
    # Let's stick to the robust one:
    # delta = 2 * ( (a[i, :] - a[j, :]) dot (b[:, v] - b[:, u]) ) 
    #         - 4 * (a[i, j] * (b[v, u] - b[u, v]) + ... ) 
    # The previous derivation was:
    # delta = Cross + Sum_k
    # Cross = 2 * (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
    # Sum_k = sum_{k!=i,j} (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[v, p[k]] - b[u, p[k]])
    # If symmetric:
    # Cross = 0
    # Sum_k = sum_{k!=i,j} 2 * (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    
    # Let's verify this symmetric formula.
    # If A, B symmetric:
    # delta = 2 * sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    # Let's test this with a small case or trust the derivation.
    # Derivation check:
    # Term for k: (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[v, p[k]] - b[u, p[k]])
    # = 2 * (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    # Yes.
    
    # So for symmetric A, B:
    # delta = 2 * dot( (a[i, :] - a[j, :]) , (b[v, p] - b[u, p]) )
    # We need to exclude k=i and k=j from the dot product.
    # Let diff_a = a[i, :] - a[j, :]
    # Let diff_b = b[v, p] - b[u, p]
    # dot_full = dot(diff_a, diff_b)
    # term_i = diff_a[i] * diff_b[i] = (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
    # term_j = diff_a[j] * diff_b[j] = (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
    # sum_k = dot_full - term_i - term_j
    # delta = 2 * sum_k
    
    # This is very fast. O(n) with vectorized ops.
    
    while time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        u = p[i]
        v = p[j]
        
        # Compute delta
        diff_a = a[i, :] - a[j, :]
        diff_b = b[v, p] - b[u, p]
        
        dot_full = np.dot(diff_a, diff_b)
        
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        sum_k = dot_full - term_i - term_j
        delta = 2.0 * sum_k
        
        # Acceptance
        if delta < 0 or (current_T > 1e-9 and np.random.random() < np.exp(-delta / current_T)):
            # Accept swap
            p[i], p[j] = p[j], p[i]
            current_cost += int(delta)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cooling
        current_T *= alpha
        iter_count += 1
        
        # Reset if stuck? 
        # Simple SA doesn't reset.
        
        if iter_count % 10000 == 0:
             if time.time() > deadline - 5:
                 break

    return best_p.tolist()
# EVOLVE-BLOCK-END
