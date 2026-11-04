"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    """
    Solves the QAP instance using rank-matching initialization,
    efficient vectorized 2-opt local search, and SA polish.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute for fast delta evaluation
    # For swap (i, j) with locations u=p[i], v=p[j]:
    # delta = sum_k (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) - corrections
    # Let A = a, B = b
    # Define row sums for faster computation
    
    # 1. Initialization: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    heavy_order = np.argsort(-flow_sums, kind='stable')
    central_order = np.argsort(dist_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    current_cost = np.sum(a * b[np.ix_(p, p)])
    best_p = p.copy()
    best_cost = current_cost

    def compute_cost(perm):
        return np.sum(a * b[np.ix_(perm, perm)])

    # 2. Efficient 2-Opt Local Search with vectorized inner loop
    # Key insight: For fixed i, we can vectorize over all j > i
    # delta(i,j) = dot(a[i]-a[j], b[p[j],p]-b[p[i],p]) - corrections
    
    while time.time() < deadline - 20.0:  # Reserve 20s for SA polish
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Precompute B_p = b[p, :] so B_p[i, k] = b[p[i], k]
        B_p = b[p, :]  # shape (n, n), B_p[i,k] = b[p[i],k]
        
        for i in range(n):
            u = p[i]
            a_i = a[i]
            b_u_p = B_p[i]  # = b[u, p]
            
            # For all j > i:
            # delta = dot(a[i]-a[j], b[p[j],p]-b[u,p]) - corr_i - corr_j
            # = dot(a[i], b[p[j],p]) - dot(a[i], b[u,p]) - dot(a[j], b[p[j],p]) + dot(a[j], b[u,p]) - corr_i - corr_j
            
            # Vectorize: for j in range(i+1, n)
            # b[p[j], p] = B_p[j]
            # So we need: for each j, dot(a_i, B_p[j]) - dot(a_i, b_u_p) - dot(a[j], B_p[j]) + dot(a[j], b_u_p)
            
            # dot(a_i, B_p[j]) for all j: this is a_i @ B_p.T -> shape (n,)
            # dot(a[j], B_p[j]) for all j: this is sum(a * B_p, axis=1) -> shape (n,)
            # dot(a[j], b_u_p) for all j: this is a @ b_u_p -> shape (n,)
            # dot(a_i, b_u_p) is scalar
            
            # Let's compute these vectorized quantities:
            # term1[j] = dot(a_i, B_p[j]) = (a_i @ B_p.T)[j]
            term1 = a_i @ B_p.T  # shape (n,)
            
            # term2 = dot(a_i, b_u_p) = scalar
            term2 = a_i @ b_u_p  # scalar
            
            # term3[j] = dot(a[j], B_p[j]) = sum(a * B_p, axis=1)[j]
            term3 = np.sum(a * B_p, axis=1)  # shape (n,)
            
            # term4[j] = dot(a[j], b_u_p) = a @ b_u_p
            term4 = a @ b_u_p  # shape (n,)
            
            # sum_all[j] = term1[j] - term2 - term3[j] + term4[j]
            sum_all = term1 - term2 - term3 + term4  # shape (n,)
            
            # Corrections for k=i and k=j:
            # corr_i = (a[i,i] - a[j,i]) * (b[p[j],u] - b[u,u])
            # corr_j = (a[i,j] - a[j,j]) * (b[p[j],p[j]] - b[u,p[j]])
            
            # b[p[j], u] = b[p[j], p[i]] = B_p[j, i]
            # b[u, u] = b[u, u] scalar
            # b[p[j], p[j]] = b[p[j], p[j]] = B_p[j, j]
            # b[u, p[j]] = b_u_p[j]
            
            # corr_i[j] = (a[i,i] - a[j,i]) * (B_p[j,i] - b[u,u])
            # corr_j[j] = (a[i,j] - a[j,j]) * (B_p[j,j] - b_u_p[j])
            
            # For j in range(i+1, n):
            a_ii = a[i, i]
            buu = b[u, u]
            
            # Vectorized corrections for j > i:
            # corr_i = (a_ii - a[j, i]) * (B_p[j, i] - buu)
            corr_i = (a_ii - a[:, i]) * (B_p[:, i] - buu)  # shape (n,)
            
            # corr_j = (a[i, j] - a[j, j]) * (B_p[j, j] - b_u_p[j])
            a_ii_diag = np.diag(a)  # a[j,j]
            corr_j = (a[i, :] - a_ii_diag) * (np.diag(B_p) - b_u_p)  # shape (n,)
            
            # delta[j] = sum_all[j] - corr_i[j] - corr_j[j]
            delta = sum_all - corr_i - corr_j  # shape (n,)
            
            # Only consider j > i
            delta[:i+1] = 0  # Mask out invalid indices
            # Also mask out j == i (already done by i+1)
            
            # Find best j > i
            j_candidates = np.arange(i+1, n)
            if len(j_candidates) == 0:
                continue
            
            best_j_local = j_candidates[np.argmin(delta[j_candidates])]
            best_delta_local = delta[best_j_local]
            
            if best_delta_local < best_delta:
                best_delta = best_delta_local
                best_i = i
                best_j = best_j_local
        
        if best_i == -1:
            break
        
        # Apply swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # 3. Simulated Annealing Polish
    # Use SA to escape local optima
    if time.time() < deadline - 2.0:
        T_start = max(1.0, best_cost * 0.01)
        T_end = 0.01
        cooling = 0.995
        T = T_start
        rng = np.random.default_rng(42)
        
        p_sa = best_p.copy()
        cost_sa = best_cost
        
        # Precompute for SA
        while T > T_end and time.time() < deadline - 2.0:
            # Random swap
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            
            u, v = p_sa[i], p_sa[j]
            
            # Compute delta for swap
            diff_a = a[i] - a[j]
            diff_b = b[v, p_sa] - b[u, p_sa]
            sum_all = np.dot(diff_a, diff_b)
            
            term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
            term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
            delta = sum_all - term_i - term_j
            
            # Accept if improvement or with probability exp(-delta/T)
            if delta < 0 or (rng.random() < np.exp(-delta / T)):
                p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
                cost_sa += delta
                
                if cost_sa < best_cost:
                    best_cost = cost_sa
                    best_p = p_sa.copy()
            
            T *= cooling
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
