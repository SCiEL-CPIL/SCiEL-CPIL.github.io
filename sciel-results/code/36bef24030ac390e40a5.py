# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = 118.0  # safety margin before 120s/150s kill

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    # --- Initialization strategies ---
    
    # 1. Spectral initialization
    A_sym = (a + a.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    idx_a = np.argmax(eigvals)
    v_a = eigvecs[:, idx_a]

    B_sym = (b + b.T) / 2.0
    eigvals_b, eigvecs_b = np.linalg.eigh(B_sym)
    idx_b = np.argmax(eigvals_b)
    v_b = eigvecs_b[:, idx_b]

    fac_order = np.argsort(-v_a, kind="stable")
    loc_order = np.argsort(v_b, kind="stable")
    p_spectral = np.empty(n, dtype=np.int64)
    p_spectral[fac_order] = loc_order
    cost_spectral = compute_cost(p_spectral)

    # 2. Rank-based initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)

    # 3. Hungarian algorithm initialization (Linear Surrogate)
    C_hung = np.zeros((n, n))
    for i in range(n):
        C_hung[i, :] = np.dot(a[i, :], b)
    
    # Solve linear assignment
    row_ind, col_ind = linear_sum_assignment(C_hung)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    cost_hung = compute_cost(p_hung)

    # Select best initial
    candidates = [
        (p_spectral, cost_spectral),
        (p_rank, cost_rank),
        (p_hung, cost_hung)
    ]
    p, best_cost = min(candidates, key=lambda x: x[1])

    # --- Local Search ---
    # Precompute row and col sums of 'a' for efficient delta calculation
    a_row_sums = a.sum(axis=1)
    a_col_sums = a.sum(axis=0)
    
    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        
        # Delta for swapping locations u and v for facilities i and j
        # Cost change = (a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[i,i]*b[v,v] + a[j,j]*b[u,u]) 
        #             - (a[i,i]*b[u,u] + a[j,j]*b[v,v] + a[i,j]*b[u,v] + a[j,i]*b[v,u])
        #             + (sum_{k!=i,j} a[i,k]*(b[v,p[k]] - b[u,p[k]]) + sum_{k!=i,j} a[j,k]*(b[u,p[k]] - b[v,p[k]])
        #              + sum_{k!=i,j} a[k,i]*(b[p[k],v] - b[p[k],u]) + sum_{k!=i,j} a[k,j]*(b[p[k],u] - b[p[k],v]))
        
        # Let's group terms:
        # 1. Direct terms involving i,j and u,v
        direct_new = a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[i,i]*b[v,v] + a[j,j]*b[u,u]
        direct_old = a[i,i]*b[u,u] + a[j,j]*b[v,v] + a[i,j]*b[u,v] + a[j,i]*b[v,u]
        
        # 2. Cross terms with other facilities k
        # sum_k a[i,k] b[v, p[k]] - sum_k a[i,k] b[u, p[k]]
        # = a[i,v]*b[v,v] + a[i,u]*b[v,u] + sum_{k!=i,j} a[i,k] (b[v,p[k]] - b[u,p[k]])
        #   - (a[i,v]*b[u,v] + a[i,u]*b[u,u] + sum_{k!=i,j} a[i,k] (b[u,p[k]] - b[v,p[k]]) )
        
        # This is getting complicated. Let's use the vectorized approach which is O(n) and fast enough for n=100.
        # delta = (a[i,:] - a[j,:]) @ (b[v,p] - b[u,p]) + (a[:,i] - a[:,j]) @ (b[p,v] - b[p,u]) + correction
        
        # Let's derive the vectorized formula carefully.
        # C = sum_i sum_j a[i,j] b[p[i], p[j]]
        # Swap i and j. p'[i]=v, p'[j]=u, p'[k]=p[k] for k!=i,j.
        # C' - C = sum_k (a[i,k] b[v, p'[k]] + a[k,i] b[p'[k], v]) - sum_k (a[i,k] b[u, p[k]] + a[k,i] b[p[k], u])
        #         + sum_k (a[j,k] b[u, p'[k]] + a[k,j] b[p'[k], u]) - sum_k (a[j,k] b[v, p[k]] + a[k,j] b[p[k], v])
        
        # Term 1: sum_k a[i,k] (b[v, p'[k]] - b[u, p[k]])
        # p'[k] = p[k] for k!=i,j.
        # k=i: a[i,i] (b[v,v] - b[u,u])
        # k=j: a[i,j] (b[v,u] - b[u,v])
        # k!=i,j: a[i,k] (b[v, p[k]] - b[u, p[k]])
        
        # Term 2: sum_k a[k,i] (b[p'[k], v] - b[p[k], u])
        # k=i: a[i,i] (b[v,v] - b[u,u])
        # k=j: a[j,i] (b[u,v] - b[v,u])
        # k!=i,j: a[k,i] (b[p[k], v] - b[p[k], u])
        
        # Term 3: sum_k a[j,k] (b[u, p'[k]] - b[v, p[k]])
        # k=i: a[j,i] (b[u,v] - b[v,u])
        # k=j: a[j,j] (b[u,u] - b[v,v])
        # k!=i,j: a[j,k] (b[u, p[k]] - b[v, p[k]])
        
        # Term 4: sum_k a[k,j] (b[p'[k], u] - b[p[k], v])
        # k=i: a[i,j] (b[v,u] - b[u,v])
        # k=j: a[j,j] (b[u,u] - b[v,v])
        # k!=i,j: a[k,j] (b[p[k], u] - b[p[k], v])
        
        # Summing all:
        # Direct terms (i,i), (j,j), (i,j), (j,i):
        # 2*a[i,i]*(b[v,v]-b[u,u]) + 2*a[j,j]*(b[u,u]-b[v,v]) + 2*a[i,j]*(b[v,u]-b[u,v]) + 2*a[j,i]*(b[u,v]-b[v,u])
        # Note: b is not necessarily symmetric.
        
        # Cross terms (k!=i,j):
        # sum_{k!=i,j} [ a[i,k](b[v,p[k]]-b[u,p[k]]) + a[k,i](b[p[k],v]-b[p[k],u]) + a[j,k](b[u,p[k]]-b[v,p[k]]) + a[k,j](b[p[k],u]-b[p[k],v]) ]
        # = sum_{k!=i,j} [ (a[i,k]-a[j,k])(b[v,p[k]]-b[u,p[k]]) + (a[k,i]-a[k,j])(b[p[k],v]-b[p[k],u]) ]
        
        # So delta = Direct + Cross
        # Cross = (a[i,:]-a[j,:]) @ (b[v,p]-b[u,p]) - (a[i,i]-a[j,i])*(b[v,u]-b[u,u]) - (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
        #        + (a[:,i]-a[:,j]) @ (b[p,v]-b[p,u]) - (a[i,i]-a[i,j])*(b[u,v]-b[u,u]) - (a[j,i]-a[j,j])*(b[v,v]-b[u,v])
        
        # Let's compute the full dot products first (includes i and j terms) and then subtract the i,j specific parts if they are wrong in the vectorized form.
        # Actually, the vectorized form (a[i,:]-a[j,:]) @ (b[v,p]-b[u,p]) includes k=i and k=j.
        # Let's check if the vectorized form matches the cross term derivation.
        # Vector part 1: sum_k (a[i,k]-a[j,k]) (b[v,p[k]]-b[u,p[k]])
        # For k=i: (a[i,i]-a[j,i]) (b[v,u]-b[u,u])
        # For k=j: (a[i,j]-a[j,j]) (b[v,v]-b[u,v])
        # For k!=i,j: (a[i,k]-a[j,k]) (b[v,p[k]]-b[u,p[k]])
        
        # Vector part 2: sum_k (a[k,i]-a[k,j]) (b[p[k],v]-b[p[k],u])
        # For k=i: (a[i,i]-a[i,j]) (b[u,v]-b[u,u])
        # For k=j: (a[j,i]-a[j,j]) (b[v,v]-b[u,v])
        # For k!=i,j: (a[k,i]-a[k,j]) (b[p[k],v]-b[p[k],u])
        
        # Sum of vector parts = Cross + (terms for k=i and k=j in vector parts)
        # We need to subtract the k=i and k=j terms from the vector parts and add the correct Direct terms.
        
        # This is getting messy. For n=100, the O(n) vectorized computation is fast.
        # Let's just compute:
        # delta = np.dot(a[i]-a[j], b[v,p]-b[u,p]) + np.dot(a[:,i]-a[:,j], b[p,v]-b[p,u])
        # and then add the correction for the double counting or wrong terms.
        
        # Actually, a simpler way for n=100 is to just compute the cost change directly by summing the relevant terms.
        # But that's O(n) too.
        
        # Let's try the vectorized formula and see if it works.
        # If not, I'll fall back to the explicit sum.
        
        # Explicit sum is safer.
        # delta = 0
        # for k in range(n):
        #     if k == i or k == j: continue
        #     delta += a[i,k]*(b[v,p[k]]-b[u,p[k]])
        #     delta += a[k,i]*(b[p[k],v]-b[p[k],u])
        #     delta += a[j,k]*(b[u,p[k]]-b[v,p[k]])
        #     delta += a[k,j]*(b[p[k],u]-b[p[k],v])
        # delta += a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v])
        # delta += a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v])
        # delta += a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v])  # Wait, terms 1 and 2 both have i,i and i,j?
        # Let's re-evaluate the direct terms from the 4 sums.
        # Sum 1 (i row): a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v])
        # Sum 2 (i col): a[i,i](b[v,v]-b[u,u]) + a[j,i](b[u,v]-b[v,u])  <- k=i term is a[i,i](b[v,v]-b[u,u])? No, p'[i]=v, p[i]=u. b[p'[i],v]-b[p[i],u] = b[v,v]-b[u,u]. Correct.
        # Sum 3 (j row): a[j,i](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v])
        # Sum 4 (j col): a[i,j](b[v,u]-b[u,v]) + a[j,j](b[u,u]-b[v,v])
        
        # Total Direct = 2*a[i,i]*(b[v,v]-b[u,u]) + 2*a[j,j]*(b[u,u]-b[v,v]) + 2*a[i,j]*(b[v,u]-b[u,v]) + 2*a[j,i]*(b[u,v]-b[v,u])
        
        # Cross terms (k!=i,j) as derived before.
        
        # Let's implement this explicit formula. It's O(n) and safe.
        
        # Optimization: Precompute b[v,p] and b[u,p] etc? No, p changes.
        # But we can compute the vectors once per swap call.
        
        bv_p = b[v, p]
        bu_p = b[u, p]
        bp_v = b[p, v]
        bp_u = b[p, u]
        
        # Cross terms
        # sum_{k!=i,j} (a[i,k]-a[j,k])*(bv_p[k]-bu_p[k]) + (a[k,i]-a[k,j])*(bp_v[k]-bp_u[k])
        
        # We can compute the full sum and subtract i,j terms.
        diff_a_row = a[i] - a[j]
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = bv_p - bu_p
        diff_b_col = bp_v - bp_u
        
        cross_full = np.dot(diff_a_row, diff_b_row) + np.dot(diff_a_col, diff_b_col)
        
        # Subtract k=i and k=j terms from cross_full
        # k=i: (a[i,i]-a[j,i])*(bv_p[i]-bu_p[i]) + (a[i,i]-a[i,j])*(bp_v[i]-bp_u[i])
        #      = (a[i,i]-a[j,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[i,j])*(b[u,v]-b[u,u])
        # k=j: (a[i,j]-a[j,j])*(bv_p[j]-bu_p[j]) + (a[j,i]-a[j,j])*(bp_v[j]-bp_u[j])
        #      = (a[i,j]-a[j,j])*(b[v,v]-b[u,v]) + (a[j,i]-a[j,j])*(b[v,v]-b[u,v])
        
        term_i = (a[i,i]-a[j,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[i,j])*(b[u,v]-b[u,u])
        term_j = (a[i,j]-a[j,j])*(b[v,v]-b[u,v]) + (a[j,i]-a[j,j])*(b[v,v]-b[u,v])
        
        cross = cross_full - term_i - term_j
        
        direct = 2*a[i,i]*(b[v,v]-b[u,u]) + 2*a[j,j]*(b[u,u]-b[v,v]) + 2*a[i,j]*(b[v,u]-b[u,v]) + 2*a[j,i]*(b[u,v]-b[v,u])
        
        return cross + direct

    def local_search(p, current_cost):
        p = p.copy()
        improved = True
        while improved:
            if time.time() - t_start > deadline:
                break
            improved = False
            # Best improvement to escape shallow local minima? Or First improvement for speed?
            # Best improvement is slower but more robust. Let's try Best Improvement.
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                if time.time() - t_start > deadline:
                    break
                for j in range(i+1, n):
                    delta = get_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_delta < -1e-9:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += int(best_delta)
                improved = True
        return p, current_cost

    # Run local search on best initial
    p, best_cost = local_search(p, best_cost)

    # --- Iterated Local Search ---
    rng = np.random.RandomState(42)
    no_improve_count = 0
    max_no_improve = 20

    while time.time() - t_start < deadline - 2.0:
        # Perturbation
        n_swaps = rng.randint(2, 5)
        p_pert = p.copy()
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        p_new, cost_new = local_search(p_pert, compute_cost(p_pert))
        
        if cost_new < best_cost:
            p = p_new
            best_cost = cost_new
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                no_improve_count = 0
                n_swaps = rng.randint(5, 10)
                p_pert = p.copy()
                for _ in range(n_swaps):
                    i, j = rng.choice(n, 2, replace=False)
                    p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
                p_new, cost_new = local_search(p_pert, compute_cost(p_pert))
                if cost_new < best_cost:
                    p = p_new
                    best_cost = cost_new

    return p.tolist()
# EVOLVE-BLOCK-END
