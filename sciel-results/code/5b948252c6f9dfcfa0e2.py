"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def calc_cost(p):
        """Calculate the QAP cost for a permutation p."""
        return int(np.sum(a * b[np.ix_(p, p)]))

    def deterministic_bin_2opt(p, a, b, time_limit):
        """
        Deterministic Best-In-Neighborhood 2-opt local search.
        Repeatedly finds the swap (i, j) that yields the maximum cost reduction
        and applies it, until no improving swap exists or time limit is reached.
        """
        current_p = p.copy()
        current_cost = calc_cost(current_p)
        
        # Precompute Bp: Bp[loc, fac] = b[loc, p[fac]]
        # This allows O(1) access to b[loc, p[fac]] via Bp[loc, fac]
        Bp = b[:, current_p]
        
        # Precompute A rows for quick access
        # a is symmetric, so a[i] is the row.
        
        while time.time() < time_limit:
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # To speed up, we can scan all pairs. n=150 -> 11175 pairs.
            # Each pair evaluation is O(n) if done naively, but we can vectorize the inner loop?
            # Or just loop i, and vectorize j?
            # Let's try full scan with vectorization over j for each i.
            
            found_improvement = False
            for i in range(n):
                u = current_p[i]
                # Vectorized calculation of deltas for all j != i
                # We need delta(i, j) for all j.
                # delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) + self_term
                # This is hard to fully vectorize over j because v=p[j] changes.
                
                # However, we can compute the "base" dot products.
                # Let's stick to the O(n^3) full scan but with tight loops in Python? 
                # n=150. 150^3 = 3.3M ops. Python loop overhead is high.
                # Better: Vectorize the inner sum for a fixed i, j pair? No.
                
                # Let's use the formula:
                # Delta(i, j) = 2 * ( dot(a[i] - a[j], Bp[v] - Bp[u]) - excl_i - excl_j ) + self_term
                # Where excl and self depend on i, j.
                
                # We can compute the main dot product term for all j at once?
                # Let M = Bp - Bp[u, :][None, :]  # n x n matrix, M[loc, k] = b[loc, p[k]] - b[u, p[k]]
                # Then dot(a[i] - a[j], M[v, :]) is what we want.
                # This is (a[i] - a[j]) @ M[v].
                # Since a[j] varies, we can't easily batch this without O(n^2) memory for intermediate terms.
                
                # Given n=150, a simple nested loop with NumPy dot products might be slow.
                # Let's optimize the delta calculation.
                # delta(i, j) = 2 * ( S(i, j) - E(i, j) ) + T(i, j)
                # S(i, j) = sum_k (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
                #         = sum_k a[i,k] b[v, p[k]] - sum_k a[i,k] b[u, p[k]] - sum_k a[j,k] b[v, p[k]] + sum_k a[j,k] b[u, p[k]]
                
                # Let C[i, v] = sum_k a[i,k] b[v, p[k]] = dot(a[i], Bp[v])
                # Then S(i, j) = C[i, v] - C[i, u] - C[j, v] + C[j, u]
                # We can precompute C for all i, v? C is n x n.
                # C[i, v] = dot(a[i], Bp[v])
                # C = a @ Bp.T  (since Bp[v] is row v of Bp, Bp.T[:, v] is col v which is Bp[v] as col? No.
                # Bp is n x n. Bp[v, :] is row v.
                # C[i, v] = sum_k a[i, k] Bp[v, k] = (a @ Bp.T)[i, v]?
                # (a @ Bp.T)[i, v] = sum_k a[i, k] Bp.T[k, v] = sum_k a[i, k] Bp[v, k]. Yes.
                
                # So, precompute C = a @ Bp.T. This is O(n^3) matrix mult, very fast in BLAS.
                # Then S(i, j) = C[i, v] - C[i, u] - C[j, v] + C[j, u]
                
                # E(i, j) = excl_i + excl_j
                # excl_i = (a[i,i] - a[j,i]) * (b[v,u] - b[u,u])
                # excl_j = (a[i,j] - a[j,j]) * (b[v,v] - b[u,v])
                
                # T(i, j) = (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
                
                # So Delta(i, j) = 2 * (C[i, v] - C[i, u] - C[j, v] + C[j, u] - excl_i - excl_j) + T(i, j)
                
                # We can vectorize the search for best j for a fixed i.
                u = current_p[i]
                # C_i = C[i, :]
                # For each j, v = p[j].
                # Term1 = C[i, v] - C[i, u]
                # Term2 = - C[j, v] + C[j, u]
                # Term3 = - excl_i
                # Term4 = - excl_j
                # Term5 = T(i, j)
                
                # Let's compute the vector of deltas for all j in one go.
                # v_all = current_p
                # C_i_v = C[i, v_all]  # vector of C[i, p[j]] for all j
                # C_j_v = C[current_p, v_all]? No, C[j, v] where j varies and v=p[j].
                # This is the diagonal of C[current_p, :]? No.
                # C[j, p[j]] is the diagonal element of C indexed by j and p[j].
                # Let's create a vector diag_C = C[np.arange(n), current_p]
                # Then C[j, p[j]] = diag_C[j]
                # C[j, u] = C[current_p, u] (vector)
                
                v_all = current_p
                C_i_v = C[i, v_all]
                C_i_u = C[i, u]
                C_j_v = C[np.arange(n), v_all] # This is diag_C
                C_j_u = C[v_all, u]
                
                # S_vec = C_i_v - C_i_u - C_j_v + C_j_u
                
                # Now excl terms.
                # excl_i = (a[i,i] - a[j,i]) * (b[v,u] - b[u,u])
                # a[j,i] is a column i of a. Since a is symmetric, a[:, i].
                # b[v,u] is b[v_all, u]
                # b[u,u] is scalar.
                excl_i_vec = (a[i, i] - a[:, i]) * (b[v_all, u] - b[u, u])
                
                # excl_j = (a[i,j] - a[j,j]) * (b[v,v] - b[u,v])
                # a[i,j] is a[i, v_all]
                # a[j,j] is diag(a)
                # b[v,v] is diag(b) indexed by v_all -> b[v_all, v_all]
                # b[u,v] is b[u, v_all]
                excl_j_vec = (a[i, v_all] - np.diag(a)) * (b[v_all, v_all] - b[u, v_all])
                
                # T_vec = (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
                T_vec = (a[i, i] - np.diag(a)) * (b[v_all, v_all] - b[u, u])
                
                delta_vec = 2 * (S_vec - excl_i_vec - excl_j_vec) + T_vec
                
                # Exclude j == i
                delta_vec[i] = np.inf
                
                # Find best j
                best_j_for_i = np.argmin(delta_vec)
                delta_val = delta_vec[best_j_for_i]
                
                if delta_val < best_delta:
                    best_delta = delta_val
                    best_i = i
                    best_j = best_j_for_i
            
            if best_delta < 0:
                # Apply the best move found
                i, j = best_i, best_j
                u, v = current_p[i], current_p[j]
                
                current_p[i], current_p[j] = v, u
                current_cost += int(best_delta)
                
                # Update Bp
                Bp[:, i] = b[:, v]
                Bp[:, j] = b[:, u]
                
                # Update C?
                # C = a @ Bp.T
                # Since Bp changed columns i and j, C changes.
                # C[:, i] = a @ Bp.T[:, i] = a @ Bp[i, :]^T = a @ b[:, v] (since Bp[i, :] was b[:, u] now b[:, v]?)
                # Wait, Bp[:, i] is the column corresponding to facility i.
                # Bp[loc, fac] = b[loc, p[fac]].
                # So Bp[:, i] = b[:, p[i]].
                # After swap, p[i]=v, so Bp[:, i] = b[:, v].
                # C = a @ Bp.T.
                # C[:, i] = a @ (Bp.T)[:, i] = a @ Bp[i, :].
                # Bp[i, :] is row i of Bp.
                # Row i of Bp: Bp[i, k] = b[i, p[k]].
                # This is getting complicated to update C incrementally.
                # Just recompute C = a @ Bp.T. O(n^3) = 3.3M ops. Very fast.
                C = a @ Bp.T
                
                found_improvement = True
            else:
                # No improving move
                break
                
        return current_p, current_cost

    # 1. Initialization: Spectral
    try:
        w_a, v_a = np.linalg.eigh(a)
        w_b, v_b = np.linalg.eigh(b)
        
        # Use top eigenvectors
        idx_a = np.argmax(np.abs(w_a))
        idx_b = np.argmax(np.abs(w_b))
        vec_a = v_a[:, idx_a]
        vec_b = v_b[:, idx_b]
        
        order_f = np.argsort(vec_a)
        order_l = np.argsort(vec_b)
        
        ranks_f = np.empty(n, dtype=np.int64)
        ranks_f[order_f] = np.arange(n)
        
        p_init = order_l[ranks_f]
        
    except Exception:
        p_init = np.random.permutation(n)
        
    # 2. ILS Loop
    best_p = p_init.copy()
    best_cost = calc_cost(best_p)
    
    # Initial polish
    current_p, current_cost = deterministic_bin_2opt(best_p, a, b, deadline - 5.0)
    if current_cost < best_cost:
        best_p = current_p
        best_cost = current_cost
        
    # ILS Perturbation and Descent
    while time.time() < deadline - 5.0:
        # Perturb best_p
        p_pert = best_p.copy()
        # Swap 3-5 random pairs
        k = np.random.randint(3, 6)
        for _ in range(k):
            i, j = np.random.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
        # Descent
        current_p, current_cost = deterministic_bin_2opt(p_pert, a, b, deadline - 5.0)
        
        if current_cost < best_cost:
            best_p = current_p
            best_cost = current_cost
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
