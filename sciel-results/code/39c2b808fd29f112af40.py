"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    p_arr = np.asarray(p)
    total = 0
    for i in range(len(a)):
        total += np.dot(a[i], b[p_arr[i]])
    return total

def _greedy_initial(a, b, n):
    """Create a good initial solution using a simple greedy heuristic."""
    # Sort facilities by row sum of a
    a_row_sums = a.sum(axis=1)
    facility_order = np.argsort(-a_row_sums, kind='stable')
    
    assigned = np.full(n, -1, dtype=np.int64)
    used_locations = set()
    
    for idx in facility_order:
        best_loc = -1
        best_cost = np.inf
        
        # Vectorized incremental cost calculation for all available locations
        # inc_cost[loc] = a[idx, idx]*b[loc, loc] + sum_{k assigned} (a[idx, k]*b[loc, assigned[k]] + a[k, idx]*b[assigned[k], loc])
        
        # Precompute sums for assigned facilities
        # This is O(n^2) per step, O(n^3) total. Acceptable for n=256.
        
        # To speed up, we can just loop through available locations
        # But n=256 is small enough for a simple loop in Python if not too many iterations?
        # Actually, let's just use a random initialization followed by local search, 
        # or a simple greedy. 
        # Given the complexity, a random start + strong local search is often robust.
        pass

    # Fallback to random if greedy is too complex to implement correctly/fast
    return list(range(n))

def _vectorized_local_search(a, b, p, time_limit=110):
    """
    Perform vectorized best-improvement 2-swap local search.
    """
    n = len(p)
    p = np.array(p, dtype=np.int64)
    current_cost = _qap_cost(a, b, p)
    
    # Precompute symmetric parts
    A_sym = a + a.T
    # B is assumed symmetric, but we use B directly
    
    start_time = time.time()
    improved = True
    
    while improved:
        improved = False
        
        # Iterate through all i
        for i in range(n):
            if time.time() - start_time > time_limit:
                return p.tolist(), current_cost
                
            u = p[i]
            
            # Compute components for all j
            # R_j = sum_k A_sym[j, k] * B[p[j], p[k]]
            # B_p = B[p]
            B_p = b[p]
            R = np.sum(A_sym * B_p, axis=1)
            
            # Q_j = sum_k A_sym[i, k] * B[p[j], p[k]]
            # A_sym[i] is a vector
            Q = B_p @ A_sym[i]
            
            # S_j = sum_k A_sym[j, k] * B[u, p[k]]
            # B_u_p = B[u, p]
            B_u_p = b[u, p]
            S = A_sym @ B_u_p
            
            # T = sum_k A_sym[i, k] * B[u, p[k]]
            T = np.dot(A_sym[i], B_u_p)
            
            # Compute deltas for all j != i
            # Delta_ij = 
            # A_ii(B_vv - B_uu) + A_jj(B_uu - B_vv) + (A_ij - A_ji)(B_vu - B_uv)
            # + (R_j - A_sym_ji B_vu - A_sym_jj B_vv)
            # - (Q_j - A_sym_ii B_vu - A_sym_ij B_vv)
            # - (S_j - A_sym_ji B_uu - A_sym_jj B_uv)
            # + (T - A_sym_ii B_uu - A_sym_ij B_uv)
            
            # Let's group terms by j
            # Constant terms (independent of j, or dependent on i only):
            # C1 = A_ii B_uu - A_sym_ii B_uu + T - A_sym_ii B_uu
            # Wait, let's just compute the vector of deltas for all j.
            
            # Terms involving j:
            # 1. A_jj * (-B_vv + B_uu)  -> -A_jj B_vv + A_jj B_uu
            # 2. (A_ij - A_ji) * (B_vu - B_uv)
            # 3. R_j
            # 4. - A_sym_ji B_vu - A_sym_jj B_vv
            # 5. - Q_j + A_sym_ii B_vu + A_sym_ij B_vv
            # 6. - S_j + A_sym_ji B_uu + A_sym_jj B_uv
            # 7. + T - A_sym_ii B_uu - A_sym_ij B_uv
            
            # Let's compute arrays for all j:
            # B_vv = b[p, p] (diagonal of b indexed by p)
            B_vv = np.diag(b[p]) # Or b[p, p]
            # B_uu = b[u, u]
            B_uu = b[u, u]
            
            # B_vu = b[p, u]
            B_vu = b[p, u]
            # B_uv = b[u, p]
            B_uv = b[u, p]
            
            # A_jj = a[p, p] ? No, A_jj is a[j, j]. j is index, not location.
            # A_jj = a[np.arange(n), np.arange(n)]? No, a is facility matrix.
            # a[j, j] is the diagonal of a.
            A_jj = np.diag(a)
            
            # A_ij = a[i, j]
            A_ij = a[i, :]
            # A_ji = a[j, i]
            A_ji = a[:, i]
            
            # A_sym_ji = A_sym[j, i]
            A_sym_ji = A_sym[:, i]
            # A_sym_ij = A_sym[i, j]
            A_sym_ij = A_sym[i, :]
            # A_sym_jj = A_sym[j, j]
            A_sym_jj = np.diag(A_sym)
            # A_sym_ii = A_sym[i, i]
            A_sym_ii = A_sym[i, i]
            
            # Vectorized calculation
            # Term 1: A_jj * (B_uu - B_vv)
            t1 = A_jj * (B_uu - B_vv)
            
            # Term 2: (A_ij - A_ji) * (B_vu - B_uv)
            t2 = (A_ij - A_ji) * (B_vu - B_uv)
            
            # Term 3: R
            t3 = R
            
            # Term 4: - (A_sym_ji * B_vu + A_sym_jj * B_vv)
            t4 = -(A_sym_ji * B_vu + A_sym_jj * B_vv)
            
            # Term 5: - Q + A_sym_ii * B_vu + A_sym_ij * B_vv
            t5 = -Q + A_sym_ii * B_vu + A_sym_ij * B_vv
            
            # Term 6: - S + A_sym_ji * B_uu + A_sym_jj * B_uv
            t6 = -S + A_sym_ji * B_uu + A_sym_jj * B_uv
            
            # Term 7: T - A_sym_ii * B_uu - A_sym_ij * B_uv
            t7 = T - A_sym_ii * B_uu - A_sym_ij * B_uv
            
            # Total Delta
            deltas = t1 + t2 + t3 + t4 + t5 + t6 + t7
            
            # Mask out i
            deltas[i] = np.inf
            
            # Find best j
            j_best = np.argmin(deltas)
            delta_best = deltas[j_best]
            
            if delta_best < 0:
                # Perform swap
                p[i], p[j_best] = p[j_best], p[i]
                current_cost += delta_best
                improved = True
                # Restart the loop or continue? 
                # Best improvement usually restarts or continues to next i.
                # Let's break and restart to ensure we find the global best improvement in the neighborhood
                # Or just continue. Continuing is faster (First Improvement style per i, but Best per i).
                # Since we found the best j for this i, we accept it.
                # We should probably restart the i-loop to see if new swaps are possible.
                break
        
        if time.time() - start_time > time_limit:
            break

    return p.tolist(), current_cost


def solve(a, b):
    n = len(a)
    
    # Random initialization
    random.seed(42)
    p = list(range(n))
    random.shuffle(p)
    
    # Run local search
    p, cost = _vectorized_local_search(a, b, p, time_limit=115)
    
    return p
# EVOLVE-BLOCK-END
