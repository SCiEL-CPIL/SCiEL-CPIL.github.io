"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Precompute some useful quantities
    a_row_sum = a.sum(axis=1)
    a_col_sum = a.sum(axis=0)
    b_row_sum = b.sum(axis=1)
    b_col_sum = b.sum(axis=0)

    def compute_cost(p):
        """Compute the QAP cost for permutation p (p[i] = location of facility i)."""
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        return np.sum(a * b[np.ix_(p, p)])

    def linex(p, b_matrix):
        """
        Linear assignment subproblem: given fixed assignment of all but one facility,
        find best location for the remaining facility.
        Actually, we use a different approach: for 2-exchange, we compute the cost delta
        analytically.
        """
        pass

    def two_exchange_delta(p, i, j):
        """
        Compute the cost delta of swapping facilities i and j (swapping p[i] and p[j]).
        Let u = p[i], v = p[j]. After swap, p[i]=v, p[j]=u.
        Delta = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) + sum_k a[j,k]*(b[u,p[k]] - b[v,p[k]])
               + sum_k a[k,i]*(b[p[k],v] - b[p[k],u]) + sum_k a[k,j]*(b[p[k],u] - b[p[k],v])
               - a[i,j]*(b[u,v] - b[v,u]) - a[j,i]*(b[v,u] - b[u,v])
        Since b[u,v] - b[v,u] appears, and a[i,j]*b[u,v] terms...
        
        More carefully:
        Original contribution involving i or j:
        For pair (i,k), k != j: a[i,k]*b[u, p[k]]
        For pair (j,k), k != i: a[j,k]*b[v, p[k]]
        For pair (k,i), k != j: a[k,i]*b[p[k], u]
        For pair (k,j), k != i: a[k,j]*b[p[k], v]
        For pair (i,j): a[i,j]*b[u,v]
        For pair (j,i): a[j,i]*b[v,u]

        After swap (i gets v, j gets u):
        For pair (i,k), k != j: a[i,k]*b[v, p[k]]
        For pair (j,k), k != i: a[j,k]*b[u, p[k]]
        For pair (k,i), k != j: a[k,i]*b[p[k], v]
        For pair (k,j), k != i: a[k,j]*b[p[k], u]
        For pair (i,j): a[i,j]*b[v,u]
        For pair (j,i): a[j,i]*b[u,v]

        Delta = sum_{k!=i,j} [a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[j,k]*(b[u,p[k]]-b[v,p[k]])
                              + a[k,i]*(b[p[k],v]-b[p[k],u]) + a[k,j]*(b[p[k],u]-b[p[k],v])]
               + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
        
        Note: b[v,u]-b[u,v] = -(b[u,v]-b[v,u])
        So last two terms: a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
                          = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
                          = (a[j,i] - a[i,j]) * (b[u,v] - b[v,u])
        """
        u, v = p[i], p[j]
        # Vectorized computation
        # Terms involving i and j with all other k
        # a[i, :]*(b[v, :] - b[u, :]) but we need to exclude k=i and k=j
        # Actually let's just compute the full sum and subtract the i,j terms
        diff_b_row = b[v, :] - b[u, :]  # b[v,k] - b[u,k]
        diff_b_col = b[:, v] - b[:, u]  # b[k,v] - b[k,u]
        
        # Full sums
        s1 = a[i, :] @ diff_b_row  # sum_k a[i,k]*(b[v,k]-b[u,k])
        s2 = a[j, :] @ (-diff_b_row)  # sum_k a[j,k]*(b[u,k]-b[v,k])
        s3 = a[:, i] @ diff_b_col  # sum_k a[k,i]*(b[k,v]-b[k,u])
        s4 = a[:, j] @ (-diff_b_col)  # sum_k a[k,j]*(b[k,u]-b[k,v])
        
        # Subtract terms where k=i or k=j
        # For s1: k=i gives a[i,i]*(b[v,i]-b[u,i]), k=j gives a[i,j]*(b[v,j]-b[u,j])
        s1 -= a[i, i] * (b[v, i] - b[u, i])
        s1 -= a[i, j] * (b[v, j] - b[u, j])
        
        # For s2: k=i gives a[j,i]*(b[u,i]-b[v,i]), k=j gives a[j,j]*(b[u,j]-b[v,j])
        s2 -= a[j, i] * (b[u, i] - b[v, i])
        s2 -= a[j, j] * (b[u, j] - b[v, j])
        
        # For s3: k=i gives a[i,i]*(b[i,v]-b[i,u]), k=j gives a[j,i]*(b[j,v]-b[j,u])
        s3 -= a[i, i] * (b[i, v] - b[i, u])
        s3 -= a[j, i] * (b[j, v] - b[j, u])
        
        # For s4: k=i gives a[i,j]*(b[i,u]-b[i,v]), k=j gives a[j,j]*(b[j,u]-b[j,v])
        s4 -= a[i, j] * (b[i, u] - b[i, v])
        s4 -= a[j, j] * (b[j, u] - b[j, v])
        
        # Cross terms (i,j) and (j,i)
        cross = (a[j, i] - a[i, j]) * (b[u, v] - b[v, u])
        
        return s1 + s2 + s3 + s4 + cross

    # Initial solution: rank matching
    heavy = np.argsort(-a_row_sum, kind="stable")
    central = np.argsort(b_row_sum, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = compute_cost(p)
    
    # Also try a few random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        p_rand = rng.permutation(n)
        c = compute_cost(p_rand)
        if c < best_cost:
            best_cost = c
            best_p = p_rand.copy()
            p = p_rand.copy()
    
    # Local search: 2-exchange with vectorized best-move selection
    improved = True
    while improved and time.time() < deadline:
        improved = False
        p = best_p.copy()
        current_cost = best_cost
        
        # For efficiency, do multiple passes
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Find best 2-exchange
            best_delta = 0
            best_i, best_j = -1, -1
            
            # Sample a subset of pairs for speed, or do full search if n is small enough
            # For n=100, full search is 4950 pairs. Let's try full search.
            # Vectorize: compute all deltas at once is O(n^3) which is 1e6, fine.
            
            # Actually, let's compute deltas for all pairs (i,j) with i<j
            # This is O(n^2 * n) = O(n^3) per iteration which might be too slow in Python
            # Let's use a smarter approach: for each i, find best j
            
            for i in range(n):
                if time.time() > deadline:
                    break
                u = p[i]
                diff_b_row_i = b[u, :]  # we'll reuse
                
                # For each j > i, compute delta
                # Let's vectorize over j
                js = np.arange(i+1, n)
                vs = p[js]  # locations of facilities j
                
                # For each (i, j): delta involves:
                # a[i, :] @ (b[vs, :] - b[u, :]) type terms... this is complex to vectorize fully
                # Let's just do a loop but with early termination
                
                # Actually, for n=100, 4950 pairs * ~100 ops each = ~500k ops per pass
                # In Python this is slow. Let's use NumPy to speed up.
                
                # Compute for all j at once:
                # s1_j = sum_k a[i,k]*(b[p[j],k] - b[p[i],k]) for k != i,j
                # This is: a[i,:] @ b[p[j],:] - a[i,:] @ b[p[i],:] - corrections
                
                a_i = a[i, :]
                b_u = b[u, :]
                
                # b[vs, :] is (m, n) where m = n-i-1
                # a_i @ b[vs, :].T gives (m,) = sum_k a[i,k]*b[p[j],k]
                s1_full = b[vs, :].T @ a_i  # (m,)
                s1_u = b_u @ a_i  # scalar
                s1 = s1_full - s1_u
                
                # s2_j = sum_k a[j,k]*(b[p[i],k] - b[p[j],k]) = a[j,:] @ (b_u - b[p[j],:])
                # = a[j,:] @ b_u - a[j,:] @ b[p[j],:]
                # a[vs, :] @ b_u gives (m,) = sum_k a[j,k]*b[u,k]
                s2_full = b_u @ a[vs, :].T  # (m,)
                s2_self = np.einsum('ij,ij->i', a[vs, :], b[vs, :])  # (m,)
                s2 = s2_full - s2_self
                
                # s3_j = sum_k a[k,i]*(b[k,p[j]] - b[k,p[i]])
                # = a[:,i] @ (b[:,p[j]] - b[:,u])
                a_col_i = a[:, i]
                b_col_u = b[:, u]
                s3_full = b[:, vs].T @ a_col_i  # (m,)
                s3_u = b_col_u @ a_col_i  # scalar
                s3 = s3_full - s3_u
                
                # s4_j = sum_k a[k,j]*(b[k,p[i]] - b[k,p[j]])
                # = a[:,j] @ (b[:,u] - b[:,p[j]])
                # = a[:,j] @ b[:,u] - a[:,j] @ b[:,p[j]]
                s4_full = b_col_u @ a[:, vs].T  # (m,)
                s4_self = np.einsum('ij,ij->i', a[:, vs], b[:, vs])  # (m,)
                s4 = s4_full - s4_self
                
                # Cross term: (a[j,i] - a[i,j]) * (b[u,v] - b[v,u])
                a_ji = a[vs, i]  # (m,)
                a_ij = a[i, vs]  # (m,)
                b_uv = b[u, vs]  # (m,)
                b_vu = b[vs, u]  # (m,)
                cross = (a_ji - a_ij) * (b_uv - b_vu)
                
                deltas = s1 + s2 + s3 + s4 + cross
                
                # Find best
                idx = np.argmin(deltas)
                if deltas[idx] < best_delta:
                    best_delta = deltas[idx]
                    best_j = js[idx]
            
            if best_j == -1 or best_delta >= 0:
                break
            
            # Apply the swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # If no improvement in this pass, try a random kick to escape local optimum
        if not improved:
            # Random kick: swap 2-3 random pairs
            for _ in range(3):
                i, j = rng.integers(0, n, 2)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            c = compute_cost(p)
            if c < best_cost:
                best_cost = c
                best_p = p.copy()
    
    # Final polish: exact 2-exchange until no improvement
    p = best_p.copy()
    current_cost = best_cost
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        found = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            u = p[i]
            for j in range(i+1, n):
                v = p[j]
                d = two_exchange_delta(p, i, j)
                if d < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    current_cost += d
                    found = True
                    break
            if found:
                break
        if not found:
            break
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
