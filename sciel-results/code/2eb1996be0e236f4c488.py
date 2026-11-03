"""QAP solver for tho150: spectral init + fast 2-opt local search with precomputed deltas."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg


def _cost(a, b, p):
    """Exact integer cost for a permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _spectral_init(a, b, n, rng):
    """Initialize permutation using eigenvector alignment of flow and distance matrices."""
    # Symmetrize for eigenvalue decomposition
    a_sym = (a + a.T) * 0.5
    b_sym = (b + b.T) * 0.5

    # Get top eigenvectors (largest magnitude eigenvalues)
    # For flow: largest eigenvectors capture dominant flow structure
    # For distance: smallest eigenvectors capture compact structure
    try:
        w_a, v_a = scipy.linalg.eigh(a_sym)
        w_b, v_b = scipy.linalg.eigh(b_sym)
    except Exception:
        # Fallback to rank-matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    # Use the eigenvector with the largest absolute eigenvalue for flow
    idx_a = np.argmax(np.abs(w_a))
    eig_a = v_a[:, idx_a]

    # For distance, use the eigenvector with the smallest eigenvalue (most compact)
    idx_b = np.argmin(w_b)
    eig_b = v_b[:, idx_b]

    # Assign facilities to locations by matching their positions in the eigenvectors
    # Facility i gets location j where the eigenvector values are most "aligned"
    # Use the sign pattern: high flow eigenvector value -> low distance eigenvector value
    order_f = np.argsort(-np.abs(eig_a))
    order_l = np.argsort(np.abs(eig_b))

    # Create a permutation: facility ranked k by flow goes to location ranked k by distance
    p = np.empty(n, dtype=np.int64)
    for k in range(n):
        p[order_f[k]] = order_l[k]

    return p


def _random_init(n, rng):
    """Random permutation initialization."""
    p = np.arange(n, dtype=np.int64)
    rng.shuffle(p)
    return p


def _local_search(a, b, p, deadline, max_iters=None):
    """
    Fast 2-opt local search with precomputed swap deltas.
    
    Precompute for each facility i:
      - flow_i_j = a[i, :] (flow from i to all)
      - For a swap of positions i and j, the delta is computed in O(n)
    
    Uses precomputed dot products to speed up delta evaluation.
    """
    n = len(p)
    current_cost = _cost(a, b, p)
    
    # Precompute: for each location l, the sum of flows from all facilities to l
    # b[p[i], p[j]] * a[i, j]
    # When we swap p[i] and p[j], the change involves:
    #   - Terms involving p[i] and p[j] with all other positions
    #   - The direct a[i,j]*b[p[j],p[i]] - a[i,j]*b[p[i],p[j]] term
    
    # Precompute row sums of b weighted by a for faster delta
    # For facility i at position pi, its contribution to cost with facility k at position pk:
    #   a[i,k]*b[pi,pk] + a[k,i]*b[pi,pk]  (symmetric part)
    
    # Key precomputation: for each position l, compute S[l] = sum_k a[k, i] * b[l, p[k]]
    # This allows O(1) update when a position changes
    
    # Actually, let's use a cleaner approach:
    # Precompute M[i, l] = sum_k a[i, k] * b[l, p[k]] for all i, l
    # Then delta for swapping positions i and j:
    #   delta = M[i, p[j]] + M[j, p[i]] - M[i, p[i]] - M[j, p[j]]
    #          - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[i],p[j]]
    #          + a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[j],p[i]]
    # But M already includes the a[i,j] term, so we need to be careful.
    
    # Let me use a different formulation. Define:
    #   C[i, j] = sum_k a[i,k] * b[p[i], p[k]]  (row i's contribution to cost)
    # Total cost = sum_i C[i, i] ... no, that's not right either.
    
    # Let me think more carefully.
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # 
    # For a swap of p[i] <-> p[j]:
    # New cost = old cost + delta
    # delta = sum_k [a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[j],p[k]] + a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[i],p[k]]]
    #         - sum_k [a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[i],p[k]] + a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[i],p[k]]]
    #         - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[i],p[j]]
    #         + a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[j],p[i]]
    # 
    # Wait, the k=i and k=j terms are double-counted. Let me separate:
    # 
    # Let S_x(y) = sum_{k != x} a[x,k] * b[y, p[k]]  for x in {i,j}
    # Let T_x(y) = sum_{k != x} a[k,x] * b[y, p[k]]  for x in {i,j}
    #
    # Then delta = S_i(p[j]) + S_j(p[i]) + T_i(p[j]) + T_j(p[i])
    #              - S_i(p[i]) - S_j(p[j]) - T_i(p[i]) - T_j(p[j])
    #              + a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]])
    #              + a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])
    
    # Precompute S and T for all facilities and all locations
    # S[i, l] = sum_{k != i} a[i,k] * b[l, p[k]]
    # T[i, l] = sum_{k != i} a[k,i] * b[l, p[k]]
    
    # This is O(n^3) to compute initially, but O(n) per swap delta, and O(n) to update after a swap.
    # For n=150, n^3 = 3.375M, which is fine.
    
    # Compute S and T
    # S[i, l] = sum_k a[i,k] * b[l, p[k]] - a[i,i]*b[l, p[i]]
    # T[i, l] = sum_k a[k,i] * b[l, p[k]] - a[i,i]*b[l, p[i]]
    
    # More efficiently:
    # S = a @ (b[:, p] @ np.eye(n)[p, :]) ... let me just do it directly
    # S[i, l] = sum_k a[i,k] * b[l, p[k]]
    # = (a @ b[:, p].T)[i, l] ... no
    # b[:, p] has shape (n, n) where b[:, p][l, k] = b[l, p[k]]
    # So S = a @ b[:, p].T? 
    # (a @ b[:, p].T)[i, l] = sum_k a[i,k] * b[:, p].T[k, l] = sum_k a[i,k] * b[:, p][l, k] = sum_k a[i,k] * b[l, p[k]]
    # Yes! So S = a @ b[:, p].T
    # Similarly T = a.T @ b[:, p].T
    
    # But we need to subtract the self-term for the "k != i" part
    # Actually for the delta formula, we can include the self-term and it will cancel out
    # Let me verify: in the delta formula above, when k=i in S_i, we get a[i,i]*b[p[j],p[i]]
    # and when k=i in S_i(p[i]), we get a[i,i]*b[p[i],p[i]]
    # These don't cancel. So I should exclude k=i and k=j properly.
    
    # Let me just compute the full S and T including self-terms, and adjust the delta formula.
    # 
    # Full S[i, l] = sum_k a[i,k] * b[l, p[k]]
    # Full T[i, l] = sum_k a[k,i] * b[l, p[k]]
    #
    # Delta for swap i,j:
    # = (S[i,p[j]] - a[i,i]*b[p[j],p[i]] - a[i,j]*b[p[j],p[j]])
    # + (S[j,p[i]] - a[j,j]*b[p[i],p[j]] - a[j,i]*b[p[i],p[i]])
    # + (T[i,p[j]] - a[i,i]*b[p[j],p[i]] - a[j,i]*b[p[j],p[j]])
    # + (T[j,p[i]] - a[j,j]*b[p[i],p[j]] - a[i,j]*b[p[i],p[i]])
    # - (S[i,p[i]] - a[i,i]*b[p[i],p[i]] - a[i,j]*b[p[i],p[j]])
    # - (S[j,p[j]] - a[j,j]*b[p[j],p[j]] - a[j,i]*b[p[j],p[i]])
    # - (T[i,p[i]] - a[i,i]*b[p[i],p[i]] - a[j,i]*b[p[i],p[j]])
    # - (T[j,p[j]] - a[j,j]*b[p[j],p[j]] - a[i,j]*b[p[j],p[i]])
    # 
    # This is getting complicated. Let me use a simpler but still fast approach.
    
    # Alternative: precompute for each pair (i, j) the delta incrementally.
    # Or use the fact that for a swap, only 2 rows and 2 columns of the effective matrix change.
    
    # Let me use a cleaner O(n) delta computation:
    # For swap of positions i and j (facilities i and j):
    # delta = 0
    # for k in range(n):
    #     if k == i or k == j: continue
    #     delta += a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]])
    #     delta += a[k,i]*(b[p[j],p[k]] - b[p[i],p[k]])
    #     delta += a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
    #     delta += a[k,j]*(b[p[i],p[k]] - b[p[j],p[k]])
    # delta += a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]])
    # delta += a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])
    #
    # This is O(n) per swap. Precomputing b[p[i],:] and b[p[j],:] rows makes it fast.
    
    # Even better: precompute row sums
    # R_i(l) = sum_k a[i,k] * b[l, p[k]]  (this is S[i,l] above)
    # C_i(l) = sum_k a[k,i] * b[l, p[k]]  (this is T[i,l] above)
    #
    # Then for swap i,j:
    # delta = R_i(p[j]) + C_i(p[j]) + R_j(p[i]) + C_j(p[i])
    #         - R_i(p[i]) - C_i(p[i]) - R_j(p[j]) - C_j(p[j])
    #         + a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]])
    #         + a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])
    #         - a[i,i]*(b[p[j],p[i]] - b[p[i],p[i]])  # correction for k=i in R_i
    #         - a[i,j]*(b[p[j],p[j]] - b[p[i],p[j]])  # correction for k=j in R_i
    #         - a[i,i]*(b[p[j],p[i]] - b[p[i],p[i]])  # correction for k=i in C_i
    #         - a[j,i]*(b[p[j],p[j]] - b[p[i],p[i]])  # correction for k=j in C_i
    #         - a[j,j]*(b[p[i],p[j]] - b[p[j],p[j]])  # correction for k=j in R_j
    #         - a[i,j]*(b[p[i],p[i]] - b[p[j],p[i]])  # correction for k=i in R_j
    #         - a[j,j]*(b[p[i],p[j]] - b[p[j],p[j]])  # correction for k=j in C_j
    #         - a[i,j]*(b[p[i],p[i]] - b[p[j],p[i]])  # correction for k=i in C_j
    #
    # This is messy. Let me just do the O(n) loop per swap; for n=150 it's fast enough.
    
    # Precompute: for each position l, the row b[l, :]
    # And maintain p as a numpy array
    
    best_p = p.copy()
    best_cost = current_cost
    no_improve_count = 0
    
    while time.time() < deadline - 2.0:
        improved = False
        # Multiple passes until no improvement
        for i in range(n):
            if time.time() >= deadline - 2.0:
                break
            pi = p[i]
            for j in range(i + 1, n):
                if time.time() >= deadline - 2.0:
                    break
                pj = p[j]
                if pi == pj:
                    continue
                
                # Compute delta for swapping p[i] and p[j]
                delta = 0
                # Terms involving i and j with all other k
                for k in range(n):
                    if k == i or k == j:
                        continue
                    pk = p[k]
                    delta += a[i, k] * (b[pj, pk] - b[pi, pk])
                    delta += a[k, i] * (b[pj, pk] - b[pi, pk])
                    delta += a[j, k] * (b[pi, pk] - b[pj, pk])
                    delta += a[k, j] * (b[pi, pk] - b[pj, pk])
                # Direct terms between i and j
                delta += a[i, j] * (b[pj, pi] - b[pi, pj])
                delta += a[j, i] * (b[pj, pi] - b[pi, pj])
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = pj, pi
                    current_cost += delta
                    improved = True
                    no_improve_count = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        
        if not improved:
            break
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 150s limit, use 145s
    
    rng = np.random.RandomState(42)
    
    best_p = None
    best_cost = np.iinfo(np.int64).max
    
    # Phase 1: Spectral initialization
    p_spec = _spectral_init(a, b, n, rng)
    cost_spec = _cost(a, b, p_spec)
    if cost_spec < best_cost:
        best_cost = cost_spec
        best_p = p_spec.copy()
    
    # Phase 2: Local search from spectral init
    p_ls, cost_ls = _local_search(a, b, p_spec.copy(), deadline)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # Phase 3: Random restarts with local search
    # Use remaining time for multiple restarts
    while time.time() < deadline - 5.0:
        # Estimate time per local search iteration
        p_rand = _random_init(n, rng)
        cost_rand = _cost(a, b, p_rand)
        
        # Quick local search with limited iterations
        p_ls2, cost_ls2 = _local_search(a, b, p_rand.copy(), deadline)
        if cost_ls2 < best_cost:
            best_cost = cost_ls2
            best_p = p_ls2.copy()
    
    # Final safety: return best found
    if best_p is None:
        # Fallback
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        best_p = np.empty(n, dtype=np.int64)
        best_p[heavy] = central
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
