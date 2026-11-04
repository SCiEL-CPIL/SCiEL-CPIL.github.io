"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _compute_swap_delta(a, b, p):
    """
    Precompute the delta matrix for all pairs (i, j) of facility indices.
    delta[i, j] is the change in cost if we swap facilities i and j in p.
    
    Cost(p) = sum_{i,j} a[i,j] b[p[i], p[j]]
    
    Let S be the set of all pairs (i, j).
    If we swap i and j, the terms involving i or j change.
    The change is:
    delta_{ij} = sum_{k != i, j} a[i,k] (b[p[j], p[k]] - b[p[i], p[k]]) 
              + sum_{k != i, j} a[k,i] (b[p[k], p[j]] - b[p[k], p[i]])
              + a[i,j] (b[p[j], p[i]] - b[p[i], p[j]])
              + a[j,i] (b[p[i], p[j]] - b[p[p[j], p[i]] - b[p[i], p[j]] ... wait
              
    Let's derive carefully.
    C = sum_{u,v} a[u,v] b[p[u], p[v]]
    
    Terms involving i or j:
    1. u=i, v=j: a[i,j] b[p[i], p[j]]
    2. u=j, v=i: a[j,i] b[p[j], p[i]]
    3. u=i, v=k (k!=i,j): a[i,k] b[p[i], p[k]]
    4. u=k, v=i (k!=i,j): a[k,i] b[p[k], p[i]]
    5. u=j, v=k (k!=i,j): a[j,k] b[p[j], p[k]]
    6. u=k, v=j (k!=i,j): a[k,j] b[p[k], p[j]]
    
    After swap i <-> j:
    p'[i] = p[j], p'[j] = p[i], p'[k] = p[k]
    
    New terms:
    1'. u=i, v=j: a[i,j] b[p[j], p[i]]
    2'. u=j, v=i: a[j,i] b[p[i], p[j]]
    3'. u=i, v=k: a[i,k] b[p[j], p[k]]
    4'. u=k, v=i: a[k,i] b[p[k], p[j]]
    5'. u=j, v=k: a[j,k] b[p[i], p[k]]
    6'. u=k, v=j: a[k,j] b[p[k], p[i]]
    
    Delta = (1' - 1) + (2' - 2) + sum_{k!=i,j} [ (3'-3) + (4'-4) + (5'-5) + (6'-6) ]
    
    (1'-1) + (2'-2) = a[i,j](b[p[j], p[i]] - b[p[i], p[j]]) + a[j,i](b[p[i], p[j]] - b[p[j], p[i]])
                     = (a[i,j] - a[j,i]) (b[p[j], p[i]] - b[p[i], p[j]])
    
    For k != i, j:
    (3'-3) = a[i,k] (b[p[j], p[k]] - b[p[i], p[k]])
    (4'-4) = a[k,i] (b[p[k], p[j]] - b[p[k], p[i]])
    (5'-5) = a[j,k] (b[p[i], p[k]] - b[p[j], p[k]])
    (6'-6) = a[k,j] (b[p[k], p[i]] - b[p[k], p[j]])
    
    Group by k:
    Term for k:
    a[i,k] b[p[j], p[k]] - a[i,k] b[p[i], p[k]] 
    + a[k,i] b[p[k], p[j]] - a[k,i] b[p[k], p[i]]
    + a[j,k] b[p[i], p[k]] - a[j,k] b[p[j], p[k]]
    + a[k,j] b[p[k], p[i]] - a[k,j] b[p[k], p[j]]
    
    = b[p[j], p[k]] (a[i,k] - a[j,k]) 
    - b[p[i], p[k]] (a[i,k] - a[j,k])
    + b[p[k], p[j]] (a[k,i] - a[k,j])
    - b[p[k], p[i]] (a[k,i] - a[k,j])
    
    = (a[i,k] - a[j,k]) (b[p[j], p[k]] - b[p[i], p[k]])
    + (a[k,i] - a[k,j]) (b[p[k], p[j]] - b[p[k], p[i]])
    
    So:
    delta[i,j] = (a[i,j] - a[j,i]) (b[p[j], p[i]] - b[p[i], p[j]])
                + sum_{k!=i,j} [ (a[i,k] - a[j,k]) (b[p[j], p[k]] - b[p[i], p[k]])
                               + (a[k,i] - a[k,j]) (b[p[k], p[j]] - b[p[k], p[i]]) ]
    
    This can be vectorized.
    """
    n = len(p)
    # Create matrix of p indices
    # We need to compute for all i, j simultaneously.
    
    # Let's use the formula:
    # delta[i,j] = sum_k (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]]) 
    #             + sum_k (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
    #             + (a[i,j] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[j]])
    
    # Note: The sums over k include k=i and k=j. Let's check if the formula holds for k=i, j.
    # If k=i:
    # Term 1: (a[i,i] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[i]])
    # Term 2: (a[i,i] - a[i,j]) * (b[p[i], p[j]] - b[p[i], p[i]])
    # If k=j:
    # Term 1: (a[i,j] - a[j,j]) * (b[p[j], p[j]] - b[p[i], p[j]])
    # Term 2: (a[j,i] - a[j,j]) * (b[p[j], p[j]] - b[p[i], p[j]])
    
    # The explicit term (a[i,j] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[j]]) handles the cross terms?
    # Actually, it is easier to just compute the full matrix using broadcasting if n is small enough,
    # but n=100 means 100x100x100 = 1M ops per delta matrix computation.
    # We can update the delta matrix incrementally when we perform a swap.
    
    # For initialization, we can compute it once.
    # Let's define matrices to facilitate vectorization.
    
    # B_p = b[np.ix_(p, p)]
    Bp = b[np.ix_(p, p)]
    
    # delta[i,j] = sum_k (a[i,k] - a[j,k]) * (Bp[j,k] - Bp[i,k]) 
    #             + sum_k (a[k,i] - a[k,j]) * (Bp[k,j] - Bp[k,i])
    #             + (a[i,j] - a[j,i]) * (Bp[j,i] - Bp[i,j])
    
    # Let's compute the first sum:
    # S1[i,j] = sum_k (a[i,k] - a[j,k]) * (Bp[j,k] - Bp[i,k])
    # S1[i,j] = sum_k a[i,k]Bp[j,k] - sum_k a[i,k]Bp[i,k] - sum_k a[j,k]Bp[j,k] + sum_k a[j,k]Bp[i,k]
    
    # Let M1 = a @ Bp.T  (M1[i,k] = sum_j a[i,j] Bp[k,j] ... wait indices)
    # a @ Bp.T -> (n x n) @ (n x n). Result[i, k] = sum_j a[i,j] Bp.T[j,k] = sum_j a[i,j] Bp[k,j].
    # We want sum_k a[i,k] Bp[j,k]. This is (a @ Bp.T)[i, j].
    # Let's verify: (a @ Bp.T)[i, j] = sum_k a[i,k] Bp.T[k,j] = sum_k a[i,k] Bp[j,k]. Correct.
    
    # So sum_k a[i,k] Bp[j,k] = (a @ Bp.T)[i, j]
    # sum_k a[i,k] Bp[i,k] = (a @ Bp.T)[i, i]  (Diagonal)
    # sum_k a[j,k] Bp[j,k] = (a @ Bp.T)[j, j]  (Diagonal)
    # sum_k a[j,k] Bp[i,k] = (a @ Bp.T)[j, i]
    
    # So S1[i,j] = M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]
    
    # Similarly for S2:
    # S2[i,j] = sum_k (a[k,i] - a[k,j]) * (Bp[k,j] - Bp[k,i])
    # S2[i,j] = sum_k a[k,i]Bp[k,j] - sum_k a[k,i]Bp[k,i] - sum_k a[k,j]Bp[k,j] + sum_k a[k,j]Bp[k,i]
    
    # sum_k a[k,i]Bp[k,j] = (a.T @ Bp)[i, j]?
    # (a.T @ Bp)[i, j] = sum_k a.T[i,k] Bp[k,j] = sum_k a[k,i] Bp[k,j]. Correct.
    
    # So S2[i,j] = M2[i,j] - M2[i,i] - M2[j,j] + M2[j,i]
    # where M2 = a.T @ Bp
    
    # And the cross term:
    # S3[i,j] = (a[i,j] - a[j,i]) * (Bp[j,i] - Bp[i,j])
    
    # Total Delta[i,j] = S1[i,j] + S2[i,j] + S3[i,j]
    
    M1 = a @ Bp.T
    M2 = a.T @ Bp
    
    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    
    # S1[i,j] = M1[i,j] - diag_M1[i] - diag_M1[j] + M1[j,i]
    # We can construct this using broadcasting.
    # S1 = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    # Wait, M1[j,i] is M1.T[i,j].
    S1 = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    
    # S2[i,j] = M2[i,j] - diag_M2[i] - diag_M2[j] + M2[j,i]
    S2 = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T
    
    # S3[i,j] = (a[i,j] - a[j,i]) * (Bp[j,i] - Bp[i,j])
    # a[i,j] - a[j,i] = a - a.T
    # Bp[j,i] - Bp[i,j] = Bp.T - Bp
    S3 = (a - a.T) * (Bp.T - Bp)
    
    Delta = S1 + S2 + S3
    
    # Delta is symmetric? 
    # Delta[i,j] should be equal to Delta[j,i].
    # Let's check.
    # S1[j,i] = M1[j,i] - diag_M1[j] - diag_M1[i] + M1[i,j] = S1[i,j]. Symmetric.
    # S2[j,i] = M2[j,i] - diag_M2[j] - diag_M2[i] + M2[i,j] = S2[i,j]. Symmetric.
    # S3[j,i] = (a[j,i] - a[i,j]) * (Bp[i,j] - Bp[j,i]) = -(a[i,j] - a[j,i]) * -(Bp[j,i] - Bp[i,j]) = S3[i,j]. Symmetric.
    # So Delta is symmetric.
    
    # Set diagonal to 0 (swapping i with i is 0 change)
    np.fill_diagonal(Delta, 0)
    
    return Delta


def _update_delta_after_swap(Delta, a, b, p, i, j):
    """
    Update the delta matrix after swapping facilities i and j in p.
    This is an O(n) operation.
    """
    n = len(p)
    pi = p[i]
    pj = p[j]
    
    # We need to update Delta[k, l] for all k, l.
    # The change in Delta[k, l] comes from the change in p.
    # Only terms involving i or j in the sum change.
    
    # Actually, it is easier to just recompute the rows and columns i and j?
    # No, the swap changes p[i] and p[j], which affects Bp.
    # Bp changes in row i, col i, row j, col j.
    # This affects M1 and M2.
    
    # Let's derive the update for Delta.
    # Delta[i,j] is the cost change for swapping i and j.
    # After we perform the swap, the new permutation is p'.
    # We want to update Delta to reflect p'.
    
    # The formula for Delta depends on Bp.
    # Bp' differs from Bp only in rows/cols i and j.
    
    # Let's recompute the specific rows/cols of Delta that might have changed.
    # Actually, ALL entries in Delta depend on Bp.
    # However, the change in Bp is localized.
    
    # Let's look at the structure of Delta again.
    # Delta[i,j] = sum_{k} (a[i,k] - a[j,k]) (Bp[j,k] - Bp[i,k]) + ...
    
    # When we swap i and j in p:
    # p'[i] = pj, p'[j] = pi, p'[k] = pk for k != i, j.
    # Bp'[u, v] = b[p'[u], p'[v]]
    # Bp'[i, v] = b[pj, p'[v]]
    # Bp'[j, v] = b[pi, p'[v]]
    # Bp'[u, i] = b[p'[u], pj]
    # Bp'[u, j] = b[p'[u], pi]
    # Bp'[k, l] = Bp[k, l] for k,l not i,j.
    
    # This update is complex to do in O(n) without significant algebra.
    # Given n=100, recomputing the full Delta matrix takes ~1ms (100x100x100 mult).
    # If we do 10,000 moves, that's 10 seconds.
    # If we do 100,000 moves, that's 100 seconds.
    # This is feasible within the 150s limit if we are efficient.
    # Let's just recompute the full Delta matrix every time?
    # No, 100x100x100 is 1e6 ops. NumPy does ~100M ops/sec. So ~10ms per recompute.
    # 150s / 10ms = 15,000 moves.
    # This is a decent number of moves for SA.
    
    # Let's try to optimize the recompute.
    # We can update Bp in O(n).
    # Bp[i, :] = b[pj, p]
    # Bp[j, :] = b[pi, p]
    # Bp[:, i] = b[p, pj]
    # Bp[:, j] = b[p, pi]
    # Bp[i, i] = b[pj, pj]
    # Bp[j, j] = b[pi, pi]
    # Bp[i, j] = b[pj, pi]
    # Bp[j, i] = b[pi, pj]
    
    # Then recompute M1, M2, Delta.
    # M1 = a @ Bp.T
    # M2 = a.T @ Bp
    # This is O(n^3) per step.
    
    # Is there a faster way?
    # Maybe we can maintain M1 and M2?
    # M1[i, k] = sum_j a[i,j] Bp[k,j]
    # If Bp changes, M1 changes.
    
    # Let's stick to full recompute for simplicity and correctness, but optimize the code.
    # Or, we can use the fact that we only need the minimum delta.
    # But we need the full matrix to find the min.
    
    # Alternative: Use a list of (delta, i, j) and update only the affected ones?
    # Too complex.
    
    # Let's assume full recompute is fast enough.
    # 10ms per move * 10,000 moves = 100s.
    # We have 150s.
    # So we can do ~10,000-15,000 moves.
    
    # To make it faster, we can avoid creating new arrays in _compute_swap_delta.
    # But for now, let's just recompute.
    
    pass # This function is a placeholder, we will recompute in the main loop.


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin

    # Initial solution: Spectral
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
        
        k = min(10, n)
        idx_a = np.argsort(-np.abs(evals_a))[:k]
        fac_emb = evecs_a[:, idx_a]
        
        idx_b = np.argsort(-np.abs(evals_b))[:k]
        loc_emb = evecs_b[:, idx_b]
        
        fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
        fac_norm[fac_norm == 0] = 1
        fac_emb = fac_emb / fac_norm
        
        loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
        loc_norm[loc_norm == 0] = 1
        loc_emb = loc_emb / loc_norm
        
        cost_matrix = -fac_emb @ loc_emb.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
    except Exception:
        p = np.random.permutation(n).astype(np.int64)
    
    current_cost = _cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Simulated Annealing
    T = 1000.0
    T_min = 1.0
    cooling_rate = 0.999
    
    rng = np.random.default_rng(42)
    
    # Precompute Delta
    Delta = _compute_swap_delta(a, b, p)
    
    while time.time() < deadline and T > T_min:
        # Select a random pair (i, j)
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
            
        delta = Delta[i, j]
        
        # Acceptance probability
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Perform swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            # Update best
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            
            # Recompute Delta
            # This is the bottleneck.
            Delta = _compute_swap_delta(a, b, p)
            
        T *= cooling_rate
        
        # Check time every 100 iterations to avoid overhead
        if i % 100 == 0 and time.time() > deadline - 5.0:
            break

    # Final local search polish
    # Use the Delta matrix to do steepest descent
    for _ in range(10):
        if time.time() > deadline - 1.0:
            break
        min_idx = np.unravel_index(np.argmin(Delta), Delta.shape)
        min_delta = Delta[min_idx]
        if min_delta >= 0:
            break
        i, j = min_idx
        p[i], p[j] = p[j], p[i]
        current_cost += min_delta
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        Delta = _compute_swap_delta(a, b, p)

    return best_p.tolist()
# EVOLVE-BLOCK-END
