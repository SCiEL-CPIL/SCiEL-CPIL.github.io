import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Spectral Initialization
    # Compute eigenvectors of A and B
    # For symmetric matrices, we can use np.linalg.eigh
    # But QAP matrices might not be perfectly symmetric or positive definite.
    # We use SVD for robustness.
    try:
        _, _, Vt_a = np.linalg.svd(a)
        _, _, Vt_b = np.linalg.svd(b)
        # The columns of Vt_a.T and Vt_b.T are the eigenvectors
        # We want to match the "shape" of A and B
        # Typically, we align the first principal components
        vec_a = Vt_a.T[:, 0]  # First right singular vector of A
        vec_b = Vt_b.T[:, 0]  # First right singular vector of B
        
        # Rank match: sort facilities by vec_a, locations by vec_b
        # We want high flow facilities to go to central (low distance) locations?
        # Actually, spectral alignment usually means:
        # p[i] = argmin_j | vec_a[i] - vec_b[j] | ? No.
        # Standard spectral assignment:
        # Project facilities and locations onto the principal axes.
        # Let's use the standard rank matching on the eigenvectors.
        # Sort facilities by their score on the first eigenvector of A
        # Sort locations by their score on the first eigenvector of B
        # But we need to check if they should be positively or negatively correlated.
        # Usually, high flow should go to short distance.
        # Sum of A row i is roughly related to vec_a[i].
        # Sum of B row j is roughly related to vec_b[j].
        
        # Let's stick to the simple rank matching but using the spectral vectors
        # to break ties or improve the ordering.
        # Actually, the knowledge base says "Spectral initialization provides a geometrically informed starting point".
        # Let's use the dot product of the eigenvectors to define the assignment.
        # Maximize sum a[i,j] b[p[i], p[j]]? No, minimize.
        
        # A common spectral heuristic:
        # Let U_a, S_a, V_a^T = SVD(A)
        # Let U_b, S_b, V_b^T = SVD(B)
        # Assign facility i to location j that maximizes U_a[i, :] dot U_b[j, :] ?
        # Or just use the first principal component scores.
        
        scores_a = np.dot(a, Vt_a.T[:, 0])
        scores_b = np.dot(b, Vt_b.T[:, 0])
        
        # Sort facilities by scores_a descending
        # Sort locations by scores_b ascending (central locations have low scores?)
        # Actually, let's just use the standard rank matching on row sums as a fallback
        # if spectral doesn't seem to help, but spectral is usually better.
        
        idx_a = np.argsort(-scores_a)
        idx_b = np.argsort(scores_b)
        
        p = np.empty(n, dtype=np.int64)
        p[idx_a] = idx_b
        
    except Exception:
        # Fallback to simple rank matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central

    # Helper to compute cost
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    best_cost = compute_cost(p)
    best_p = p.copy()

    # Helper for local search
    def local_search(p, current_cost):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit:
                    return p, current_cost
                for j in range(i + 1, n):
                    # Calculate delta for swapping i and j
                    # Delta = 2 * sum_{k != i,j} (A[i,k] + A[j,k]) * (B[p[i], p[k]] - B[p[j], p[k]])
                    # We can compute this efficiently
                    pi = p[i]
                    pj = p[j]
                    
                    # Vectorized computation of the sum
                    # Mask to exclude i and j
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    
                    # Term: (A[i,k] + A[j,k]) * (B[pi, k] - B[pj, k])
                    # Note: k iterates over facility indices, but B is indexed by locations p[k]
                    # So B[pi, p[k]] - B[pj, p[k]]
                    
                    # We need to sum over k != i, j
                    # Let's create arrays
                    a_i_k = a[i, mask]
                    a_j_k = a[j, mask]
                    b_pi_k = b[pi, p[mask]]
                    b_pj_k = b[pj, p[mask]]
                    
                    delta = 2 * np.sum((a_i_k + a_j_k) * (b_pi_k - b_pj_k))
                    
                    # Add the term for k=i and k=j?
                    # The formula derived was for k != i, j.
                    # The terms involving i and j themselves:
                    # Original: A[i,i]B[pi,pi] + A[i,j]B[pi,pj] + A[j,i]B[pj,pi] + A[j,j]B[pj,pj]
                    # New: A[i,i]B[pj,pj] + A[i,j]B[pj,pi] + A[j,i]B[pi,pj] + A[j,j]B[pi,pi]
                    # Delta_self = A[i,i](B[pj,pj]-B[pi,pi]) + A[j,j](B[pi,pi]-B[pj,pj]) + 2*A[i,j]*(B[pj,pi]-B[pi,pj])
                    # Since B is symmetric, B[pj,pi]=B[pi,pj], so last term is 0.
                    # Delta_self = (A[i,i] - A[j,j]) * (B[pj,pj] - B[pi,pi])
                    
                    delta += (a[i, i] - a[j, j]) * (b[pj, pj] - b[pi, pi])
                    
                    if delta < 0:
                        # Perform swap
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        # Continue scanning or restart? First improvement usually continues
                        # But since p changed, the deltas for other pairs are invalid.
                        # Standard First Improvement: accept and break inner loop, restart outer?
                        # Or just accept and continue? If we continue, the current delta calculation
                        # for subsequent j's in the same i-loop is based on the OLD p[i].
                        # So we must break the inner loop and restart the outer loop.
                        break
                if improved:
                    # Restart outer loop
                    continue
        return p, current_cost

    # Initial local search
    p, best_cost = local_search(p, best_cost)
    best_p = p.copy()

    # Iterated Local Search
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        # Perturb
        p_current = best_p.copy()
        # Random number of swaps, e.g., 5-10
        num_swaps = rng.integers(5, 11)
        for _ in range(num_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p_current[i], p_current[j] = p_current[j], p_current[i]
        
        # Local search
        p_current, cost_current = local_search(p_current, compute_cost(p_current))
        
        if cost_current < best_cost:
            best_cost = cost_current
            best_p = p_current.copy()

    return best_p.tolist()
