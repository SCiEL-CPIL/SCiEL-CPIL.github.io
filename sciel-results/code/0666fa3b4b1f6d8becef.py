import numpy as np
import time

def solve(a, b):
    """
    Solves the QAP instance using Iterated Local Search (ILS) with 2-opt moves.
    """
    n = len(a)
    # Ensure float64 for numerical stability and speed in dot products
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    
    # 1. Initial Solution: Rank Matching
    # Heavy flows (high row sum in A) to Central locations (low row sum in B)
    a_sums = A.sum(axis=1)
    b_sums = B.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_sums, kind='stable')
    # Sort locations by ascending distance sum (central)
    central_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    def compute_cost(p):
        # Cost = sum a[i,j] * b[p[i], p[j]]
        # Can be computed as trace(A * B_p) where B_p is permuted B
        # Or simply:
        B_perm = B[np.ix_(p, p)]
        return np.sum(A * B_perm)

    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search: 2-Opt
    def local_search(p, cost, max_iters=1000, time_limit=None):
        """
        Performs best-improvement 2-opt local search.
        Returns (p, cost).
        """
        start_time = time.time()
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time_limit and (time.time() - start_time) > time_limit:
                break
                
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Evaluate all pairs (i, j) with i < j
            for i in range(n):
                ui = p[i]
                # Precompute part of delta for fixed i
                # Delta_ij = 2 * sum_k (A[i,k] - A[j,k]) * (B[p[k], p[j]] - B[p[k], ui]) - corrections
                
                # Vectorize over j?
                # Let's just loop j for clarity and acceptable speed for N=100
                for j in range(i + 1, n):
                    uj = p[j]
                    
                    # Calculate Delta
                    # Term 1: sum_k (A[i,k] - A[j,k]) * (B[p[k], uj] - B[p[k], ui])
                    # Let vec_diff_A = A[i, :] - A[j, :]
                    # Let vec_diff_B = B[p, uj] - B[p, ui]
                    
                    # Optimization: 
                    # We can precompute B_rows = B[p, :] ? No, p changes.
                    # But B is fixed.
                    # B[p, uj] is the column uj of B indexed by p? No.
                    # B[p[k], p[j]] means element (p[k], p[j]) of B.
                    # So for fixed j (uj), B[:, uj] is a column of B.
                    # We need B[p, uj] which is B[:, uj][p].
                    # And B[p, ui] which is B[:, ui][p].
                    
                    # This indexing is O(N) each.
                    # vec_diff_B = B[:, uj][p] - B[:, ui][p]
                    # vec_diff_A = A[i, :] - A[j, :]
                    
                    # Dot product
                    # delta_sum = np.dot(vec_diff_A, vec_diff_B)
                    
                    # Correction terms for k=i and k=j
                    # k=i: (A[i,i] - A[j,i]) * (B[ui, uj] - B[ui, ui])
                    # k=j: (A[i,j] - A[j,j]) * (B[uj, uj] - B[uj, ui])
                    
                    # Note: In the sum sum_k, k runs over all indices 0..n-1.
                    # The term for k=i uses p[i]=ui. B[p[i], uj] = B[ui, uj]. B[p[i], ui] = B[ui, ui].
                    # The term for k=j uses p[j]=uj. B[p[j], uj] = B[uj, uj]. B[p[j], ui] = B[uj, ui].
                    
                    # We can compute the full dot product and subtract these two terms.
                    
                    # To speed up, we can avoid full array creation if possible, but N=100 is small.
                    diff_A = A[i] - A[j]
                    # B[:, uj][p] is fancy indexing.
                    # Let's precompute B_cols = B.T ? No, B is not necessarily symmetric in storage, but QAPLIB is.
                    # Let's assume symmetry for B and A to simplify? 
                    # The task says "int64 NumPy arrays". QAPLIB instances are symmetric.
                    # If symmetric, B[u,v] = B[v,u].
                    
                    # Let's use the symmetry if available, but safe to not assume?
                    # Wil100 is symmetric.
                    
                    # Optimization:
                    # Precompute B_perm = B[p, :]? No, that's N x N.
                    # We need columns of B indexed by p.
                    # Let's just do the indexing.
                    
                    col_uj = B[:, uj]
                    col_ui = B[:, ui]
                    # vec_diff_B = col_uj[p] - col_ui[p]
                    
                    delta_sum = np.dot(diff_A, col_uj[p] - col_ui[p])
                    
                    # Subtract self terms
                    # k=i term: diff_A[i] * (col_uj[ui] - col_ui[ui])
                    # Note: col_uj[ui] is B[ui, uj]. col_ui[ui] is B[ui, ui].
                    term_i = diff_A[i] * (B[ui, uj] - B[ui, ui])
                    
                    # k=j term: diff_A[j] * (col_uj[uj] - col_ui[uj])
                    # Note: col_uj[uj] is B[uj, uj]. col_ui[uj] is B[uj, ui].
                    term_j = diff_A[j] * (B[uj, uj] - B[uj, ui])
                    
                    delta = 2.0 * (delta_sum - term_i - term_j)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        
            if best_i != -1:
                # Apply swap
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
                
        return p, cost

    # Run initial local search
    # Give it a generous time limit for the first pass
    t0 = time.time()
    p, current_cost = local_search(p, current_cost, max_iters=10000, time_limit=10)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
        
    # 3. Iterated Local Search (ILS)
    # Use remaining time to perturb and re-optimize
    t_start = time.time()
    time_limit_total = 110.0 # Safety margin for 120s limit
    
    while (time.time() - t_start) < time_limit_total:
        # Perturb the current best solution
        # Number of swaps to perturb: small, e.g., 2-5
        num_swaps = np.random.randint(2, 6)
        p_perturbed = best_p.copy()
        
        for _ in range(num_swaps):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i != j:
                p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
                
        # Calculate cost of perturbed solution
        # Approximate cost or exact? Exact is safer.
        # We can estimate the cost change, but exact is O(N^2) which is fine.
        B_perm = B[np.ix_(p_perturbed, p_perturbed)]
        cost_perturbed = np.sum(A * B_perm)
        
        # Run local search on perturbed solution
        # Limit time for this local search to ensure we finish
        remaining_time = time_limit_total - (time.time() - t_start)
        if remaining_time < 1.0:
            break
            
        # Allow local search to run for up to 50% of remaining time or 5s, whichever is smaller
        ls_time_limit = min(5.0, remaining_time * 0.5)
        
        p_new, cost_new = local_search(p_perturbed, cost_perturbed, max_iters=1000, time_limit=ls_time_limit)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            
    return best_p.tolist()
