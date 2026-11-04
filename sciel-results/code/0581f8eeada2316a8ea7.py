"""QAP solver for tho150 using MDS+Procrustes init and fast incremental local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit
    
    # Convert to float64 for fast arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    def compute_cost(p):
        """Compute exact cost using vectorized matrix multiplication."""
        return np.sum(a_f * b_f[np.ix_(p, p)])
    
    def mds_procrustes_init(a_mat, b_mat):
        """
        Use MDS to embed A and B into Euclidean space, then solve Procrustes 
        to find the best rotation/alignment, yielding a permutation.
        """
        def mds_embedding(matrix, dim=4):
            # Center the matrix
            M = matrix.copy()
            M -= M.mean(axis=0, keepdims=True)
            M -= M.mean(axis=1, keepdims=True)
            
            # Compute double centered matrix
            # J = I - (1/n) * 11^T
            n_pts = len(matrix)
            J = np.eye(n_pts) - np.ones((n_pts, n_pts)) / n_pts
            B = -0.5 * J @ (M ** 2) @ J
            
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(B)
            
            # Take top k eigenvalues
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx[:dim]]
            eigvecs = eigvecs[:, idx[:dim]]
            
            # Embedding
            coords = eigvecs * np.sqrt(np.maximum(eigvals, 0))
            return coords
        
        # Embed facilities (A) and locations (B)
        # Note: A is flow, B is distance. 
        # For QAP, we want to align high flow with low distance.
        # MDS on A gives "flow geometry", MDS on B gives "distance geometry".
        # We want to place facility i at location p[i] such that 
        # sum a[i,j] b[p[i], p[j]] is minimized.
        
        # This is not a direct Procrustes fit of A to B.
        # However, a common heuristic is:
        # 1. Embed A to get facility coordinates X
        # 2. Embed B to get location coordinates Y
        # 3. Find permutation that minimizes ||X - Y P||_F (Procrustes)
        
        X = mds_embedding(a_f, dim=4)
        Y = mds_embedding(b_f, dim=4)
        
        # Solve Procrustes: minimize ||X - Y P||
        # This is equivalent to finding the best rotation/reflection R
        # X R^T approx Y
        # But we need a permutation, not a rotation.
        # So we use the rotation to get an "aligned" set of points, 
        # then solve the assignment problem between X and Y.
        
        # Actually, simpler: just solve the linear assignment problem
        # minimizing sum ||x_i - y_{p[i]}||^2
        # This is a good proxy for the QAP objective if the MDS captures structure.
        
        # Compute cost matrix for assignment
        # C[i, j] = ||x_i - y_j||^2
        # Expand: ||x_i||^2 + ||y_j||^2 - 2 x_i . y_j
        # Since ||x_i||^2 and ||y_j||^2 are independent of the match, 
        # we can just minimize -2 sum x_i . y_{p[i]}
        # which is equivalent to maximizing sum x_i . y_{p[i]}
        
        # Dot product matrix
        D = X @ Y.T
        
        # Solve assignment
        # linear_sum_assignment minimizes cost, so we use -D
        row_ind, col_ind = linear_sum_assignment(-D)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p
    
    def fast_local_search(p, current_cost, end_time, seed=12345):
        """
        Fast 2-swap local search using precomputed delta arrays.
        For a swap of i and j, delta = sum_k a[i,k] (b[j,p[k]] - b[i,p[k]]) + ...
        We can precompute vectors to make each delta O(n) or use vectorized ops.
        Actually, O(n) per swap is still slow for n=150 if we do many swaps.
        Let's use the standard O(n) delta computation but vectorized across pairs? No.
        
        Better: Use the "precomputed sums" trick.
        Let C[i, j] = sum_k a[i, k] b[j, p[k]]
        Then swapping i and j changes cost by:
        delta = C[j, i] + C[i, j] - C[i, i] - C[j, j] - (sum_k a[i,k] b[j,p[k]] + sum_k a[j,k] b[i,p[k]] - sum_k a[i,k] b[i,p[k]] - sum_k a[j,k] b[j,p[k]])
        Wait, the standard formula is:
        delta = sum_{k != i, j} a[i, k] (b[j, p[k]] - b[i, p[k]]) + sum_{k != i, j} a[k, i] (b[p[k], j] - b[p[k], i]) + a[i, j] (b[j, i] - b[i, j]) + a[j, i] (b[i, j] - b[j, i]) + a[i, i] (b[j, j] - b[i, i]) + a[j, j] (b[i, i] - b[j, j])
        
        This is O(n) per swap. With n=150, that's 150 ops per swap.
        If we do 10,000 swaps, that's 1.5M ops, which is fast in NumPy.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        
        # Precompute b rows for current permutation
        # We need b[p[i], p[j]] quickly.
        
        while time.time() < end_time:
            improved = False
            # Generate random pairs to check
            # First improvement strategy: check random pairs, if one improves, do it and restart pair generation?
            # Or check all pairs? All pairs is n*(n-1)/2 = ~11k pairs.
            # 11k * 150 ops = 1.65M ops per full pass.
            # If we do 100 passes, 165M ops. Might be too slow in Python loop.
            # Let's use random sampling of pairs.
            
            # Check 1000 random pairs per iteration
            for _ in range(1000):
                if time.time() > end_time:
                    break
                
                i, j = rng.choice(n, 2, replace=False)
                
                li = curr_p[i]
                lj = curr_p[j]
                
                if li == lj:
                    continue
                
                # Compute delta
                # delta = a[i,i](b[lj,lj]-b[li,li]) + a[j,j](b[li,li]-b[lj,lj])
                #       + a[i,j](b[lj,li]-b[li,lj]) + a[j,i](b[li,lj]-b[lj,li])
                #       + sum_{k!=i,j} a[i,k](b[lj,p[k]]-b[li,p[k]])
                #       + sum_{k!=i,j} a[k,i](b[p[k],lj]-b[p[k],li])
                #       + sum_{k!=i,j} a[j,k](b[li,p[k]]-b[lj,p[k]])
                #       + sum_{k!=i,j} a[k,j](b[p[k],li]-b[p[k],lj])
                
                # Vectorized computation
                # Create mask for k != i, j
                # Instead of masking, compute full sums and subtract terms i and j
                
                # S1 = sum_k a[i,k] b[lj, p[k]]
                # S2 = sum_k a[i,k] b[li, p[k]]
                # S3 = sum_k a[k,i] b[p[k], lj]
                # S4 = sum_k a[k,i] b[p[k], li]
                # S5 = sum_k a[j,k] b[li, p[k]]
                # S6 = sum_k a[j,k] b[lj, p[k]]
                # S7 = sum_k a[k,j] b[p[k], li]
                # S8 = sum_k a[k,j] b[p[k], lj]
                
                # Note: b_f is symmetric? Not necessarily, but QAP usually assumes symmetric.
                # Let's assume general case.
                
                # p_k = curr_p
                pk = curr_p
                
                # b_lj_pk = b_f[lj, pk]
                # b_li_pk = b_f[li, pk]
                # b_pk_lj = b_f[pk, lj]
                # b_pk_li = b_f[pk, li]
                
                b_lj_pk = b_f[lj, pk]
                b_li_pk = b_f[li, pk]
                b_pk_lj = b_f[pk, lj]
                b_pk_li = b_f[pk, li]
                
                S1 = np.dot(a_f[i, :], b_lj_pk)
                S2 = np.dot(a_f[i, :], b_li_pk)
                S3 = np.dot(a_f[:, i], b_pk_lj)
                S4 = np.dot(a_f[:, i], b_pk_li)
                S5 = np.dot(a_f[j, :], b_li_pk)
                S6 = np.dot(a_f[j, :], b_lj_pk)
                S7 = np.dot(a_f[:, j], b_pk_li)
                S8 = np.dot(a_f[:, j], b_pk_lj)
                
                # Subtract terms where k=i or k=j
                # For S1 (k=i): a[i,i] b[lj, li]
                # For S1 (k=j): a[i,j] b[lj, lj]
                S1 -= a_f[i, i] * b_f[lj, li] + a_f[i, j] * b_f[lj, lj]
                
                # For S2 (k=i): a[i,i] b[li, li]
                # For S2 (k=j): a[i,j] b[li, lj]
                S2 -= a_f[i, i] * b_f[li, li] + a_f[i, j] * b_f[li, lj]
                
                # For S3 (k=i): a[i,i] b[li, lj]
                # For S3 (k=j): a[j,i] b[lj, lj]
                S3 -= a_f[i, i] * b_f[li, lj] + a_f[j, i] * b_f[lj, lj]
                
                # For S4 (k=i): a[i,i] b[li, li]
                # For S4 (k=j): a[j,i] b[lj, li]
                S4 -= a_f[i, i] * b_f[li, li] + a_f[j, i] * b_f[lj, li]
                
                # For S5 (k=i): a[j,i] b[li, li]
                # For S5 (k=j): a[j,j] b[li, lj]
                S5 -= a_f[j, i] * b_f[li, li] + a_f[j, j] * b_f[li, lj]
                
                # For S6 (k=i): a[j,i] b[lj, li]
                # For S6 (k=j): a[j,j] b[lj, lj]
                S6 -= a_f[j, i] * b_f[lj, li] + a_f[j, j] * b_f[lj, lj]
                
                # For S7 (k=i): a[i,j] b[li, li]
                # For S7 (k=j): a[j,j] b[lj, li]
                S7 -= a_f[i, j] * b_f[li, li] + a_f[j, j] * b_f[lj, li]
                
                # For S8 (k=i): a[i,j] b[li, lj]
                # For S8 (k=j): a[j,j] b[lj, lj]
                S8 -= a_f[i, j] * b_f[li, lj] + a_f[j, j] * b_f[lj, lj]
                
                delta = (S1 - S2) + (S3 - S4) + (S5 - S6) + (S7 - S8)
                
                # Add diagonal and cross terms explicitly if not included above?
                # The sums S1..S8 included k=i and k=j terms which we subtracted.
                # The formula for delta is:
                # sum_{k!=i,j} a[i,k](b[lj,p[k]]-b[li,p[k]]) = S1 - S2 - (terms for k=i,j)
                # We subtracted the terms for k=i,j from S1 and S2.
                # So (S1-S2) is the sum for k!=i,j.
                # Similarly for others.
                # But we also need the terms where k=i or k=j in the original sum?
                # Original delta = sum_k a[i,k](b[lj,p[k]]-b[li,p[k]]) + sum_k a[k,i](b[p[k],lj]-b[p[k],li]) + ...
                # When k=i: a[i,i](b[lj,li]-b[li,li])
                # When k=j: a[i,j](b[lj,lj]-b[li,lj])
                # These are NOT in the "k!=i,j" sum.
                # So we must add them back.
                
                # Term k=i for first sum: a[i,i](b[lj,li]-b[li,li])
                # Term k=j for first sum: a[i,j](b[lj,lj]-b[li,lj])
                # Term k=i for second sum: a[i,i](b[li,lj]-b[li,li])
                # Term k=j for second sum: a[j,i](b[lj,lj]-b[lj,li])
                # Term k=i for third sum: a[j,i](b[li,li]-b[lj,li])
                # Term k=j for third sum: a[j,j](b[li,lj]-b[lj,lj])
                # Term k=i for fourth sum: a[i,j](b[li,li]-b[li,lj])
                # Term k=j for fourth sum: a[j,j](b[lj,li]-b[lj,lj])
                
                extra = 0.0
                extra += a_f[i, i] * (b_f[lj, li] - b_f[li, li])
                extra += a_f[i, j] * (b_f[lj, lj] - b_f[li, lj])
                extra += a_f[i, i] * (b_f[li, lj] - b_f[li, li])
                extra += a_f[j, i] * (b_f[lj, lj] - b_f[lj, li])
                extra += a_f[j, i] * (b_f[li, li] - b_f[lj, li])
                extra += a_f[j, j] * (b_f[li, lj] - b_f[lj, lj])
                extra += a_f[i, j] * (b_f[li, li] - b_f[li, lj])
                extra += a_f[j, j] * (b_f[lj, li] - b_f[lj, lj])
                
                delta += extra
                
                if delta < 0:
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                    curr_cost += delta
                    improved = True
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
                    break # Restart pair generation after improvement
            
            if not improved:
                break
                
        return best_p, best_cost
    
    # 1. MDS + Procrustes Initialization
    p_init = mds_procrustes_init(a_f, b_f)
    cost_init = compute_cost(p_init)
    
    best_p = p_init.copy()
    best_cost = cost_init
    
    # 2. Fast Local Search
    end_time_lk = deadline - 5.0
    p_lk, cost_lk = fast_local_search(best_p, best_cost, end_time_lk, seed=12345)
    if cost_lk < best_cost:
        best_p = p_lk
        best_cost = cost_lk
        
    # 3. If time remains, try a second restart with different seed
    if time.time() < deadline - 2.0:
        # Perturb best solution
        rng_perturb = np.random.RandomState(789)
        p_pert = best_p.copy()
        for _ in range(10):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
        cost_pert = compute_cost(p_pert)
        p_lk2, cost_lk2 = fast_local_search(p_pert, cost_pert, deadline - 0.5, seed=999)
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2

    return best_p.tolist()
# EVOLVE-BLOCK-END
