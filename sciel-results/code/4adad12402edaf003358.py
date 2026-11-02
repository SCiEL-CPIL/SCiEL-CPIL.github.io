"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s
    
    deadline = time.time() + 145  # Safety margin
    
    def verify(U, V, W, tol=1e-8):
        """Check if sum_r U_r V_r W_r == T"""
        S = np.zeros((nm, mp, pn))
        for r in range(U.shape[1]):
            if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(S, T, atol=tol)
    
    def residual_norm(U, V, W):
        S = np.zeros((nm, mp, pn))
        for r in range(U.shape[1]):
            if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(S - T)

    def fast_als_reduce(U_init, V_init, W_init, max_iters=100, tol=1e-9):
        """
        Fast ALS for fixed rank r.
        U, V, W are (dim, r)
        """
        r = U_init.shape[1]
        if r == 0:
            return U_init, V_init, W_init
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Precompute reshapes for efficient dot products
        # To update U[:, idx]:
        # U[:, idx] = (T * V_idx * W_idx) / sum(V_idx * W_idx)^2
        # We can compute this by iterating over the non-zero entries of T?
        # T is sparse. It has n*m*p = 84 non-zeros.
        # This is a huge advantage.
        
        # Store indices of non-zeros in T
        nz_a, nz_b, nz_c = np.nonzero(T)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            for idx in range(r):
                v = V[:, idx]
                w = W[:, idx]
                # Denominator: sum_b,c (v[b]*w[c])^2
                # Since T is sparse, we only need to sum over (b,c) where T[a,b,c] is non-zero?
                # No, the formula is:
                # Minimize sum_{a,b,c} (T_{a,b,c} - U_{a,idx} v_{b,idx} w_{c,idx})^2 w.r.t U_{a,idx}
                # Derivative: -2 sum_{b,c} T_{a,b,c} v_{b,idx} w_{c,idx} + 2 U_{a,idx} sum_{b,c} (v_{b,idx} w_{c,idx})^2 = 0
                # U_{a,idx} = sum_{b,c} T_{a,b,c} v_{b,idx} w_{c,idx} / sum_{b,c} (v_{b,idx} w_{c,idx})^2
                
                # The denominator is constant for all 'a' for a fixed idx.
                # Denom = sum_{b,c} v_b^2 w_c^2 = (sum v^2) * (sum w^2)
                # Wait, sum_{b,c} (v_b w_c)^2 = (sum_b v_b^2) * (sum_c w_c^2)
                denom = np.dot(v, v) * np.dot(w, w)
                if denom < 1e-12:
                    U[:, idx] = 0
                    continue
                
                # Numerator for each a: sum_{b,c} T_{a,b,c} v_b w_c
                # Since T is sparse, for a given 'a', T_{a,b,c} is non-zero only for specific (b,c).
                # We can precompute for each 'a' the list of (b,c) and values.
                # But simpler: just use the sparse indices.
                # For each a, num_a = sum_{(b,c) in supp(T_a)} T_{a,b,c} v_b w_c
                
                # Efficient way:
                # Create a matrix M of shape (nm, mp*pn)? No, too big.
                # Let's just loop over non-zeros.
                # num = np.zeros(nm)
                # for k in range(len(nz_a)):
                #     a, b, c = nz_a[k], nz_b[k], nz_c[k]
                #     num[a] += T[a,b,c] * v[b] * w[c]
                
                # Vectorized version using sparse indices:
                # We need sum over k where nz_a[k] == a of T_val * v[nz_b[k]] * w[nz_c[k]]
                # This is a group sum.
                # Since nm is small (12), we can just do it in a loop or use bincount.
                # values = T[nz_a, nz_b, nz_c] * v[nz_b] * w[nz_c]
                # num = np.bincount(nz_a, weights=values, minlength=nm)
                
                values = T[nz_a, nz_b, nz_c] * v[nz_b] * w[nz_c]
                num = np.bincount(nz_a, weights=values, minlength=nm)
                U[:, idx] = num / denom

            # Update V
            for idx in range(r):
                u = U[:, idx]
                w = W[:, idx]
                denom = np.dot(u, u) * np.dot(w, w)
                if denom < 1e-12:
                    V[:, idx] = 0
                    continue
                
                values = T[nz_a, nz_b, nz_c] * u[nz_a] * w[nz_c]
                num = np.bincount(nz_b, weights=values, minlength=mp)
                V[:, idx] = num / denom

            # Update W
            for idx in range(r):
                u = U[:, idx]
                v = V[:, idx]
                denom = np.dot(u, u) * np.dot(v, v)
                if denom < 1e-12:
                    W[:, idx] = 0
                    continue
                
                values = T[nz_a, nz_b, nz_c] * u[nz_a] * v[nz_b]
                num = np.bincount(nz_c, weights=values, minlength=pn)
                W[:, idx] = num / denom

        return U, V, W

    def try_reduce_rank(target_rank, n_attempts=5):
        """
        Try to find a rank-target_rank decomposition.
        """
        for attempt in range(n_attempts):
            if time.time() > deadline:
                return None
            
            # Strategy: Start with schoolbook, pick 'target_rank' columns, then optimize.
            # If target_rank < 84, we must drop some columns.
            # Random selection of columns is bad.
            # Better: Start with random small vectors?
            
            if attempt == 0:
                # Random Gaussian
                scale = 1.0 / np.sqrt(target_rank)
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale
            else:
                # Perturbed schoolbook subset
                # Pick target_rank random columns from schoolbook
                cols = np.random.choice(n*m*p, target_rank, replace=False)
                U = U_s[:, cols].copy()
                V = V_s[:, cols].copy()
                W = W_s[:, cols].copy()
                # Add small noise to escape local minima
                U += np.random.randn(nm, target_rank) * 0.1
                V += np.random.randn(mp, target_rank) * 0.1
                W += np.random.randn(pn, target_rank) * 0.1
            
            U_opt, V_opt, W_opt = fast_als_reduce(U, V, W, max_iters=50)
            
            # Check if it converged to zero residual
            res = residual_norm(U_opt, V_opt, W_opt)
            if res < 1e-8:
                if verify(U_opt, V_opt, W_opt):
                    return U_opt, V_opt, W_opt
        
        return None

    # We know rank 84 works.
    # Lower bound is 28.
    # Let's try to find a rank < 84.
    # It is unlikely that rank 28 works for 3x4x7 without specific structure.
    # However, the problem asks for "rank-minimal".
    # If we can't find a better one, we return 84.
    # But maybe we can find a slightly better one?
    # Actually, for matrix multiplication, the rank is often exactly n*m*p unless there are specific cancellations.
    # For 2x2x2, Strassen gives 7 vs 8.
    # For 3x3x3, it's 23 (or 24?) vs 27.
    # For 3x4x7, is there a known algorithm?
    # The lower bound is max(nm, mp, pn) = 28.
    # Let's try ranks 28 to 83.
    
    # To save time, we only try a few ranks.
    # If we find one, we return it.
    
    # Heuristic: Try ranks close to the lower bound first? Or close to schoolbook?
    # Random search is more likely to find a valid decomposition for higher ranks.
    # Let's try a few ranks from 40 to 83.
    
    found = None
    # We will iterate downwards from 83 to 28.
    # But we have limited time.
    # Let's try 5 random ranks between 30 and 80.
    
    candidates = list(range(83, 27, -1))
    # Shuffle to prioritize? No, usually lower rank is harder.
    # Let's try the lowest few first with more attempts?
    # Or just a random sample.
    
    # Given the time limit, we can afford ~10-20 ALS runs.
    # Each ALS run is ~50 iters * 3 * r * (sparse ops).
    # Sparse ops are very fast.
    
    # Let's try ranks: 28, 29, 30, 31, 32, 35, 40, 50, 60, 70, 80
    test_ranks = [28, 29, 30, 31, 32, 35, 40, 45, 50, 60, 70, 80]
    
    for r in test_ranks:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        result = try_reduce_rank(r, n_attempts=3)
        if result is not None:
            U_opt, V_opt, W_opt = result
            best_rank = r
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            # If we found a low rank, we can stop or try to go lower.
            # Since we are iterating downwards, we will naturally try lower ones next if we re-order.
            # But we are iterating 83->28. So if we find 80, we will try 70 next.
            # This is good.
    
    # If we didn't find anything better than 84, we return 84.
    
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
