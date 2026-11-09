import time
import numpy as np
from scipy.optimize import nnls # Might not be needed, np.linalg.lstsq is fine

DIMENSIONS = [3, 4, 8]

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_tensor(n, m, p):
    """Build the target tensor T as a list of (a,b,c,value) tuples for non-zero entries."""
    T = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T.append((a, b, c, 1.0))
    return T

def _als_decomposition(n, m, p, rank, max_iters=20, tol=1e-9):
    """
    Try to find a rank-r decomposition using Alternating Least Squares.
    Returns (U, V, W) if found, else None.
    """
    T = _build_tensor(n, m, p)
    if not T:
        return None
    
    # Flatten indices for efficient access
    # We need to solve linear systems.
    # For U: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Let M_U[a] be the matrix of size (num_T, rank) where M_U[a][idx, r] = V[b_idx, r] * W[c_idx, r]
    # We want to solve M_U[a] * U[a] = T_vals
    
    # Precompute indices
    a_indices = [t[0] for t in T]
    b_indices = [t[1] for t in T]
    c_indices = [t[2] for t in T]
    vals = [t[3] for t in T]
    
    num_T = len(T)
    
    # Initialize
    np.random.seed(42)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    for it in range(max_iters):
        # Solve for U
        for a in range(n * m):
            # Find all T entries with this 'a'
            mask = (np.array(a_indices) == a)
            if not np.any(mask):
                continue
            b_sub = np.array(b_indices)[mask]
            c_sub = np.array(c_indices)[mask]
            vals_sub = np.array(vals)[mask]
            
            # Build matrix: rows are T entries, cols are rank
            # M[idx, r] = V[b_sub[idx], r] * W[c_sub[idx], r]
            V_sub = V[b_sub] # shape (num_sub, rank)
            W_sub = W[c_sub] # shape (num_sub, rank)
            M = V_sub * W_sub # element-wise product
            
            # Solve M * U[a] = vals_sub
            if M.size == 0:
                continue
            sol, residuals, rank_m, s = np.linalg.lstsq(M, vals_sub, rcond=None)
            U[a] = sol
            
        # Solve for V
        for b in range(m * p):
            mask = (np.array(b_indices) == b)
            if not np.any(mask):
                continue
            a_sub = np.array(a_indices)[mask]
            c_sub = np.array(c_indices)[mask]
            vals_sub = np.array(vals)[mask]
            
            U_sub = U[a_sub]
            W_sub = W[c_sub]
            M = U_sub * W_sub
            sol, residuals, rank_m, s = np.linalg.lstsq(M, vals_sub, rcond=None)
            V[b] = sol
            
        # Solve for W
        for c in range(p * n):
            mask = (np.array(c_indices) == c)
            if not np.any(mask):
                continue
            a_sub = np.array(a_indices)[mask]
            b_sub = np.array(b_indices)[mask]
            vals_sub = np.array(vals)[mask]
            
            U_sub = U[a_sub]
            V_sub = V[b_sub]
            M = U_sub * V_sub
            sol, residuals, rank_m, s = np.linalg.lstsq(M, vals_sub, rcond=None)
            W[c] = sol
            
        # Check error
        error = 0.0
        for idx in range(num_T):
            a, b, c, val = T[idx]
            recon = np.dot(U[a], V[b] * W[c]) # sum_r U[a,r] V[b,r] W[c,r]
            error += (recon - val) ** 2
            
        if error < tol:
            return U, V, W
            
    return None

def _round_to_half_integers(U, V, W, n, m, p):
    """
    Try to round U, V, W to half-integers and check if it still works.
    """
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify
    T = _build_tensor(n, m, p)
    for a, b, c, val in T:
        recon = np.dot(U_rounded[a], V_rounded[b] * W_rounded[c])
        if abs(recon - val) > 1e-6:
            return None
            
    # Also check that zero entries remain zero?
    # The tensor T is 0 elsewhere. We only checked non-zero entries.
    # We must ensure that for all (a,b,c) NOT in T, the sum is 0.
    # This is expensive to check all 9216 entries, but we can sample or check all.
    # 9216 * rank (e.g. 90) * 3 ops is ~ 2.5 million ops. Fast enough.
    
    # Build list of all zero entries? No, just iterate all a,b,c.
    # n*m = 12, m*p = 32, p*n = 24. Total 9216.
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Check if (a,b,c) is in T
                # We can create a set of valid (a,b,c)
                # Or just compute and check if it's close to 0
                recon = np.dot(U_rounded[a], V_rounded[b] * W_rounded[c])
                # If it's not a valid entry, it should be 0.
                # But we don't know if it's valid without checking indices.
                # Let's just check if recon is close to 0.
                # If recon is not close to 0, it might be a valid entry we missed?
                # No, we verified all valid entries.
                # So if recon != 0 for an invalid entry, it's invalid.
                # But how do we know if it's invalid?
                # We can just check if recon is close to 0.
                # If it's not close to 0, the decomposition is bad.
                if abs(recon) > 1e-6:
                    # Check if this is actually a valid entry
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and i == i2 and k == k2):
                        return None
                    # If it IS a valid entry, we already checked it in the first loop?
                    # Yes, the first loop checked all valid entries.
                    # So if we are here, it's an invalid entry with non-zero recon.
                    return None

    return U_rounded, V_rounded, W_rounded

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Try to find a smaller decomposition
    # Start from rank 95 down to 80
    for rank in range(95, 79, -1):
        if time.time() > deadline:
            break
        
        # Try a few random restarts
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # ALS
            result = _als_decomposition(n, m, p, rank, max_iters=10)
            if result is not None:
                U, V, W = result
                rounded = _round_to_half_integers(U, V, W, n, m, p)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    # Convert to lists of lists
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
