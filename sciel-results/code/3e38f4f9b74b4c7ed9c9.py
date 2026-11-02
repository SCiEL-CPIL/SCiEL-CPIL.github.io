import numpy as np
import random
import time

DIMENSIONS = [2, 3, 4]

def verify_decomp(U, V, W, n, m, p):
    """
    Verifies if U, V, W form an exact decomposition of the matrix multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    R = U.shape[1]
    # Flatten U, V, W to 1D arrays for faster dot products?
    # Or just loop.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def find_sub_decomp(n, m, p, max_rank, max_iters=10000):
    """
    Finds a decomposition of rank <= max_rank for n x m x p multiplication tensor.
    Returns (U, V, W, rank) or None.
    """
    # Target: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # Try ranks from 1 upwards? No, start from a guess.
    # For 2x3x2, schoolbook is 12. Try 11, 10.
    for R in range(max_rank, 0, -1):
        for _ in range(max_iters // max_rank):
            # Random initialization with small integers/half-integers
            # Use -1, 0, 1 for simplicity first
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            W = np.random.choice([-1, 0, 1], size=(p * n, R))
            
            if verify_decomp(U, V, W, n, m, p):
                return U, V, W, R
    return None

def construct():
    n, m, p = DIMENSIONS
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W

    # Try to find a better decomposition by splitting p into two halves
    # If p is even, we can split into two p/2 parts.
    if p % 2 == 0:
        p_half = p // 2
        # Search for decomposition of n x m x p_half
        # Time limit guard
        start_time = time.time()
        timeout = 10.0 # seconds
        
        # Try ranks lower than schoolbook (n*m*p_half)
        schoolbook_half = n * m * p_half
        found = None
        # Try R from schoolbook_half - 1 down to some lower bound
        # For 2x3x2, schoolbook is 12. Try 11, 10.
        for R in range(schoolbook_half - 1, 5, -1):
            if time.time() - start_time > timeout:
                break
            # Random search
            # Increase iterations as R gets smaller?
            iters = 2000
            for _ in range(iters):
                U_sub = np.random.choice([-1, 0, 1], size=(n * m, R))
                V_sub = np.random.choice([-1, 0, 1], size=(m * p_half, R))
                W_sub = np.random.choice([-1, 0, 1], size=(p_half * n, R))
                
                if verify_decomp(U_sub, V_sub, W_sub, n, m, p_half):
                    found = (U_sub, V_sub, W_sub, R)
                    break
            if found:
                break
        
        if found:
            U_sub, V_sub, W_sub, R_sub = found
            # Construct full decomposition
            # The full tensor is block diagonal in k?
            # C = A B. B = [B1, B2]. C = [A B1, A B2].
            # The terms for k in 0..p_half-1 use U, V1, W1.
            # The terms for k in p_half..p-1 use U, V2, W2.
            # Note: U is the same for both blocks?
            # U depends on i, j. Yes, i and j are the same.
            # So we can reuse U_sub for both blocks.
            # V is split: V_full[:, 0:R_sub] = V_sub for k=0..p_half-1?
            # V_full row index is j*p + k.
            # For the first block, k' = k. Row j*p + k.
            # For the second block, k' = k + p_half. Row j*p + (k+p_half).
            
            total_rank = 2 * R_sub
            U_full = np.zeros((n * m, total_rank))
            V_full = np.zeros((m * p, total_rank))
            W_full = np.zeros((p * n, total_rank))
            
            # First block (k in 0..p_half-1)
            U_full[:, 0:R_sub] = U_sub
            # V_sub has rows j*p_half + k. We need to map to j*p + k.
            for j in range(m):
                for k in range(p_half):
                    V_full[j * p + k, 0:R_sub] = V_sub[j * p_half + k, :]
            # W_sub has rows k*n + i. We need to map to k*n + i.
            # Since k is in 0..p_half-1, the row index in W_full is k*n + i.
            # This matches W_sub row index directly.
            W_full[0:p_half * n, 0:R_sub] = W_sub
            
            # Second block (k in p_half..p-1)
            U_full[:, R_sub:2*R_sub] = U_sub
            for j in range(m):
                for k in range(p_half):
                    # k_full = k + p_half
                    V_full[j * p + (k + p_half), R_sub:2*R_sub] = V_sub[j * p_half + k, :]
            # W rows for k in p_half..p-1.
            # Row index in W_full: (k + p_half) * n + i.
            # W_sub row index: k * n + i.
            W_full[p_half * n:p * n, R_sub:2*R_sub] = W_sub
            
            # Verify full decomposition
            if verify_decomp(U_full, V_full, W_full, n, m, p):
                # Convert to lists
                return {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist()
                }
            else:
                # Error in construction? Fallback
                pass

    return {"U": U, "V": V, "W": W}
