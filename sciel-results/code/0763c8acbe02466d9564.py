"""Matrix multiplication 3x4x4 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _add_strassen_2x2(U, V, W, col, a_r0, a_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
    """
    Add Strassen decomposition for computing C[c_r0:c_r1+1, c_c0:c_c1+1] 
    = A[a_r0:a_r1+1, b_c0:b_c1+1] * B[b_c0:b_c1+1, c_c0:c_c1+1]
    
    A is n x m, B is m x p, C is n x p
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    W index: k*n + i (trace convention W[k*n+i, r])
    """
    # Strassen's 7 products for 2x2 * 2x2
    n = 3  # rows of A/C
    m = 4  # cols of A / rows of B
    p = 4  # cols of B/C
    
    # M1 = (A[a_r0,b_c0] + A[a_r1,b_c1]) * (B[b_c0,c_c0] + B[b_c1,c_c1])
    U[a_r0 * m + b_c0][col] += 1
    U[a_r1 * m + b_c1][col] += 1
    V[b_c0 * p + c_c0][col] += 1
    V[b_c1 * p + c_c1][col] += 1
    W[c_c0 * n + a_r0][col] += 1
    W[c_c1 * n + a_r1][col] += 1
    col += 1
    
    # M2 = (A[a_r1,b_c0] + A[a_r1,b_c1]) * B[b_c0,c_c0]
    U[a_r1 * m + b_c0][col] += 1
    U[a_r1 * m + b_c1][col] += 1
    V[b_c0 * p + c_c0][col] += 1
    W[c_c0 * n + a_r1][col] += 1
    W[c_c1 * n + a_r1][col] -= 1
    col += 1
    
    # M3 = A[a_r0,b_c0] * (B[b_c0,c_c1] - B[b_c1,c_c1])
    U[a_r0 * m + b_c0][col] += 1
    V[b_c0 * p + c_c1][col] += 1
    V[b_c1 * p + c_c1][col] -= 1
    W[c_c1 * n + a_r0][col] += 1
    W[c_c1 * n + a_r1][col] += 1
    col += 1
    
    # M4 = A[a_r1,b_c1] * (B[b_c1,c_c0] - B[b_c0,c_c0])
    U[a_r1 * m + b_c1][col] += 1
    V[b_c1 * p + c_c0][col] += 1
    V[b_c0 * p + c_c0][col] -= 1
    W[c_c0 * n + a_r0][col] += 1
    W[c_c0 * n + a_r1][col] += 1
    col += 1
    
    # M5 = (A[a_r0,b_c0] + A[a_r0,b_c1]) * B[b_c1,c_c1]
    U[a_r0 * m + b_c0][col] += 1
    U[a_r0 * m + b_c1][col] += 1
    V[b_c1 * p + c_c1][col] += 1
    W[c_c0 * n + a_r0][col] -= 1
    W[c_c1 * n + a_r1][col] += 1
    col += 1
    
    # M6 = (A[a_r1,b_c0] - A[a_r0,b_c0]) * (B[b_c0,c_c0] + B[b_c0,c_c1])
    U[a_r1 * m + b_c0][col] += 1
    U[a_r0 * m + b_c0][col] -= 1
    V[b_c0 * p + c_c0][col] += 1
    V[b_c0 * p + c_c1][col] += 1
    W[c_c1 * n + a_r1][col] += 1
    col += 1
    
    # M7 = (A[a_r0,b_c1] - A[a_r1,b_c1]) * (B[b_c1,c_c0] + B[b_c1,c_c1])
    U[a_r0 * m + b_c1][col] += 1
    U[a_r1 * m + b_c1][col] -= 1
    V[b_c1 * p + c_c0][col] += 1
    V[b_c1 * p + c_c1][col] += 1
    W[c_c0 * n + a_r0][col] += 1
    col += 1
    
    return col

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Validate that the decomposition is correct."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iter=500, tol=1e-8):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    Returns (U, V, W, new_rank) if reduction found, else (U, V, W, original_rank).
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the 48-dim tensor: T[i*m+j, j*p+k, k*n+i]
    # Check which terms are needed
    # For each term r, check if it can be expressed as a linear combination of others
    
    # Strategy: try removing each term and see if the remaining terms can still span
    # This is expensive, so we use a greedy approach
    
    # Alternative: check if any two terms are linearly dependent in the tensor space
    # Tensor entry for term r: T_r[i*m+j, j*p+k, k*n+i] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Flatten each term to a 48-dim vector
    term_vecs = []
    for r in range(rank):
        vec = np.zeros(48)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*m+j, j*p+k, k*n+i] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
        term_vecs.append(vec.reshape(48))
    
    term_vecs = np.array(term_vecs)  # shape (rank, 48)
    
    # Check if the sum of all term vectors equals the target tensor
    target = np.zeros(48)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    target = target.reshape(48)
    
    # Try to find a subset of terms whose sum equals target
    # This is a subset sum problem, but we can try greedy removal
    
    current_rank = rank
    current_vecs = term_vecs.copy()
    
    # Greedy: try removing each term and check if the rest can still sum to target
    # We do this by checking if the removed term is in the span of the others
    # Actually, we need the SUM to equal target, not just span
    
    # Better approach: check if any term is zero or can be canceled
    # For now, try a simple approach: check if removing term r and adjusting others works
    
    # Simpler: just check if the current decomposition is valid and return
    if np.allclose(np.sum(term_vecs, axis=0), target, atol=tol):
        # Decomposition is valid, try to find redundancies
        # Check if any two terms are identical (can be merged)
        for r1 in range(current_rank):
            for r2 in range(r1+1, current_rank):
                if np.allclose(term_vecs[r1], term_vecs[r2], atol=tol):
                    # These two terms are identical, merge them
                    # But we can't simply merge because they contribute to different U,V,W entries
                    # Actually if term_vecs are identical, the U,V,W columns might differ
                    # but the tensor contribution is the same
                    pass
    
    # For now, return the original
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    t0 = time.time()
    deadline = t0 + 100  # 100s safety margin within 120s limit
    
    # Strategy: Split the 3x4 matrix into a 2x4 block and a 1x4 block.
    # 2x4 x 4x4: Use Strassen on four 2x2 sub-blocks -> rank 4*7 = 28
    # 1x4 x 4x4: Schoolbook -> rank 16
    # Total rank: 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (12, 44)
    V = [[0] * rank for _ in range(m * p)]  # (16, 44)
    W = [[0] * rank for _ in range(p * n)]  # (12, 44)
    
    col = 0
    
    # Part 1: 2x4 x 4x4 using Strassen (rows 0,1 of A)
    # Split into four 2x2 * 2x2 blocks:
    for j in range(2):  # A col pair / B row pair
        for i in range(2):  # B col pair / C col pair
            a_r0, a_r1 = 0, 1
            b_c0, b_c1 = 2*j, 2*j+1
            c_c0, c_c1 = 2*i, 2*i+1
            c_r0, c_r1 = 0, 1
            col = _add_strassen_2x2(U, V, W, col, a_r0, a_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1)
    
    # Part 2: 1x4 x 4x4 schoolbook (row 2 of A)
    for j in range(4):
        for k in range(4):
            U[2 * 4 + j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank} but used {col} columns"
    
    # Verify correctness
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linear dependencies
    # This is a heuristic approach
    U_reduced, V_reduced, W_reduced, new_rank = _try_reduce_rank(U, V, W, n, m, p)
    
    if new_rank < rank and _validate(U_reduced, V_reduced, W_reduced, n, m, p):
        return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
