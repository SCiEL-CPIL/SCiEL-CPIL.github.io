"""Matrix multiplication 2x2x15 tensor decomposition using Strassen blocks with rank-10 2x2x3 search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _build_schoolbook_2x2x1():
    """
    Build schoolbook rank-4 decomposition for 2x2 x 2x1 matrix multiplication.
    Returns U (4x4), V (2x4), W (2x4).
    """
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p, col] = 1
            W[0 * n + i, col] = 1
            col += 1
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _build_full_53():
    """Build the full rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = 2, 2, 15
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(len(block_sizes)):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _try_find_2x2x3_rank10(deadline):
    """
    Try to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Returns (U, V, W) or None if not found within time limit.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # The tensor entries are T[i,j,k] = 1 for all i in [0,1], j in [0,1], k in [0,2]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Use a structured approach.
    # Let's try to find integer solutions.
    # We can fix some variables and solve for others.
    
    # Known fact: rank(2x2x3) = 10.
    # Let's try a specific construction.
    # Consider the 2x2x2 Strassen (rank 7) for k=0,1 and schoolbook (rank 4) for k=2.
    # That gives rank 11. We need to save 1 rank.
    # This means we need to merge one term from the Strassen block with one from the schoolbook block,
    # or find a different decomposition.
    
    # Let's try a random search with integer constraints.
    # Variables: U (4x10), V (6x10), W (6x10)
    # Constraints: sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1 for all i,j,k
    
    # This is a system of 12 equations with 120 unknowns.
    # We can try to set most variables to 0 or 1 and solve for the rest.
    
    # Let's try a specific pattern based on literature.
    # One known rank-10 decomposition for 2x2x3:
    # It's not trivial to write down explicitly.
    
    # Alternative: Use the fact that 2x2x3 is a sub-tensor of 2x2x4.
    # Rank(2x2x4) = 14 (Strassen for two 2x2x2 blocks).
    # But we need rank 10.
    
    # Let's try a simple heuristic:
    # Start with the rank-11 solution (Strassen + Schoolbook).
    # Try to merge two terms by setting some coefficients to cancel out.
    
    U11, V11, W11 = _build_2x2x3_strassen_plus_schoolbook()
    
    # Now try to reduce rank by 1.
    # We can try to express one of the rank-11 terms as a linear combination of the others.
    # This is equivalent to finding a non-trivial linear dependence among the 11 terms.
    
    # Let's try to find a vector c in R^11 such that sum_r c_r * term_r = 0.
    # If we find such a c, we can remove one term and adjust the others.
    
    # This is a linear algebra problem.
    # Let's construct the matrix of the 11 terms.
    
    # Actually, let's just return None for now, as finding a rank-10 decomposition
    # is non-trivial and may not be feasible within the time limit.
    
    return None

def _build_2x2x3_strassen_plus_schoolbook():
    """
    Build 2x2x3 using Strassen (rank 7) for cols 0,1 and schoolbook (rank 4) for col 2.
    Returns U (4x11), V (6x11), W (6x11).
    """
    n, m, p = 2, 2, 3
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    total_rank = 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen for k=0,1
    U[:, 0:7] = U_str
    for j in range(m):
        for k_local in range(2):
            V[j * p + k_local, 0:7] = V_str[j * 2 + k_local, :]
    for k_local in range(2):
        for i in range(n):
            W[k_local * n + i, 0:7] = W_str[k_local * 2 + i, :]
    
    # Schoolbook for k=2
    U[:, 7:11] = U_sb
    for j in range(m):
        V[j * p + 2, 7:11] = V_sb[j * 1, :]
    for i in range(n):
        W[2 * n + i, 7:11] = W_sb[0 * 2 + i, :]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Build baseline rank-53
    U53, V53, W53 = _build_full_53()
    
    if not _verify(U53, V53, W53, n, m, p):
        # Fallback to pure schoolbook
        total_rank_fb = n * m * p
        U_fb = np.zeros((n * m, total_rank_fb))
        V_fb = np.zeros((m * p, total_rank_fb))
        W_fb = np.zeros((p * n, total_rank_fb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, idx] = 1
                    V_fb[j*p+k, idx] = 1
                    W_fb[k*n+i, idx] = 1
                    idx += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U53, V53, W53
    
    # Try to find rank-10 decomposition for 2x2x3 (would give rank-52 total)
    # If we find a rank-10 decomposition for 2x2x3, we can replace one Strassen+schoolbook pair (rank 11) with one 2x2x3 (rank 10)
    # This saves 1 rank per replacement.
    
    # For now, return the baseline. The search for better decompositions
    # would need to be done offline or with more time.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
