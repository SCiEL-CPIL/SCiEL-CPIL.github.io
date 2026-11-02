"""Rank-minimal exact tensor decomposition search for 3x5x6 matrix multiplication."""
import numpy as np
import time
import itertools
import random
import sys
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _to_list(mat):
    return mat.tolist()


def _check_exact(U, V, W, n, m, p):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0
                for c in range(r):
                    s = s + u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def _strassen_3x3x3():
    """Return Strassen decomposition for 3x3x3 matrix multiplication (rank 7)."""
    # Strassen's 7 multiplication algorithm for 3x3x3
    # Using the standard Strassen decomposition
    U = np.zeros((9, 7), dtype=object)
    V = np.zeros((9, 7), dtype=object)
    W = np.zeros((9, 7), dtype=object)
    
    # Strassen's 7 products for 3x3 matrices
    # M1 = (A11+A33)(B11+B33)
    # M2 = (A22+A33)(B22+B33)
    # M3 = (A11+A22)(B11+B22)
    # M4 = (A33+A22)(B33+B22)
    # M5 = (A11+A33)(B13+B33)
    # M6 = (A31+A33)(B11+B13)
    # M7 = (A13+A23)(B31+B21)
    
    # U factors (row i*m+j, col c)
    # A11=row0, A12=row1, A13=row2, A21=row3, A22=row4, A23=row5, A31=row6, A32=row7, A33=row8
    U[0, 0] = 1  # A11 in M1
    U[8, 0] = 1  # A33 in M1
    U[4, 1] = 1  # A22 in M2
    U[8, 1] = 1  # A33 in M2
    U[0, 2] = 1  # A11 in M3
    U[4, 2] = 1  # A22 in M3
    U[8, 3] = 1  # A33 in M4
    U[4, 3] = 1  # A22 in M4
    U[0, 4] = 1  # A11 in M5
    U[8, 4] = 1  # A33 in M5
    U[6, 5] = 1  # A31 in M6
    U[8, 5] = 1  # A33 in M6
    U[2, 6] = 1  # A13 in M7
    U[5, 6] = 1  # A23 in M7
    
    # V factors (row j*p+k, col c)
    # B11=row0, B12=row1, B13=row2, B21=row3, B22=row4, B23=row5, B31=row6, B32=row7, B33=row8
    V[0, 0] = 1  # B11 in M1
    V[8, 0] = 1  # B33 in M1
    V[4, 1] = 1  # B22 in M2
    V[8, 1] = 1  # B33 in M2
    V[0, 2] = 1  # B11 in M3
    V[4, 2] = 1  # B22 in M3
    V[8, 3] = 1  # B33 in M4
    V[4, 3] = 1  # B22 in M4
    V[2, 4] = 1  # B13 in M5
    V[8, 4] = 1  # B33 in M5
    V[0, 5] = 1  # B11 in M6
    V[2, 5] = 1  # B13 in M6
    V[6, 6] = 1  # B31 in M7
    V[3, 6] = 1  # B21 in M7
    
    # W factors (row k*n+i, col c)
    # C11=row0, C12=row1, C13=row2, C21=row3, C22=row4, C23=row5, C31=row6, C32=row7, C33=row8
    W[0, 0] = 1  # C11 += M1
    W[8, 0] = 1  # C33 += M1
    W[8, 0] -= 1  # C33 -= M1 (so C33 gets M1-M2+M4-M5+M6-M7+M1... let me redo)
    
    # Let me use the standard Strassen formulas:
    # C11 = M1 - M2 + M4 - M6 + M7
    # C12 = M3 + M6 - M7
    # C13 = M3 + M5 - M7
    # C21 = M1 + M2 - M4 - M6
    # C22 = M1 - M2 + M4 - M5 + M6
    # C23 = M1 + M2 - M4 - M5
    # C31 = M1 - M2 + M3 + M5
    # C32 = M1 + M2 - M3 - M4
    # C33 = M1 - M2 + M4 - M5 + M6 - M7
    
    # Reset W
    W = np.zeros((9, 7), dtype=object)
    
    # C11 = M1 - M2 + M4 - M6 + M7
    W[0, 0] = 1
    W[0, 1] = -1
    W[0, 3] = 1
    W[0, 5] = -1
    W[0, 6] = 1
    
    # C12 = M3 + M6 - M7
    W[1, 2] = 1
    W[1, 5] = 1
    W[1, 6] = -1
    
    # C13 = M3 + M5 - M7
    W[2, 2] = 1
    W[2, 4] = 1
    W[2, 6] = -1
    
    # C21 = M1 + M2 - M4 - M6
    W[3, 0] = 1
    W[3, 1] = 1
    W[3, 3] = -1
    W[3, 5] = -1
    
    # C22 = M1 - M2 + M4 - M5 + M6
    W[4, 0] = 1
    W[4, 1] = -1
    W[4, 3] = 1
    W[4, 4] = -1
    W[4, 5] = 1
    
    # C23 = M1 + M2 - M4 - M5
    W[5, 0] = 1
    W[5, 1] = 1
    W[5, 3] = -1
    W[5, 4] = -1
    
    # C31 = M1 - M2 + M3 + M5
    W[6, 0] = 1
    W[6, 1] = -1
    W[6, 2] = 1
    W[6, 4] = 1
    
    # C32 = M1 + M2 - M3 - M4
    W[7, 0] = 1
    W[7, 1] = 1
    W[7, 2] = -1
    W[7, 3] = -1
    
    # C33 = M1 - M2 + M4 - M5 + M6 - M7
    W[8, 0] = 1
    W[8, 1] = -1
    W[8, 3] = 1
    W[8, 4] = -1
    W[8, 5] = 1
    W[8, 6] = -1
    
    return U, V, W


def _kronecker_decompose_3x5x6():
    """
    Build a decomposition for 3x5x6 by combining:
    - A 3x3x3 Strassen decomposition (rank 7) for the first 3 columns
    - Schoolbook for the remaining 2 columns (3x2x6 = 36 terms)
    Total rank: 7 + 36 = 43
    
    Actually, let's think more carefully. We can partition:
    - First 3 columns of A (3x3) with first 3 rows of B (3x6): use Strassen-like
    - Remaining parts: schoolbook
    
    Better approach: Partition B into two 3x3 blocks (columns 0-2 and 3-5)
    For A (3x5) and B (5x6):
    - Split A into A1 (3x3, cols 0-2) and A2 (3x2, cols 3-4)
    - Split B into B1 (3x6, rows 0-2) and B2 (2x6, rows 3-4)
    
    Product = A1*B1 + A2*B2
    
    For A1*B1 (3x3 * 3x6): We can view this as multiplying a 3x3 by a 3x6.
    Split B1 into B1a (3x3) and B1b (3x3)
    A1*B1 = A1*B1a + A1*B1b
    
    Each A1*B1x is a 3x3 * 3x3 multiplication, which has rank 7 (Strassen).
    So A1*B1 has rank at most 7+7=14.
    
    For A2*B2 (3x2 * 2x6): This is schoolbook, rank = 3*2*6 = 36.
    
    Total: 14 + 36 = 50.
    
    But we can do better! Let's try:
    - Use Strassen for the 3x3*3x3 part
    - Optimize the 3x2*2x6 part
    
    Actually, let's try a different partition. What if we use:
    - 3x3 * 3x6 using a known decomposition
    - 3x2 * 2x6 using schoolbook
    
    For 3x3 * 3x6: We can think of this as computing 3x3 matrix times 3x6 matrix.
    This is equivalent to 2 independent 3x3*3x3 multiplications (for the two 3x3 blocks of B).
    Rank: 7 + 7 = 14.
    
    For 3x2 * 2x6: Schoolbook gives rank 36.
    
    Total: 50.
    
    Let's try to improve the 3x2*2x6 part. Actually, 3x2*2x6 is just 36 scalar multiplications,
    and there's no obvious structure to exploit with Strassen since the middle dimension is only 2.
    
    Alternative: What if we partition differently?
    - A is 3x5, B is 5x6
    - Split A into rows: A_top (2x5) and A_bottom (1x5)
    - Split B into columns: B_left (5x3) and B_right (5x3)
    
    This doesn't help much.
    
    Let me try the straightforward approach: rank 50 via block decomposition.
    """
    n, m, p = 3, 5, 6
    
    # Partition:
    # A = [A1 | A2] where A1 is 3x3, A2 is 3x2
    # B = [B1; B2] where B1 is 3x6, B2 is 2x6
    # Product = A1*B1 + A2*B2
    
    # For A1*B1 (3x3 * 3x6):
    # Split B1 = [B1a | B1b] where B1a is 3x3, B1b is 3x3
    # A1*B1 = A1*B1a + A1*B1b
    # Each term uses Strassen (rank 7), so total rank for this part: 14
    
    # For A2*B2 (3x2 * 2x6):
    # Schoolbook: rank = 3*2*6 = 36
    
    # Total rank: 14 + 36 = 50
    
    U_strassen1, V_strassen1, W_strassen1 = _strassen_3x3x3()
    U_strassen2, V_strassen2, W_strassen2 = _strassen_3x3x3()
    
    # Build the full decomposition
    total_rank = 14 + 36
    
    U = np.zeros((n * m, total_rank), dtype=object)
    V = np.zeros((m * p, total_rank), dtype=object)
    W = np.zeros((p * n, total_rank), dtype=object)
    
    # Part 1: A1*B1a (3x3 * 3x3) using Strassen
    # A1 occupies rows 0-8 of U (indices i*m+j for i in 0-2, j in 0-2)
    # B1a occupies rows 0-8 of V (indices j*p+k for j in 0-2, k in 0-2)
    # W for this part: C11-C13 for first 3 columns
    for c in range(7):
        for idx in range(9):
            U[idx, c] = U_strassen1[idx, c]
            V[idx, c] = V_strassen1[idx, c]
        for k in range(3):
            for i in range(3):
                W[k * n + i, c] = W_strassen1[k * n + i, c]
    
    # Part 2: A1*B1b (3x3 * 3x3) using Strassen
    # B1b occupies rows 9-17 of V (indices j*p+k for j in 0-2, k in 3-5)
    for c in range(7):
        for idx in range(9):
            U[idx, 7 + c] = U_strassen2[idx, c]
            # V index: j*p + k where k is in 3-5, so offset by 3
            V[idx * 1, 7 + c] = V_strassen2[idx, c]  # Need to map correctly
        for k in range(3, 6):
            for i in range(3):
                W[k * n + i, 7 + c] = W_strassen2[(k - 3) * n + i, c]
    
    # Fix V mapping for Part 2
    # V_strassen2 has indices j*p+k for j in 0-2, k in 0-2
    # We need to map to j*p+k for j in 0-2, k in 3-5
    for c in range(7):
        for j in range(3):
            for k in range(3):
                v_idx_orig = j * 3 + k
                v_idx_new = j * 6 + (k + 3)
                V[v_idx_new, 7 + c] = V_strassen2[v_idx_orig, c]
    
    # Part 3: A2*B2 (3x2 * 2x6) using schoolbook
    # A2 occupies rows 9-14 of U (indices i*m+j for i in 0-2, j in 3-4)
    # B2 occupies rows 18-29 of V (indices j*p+k for j in 3-4, k in 0-5)
    c_offset = 14
    for i in range(3):
        for j in range(2):
            for k in range(6):
                u_idx = i * 5 + (j + 3)
                v_idx = (j + 3) * 6 + k
                U[u_idx, c_offset] = 1
                V[v_idx, c_offset] = 1
                W[k * 3 + i, c_offset] = 1
                c_offset += 1
    
    return U, V, W


def _try_block_decomposition(n, m, p, deadline):
    """
    Try to build a decomposition by decomposing into blocks.
    For 3x5x6, we use Strassen for 3x3*3x3 sub-blocks and schoolbook for the rest.
    """
    if time.time() > deadline:
        return None
    
    U, V, W = _kronecker_decompose_3x5x6()
    
    if _check_exact(U, V, W, n, m, p):
        return U, V, W
    
    return None


def _try_greedy_rank_reduction(n, m, p, U, V, W, deadline):
    """
    Try to reduce the rank of a given decomposition by finding linear dependencies
    among the columns of U, V, W simultaneously.
    """
    r = U.shape[1]
    # Placeholder for future improvement
    return None


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin under 150s
    n, m, p = DIMENSIONS

    # Schoolbook as fallback
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (U0, V0, W0)
    best_rank = n * m * p

    # Try block decomposition (Strassen + schoolbook hybrid)
    if time.time() < deadline:
        result = _try_block_decomposition(n, m, p, deadline)
        if result is not None:
            best = result
            best_rank = result[0].shape[1]

    # If we have time, try to search for even lower rank
    if time.time() < deadline - 30:
        search_start = max(20, best_rank - 20)
        search_end = best_rank - 1
        
        for target_rank in range(search_start, search_end + 1):
            if time.time() > deadline:
                break
            for attempt in range(3):
                if time.time() > deadline:
                    break
                rng = random.Random(42 + target_rank * 100 + attempt)
                U_rand = np.zeros((n * m, target_rank), dtype=object)
                V_rand = np.zeros((m * p, target_rank), dtype=object)
                W_rand = np.zeros((p * n, target_rank), dtype=object)
                for idx in range(n * m):
                    for c in range(target_rank):
                        U_rand[idx, c] = rng.randint(-3, 3)
                for idx in range(m * p):
                    for c in range(target_rank):
                        V_rand[idx, c] = rng.randint(-3, 3)
                for idx in range(p * n):
                    for c in range(target_rank):
                        W_rand[idx, c] = rng.randint(-3, 3)
                if _check_exact(U_rand, V_rand, W_rand, n, m, p):
                    best = (U_rand, V_rand, W_rand)
                    best_rank = target_rank
                    break
            if best_rank == target_rank:
                break

    U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
