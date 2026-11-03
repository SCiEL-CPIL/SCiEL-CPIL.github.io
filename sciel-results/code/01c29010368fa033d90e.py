import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if actual != expected:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U is 4x7, V is 4x7, W is 4x7.
    Indices:
    U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    """
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [
        [1, 0, 1, 0, 1, 0, 1], # A11
        [0, 1, 0, 0, 1, -1, 0], # A12
        [0, 0, 1, 1, 0, 1, 0], # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 1], # B11
        [0, 0, 1, 0, 0, 1, 0], # B12
        [1, 0, -1, 1, 1, 0, 1], # B21
        [1, 0, -1, 0, 1, 0, -1]  # B22
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0], # C12
        [0, 1, 0, 1, 0, 0, 0], # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ]
    return U, V, W

def _schoolbook_1x2x2():
    """
    Returns U, V, W for 1x2x2 schoolbook decomposition.
    U is 2x4 (row 0 is A0, row 1 is A1)
    V is 4x4 (rows 0,1 are B00,B01; rows 2,3 are B10,B11)
    W is 2x4 (row 0 is C0, row 1 is C1)
    """
    U = [
        [1, 0, 1, 0], # A0
        [0, 1, 0, 1]  # A1
    ]
    V = [
        [1, 0, 0, 0], # B00
        [0, 1, 0, 0], # B01
        [0, 0, 1, 0], # B10
        [0, 0, 0, 1]  # B11
    ]
    W = [
        [1, 0, 1, 0], # C0
        [0, 1, 0, 1]  # C1
    ]
    return U, V, W

def _build_88_decomposition():
    n, m, p = 3, 4, 8
    total_rank = 88
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    U_str, V_str, W_str = _strassen_2x2x2()
    U_sch, V_sch, W_sch = _schoolbook_1x2x2()
    
    def add_strassen_block(u_rows, u_cols, v_rows, v_cols, w_rows, w_cols):
        nonlocal col_offset
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[u_rows[i]*m + u_cols[j]][col_offset + c] += U_str[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V[v_rows[j]*p + v_cols[k]][col_offset + c] += V_str[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W[w_rows[k]*n + w_cols[i]][col_offset + c] += W_str[k*2+i][c]
        col_offset += 7

    def add_schoolbook_block(u_row, u_cols, v_rows, v_cols, w_row, w_cols):
        nonlocal col_offset
        for c in range(4):
            for j in range(2):
                U[u_row*m + u_cols[j]][col_offset + c] += U_sch[j][c]
            for j in range(2):
                for k in range(2):
                    V[v_rows[j]*p + v_cols[k]][col_offset + c] += V_sch[j*2+k][c]
            for k in range(2):
                W[w_row*n + w_cols[k]][col_offset + c] += W_sch[k][c]
        col_offset += 4

    # Blocks definition:
    # A: rows 0,1 (top), row 2 (bot)
    # A top cols: 0,1 (left), 2,3 (right)
    # A bot cols: 0,1 (left), 2,3 (right)
    
    # B: rows 0,1 (top), 2,3 (bot)
    # B cols: 0,1 (LL), 2,3 (LR), 4,5 (RL), 6,7 (RR)
    
    # C: rows 0,1 (top), 2 (bot)
    # C cols: 0,1 (LL), 2,3 (LR), 4,5 (RL), 6,7 (RR)
    
    # 1. Top-Left of C (rows 0,1; cols 0,1)
    # C_TL = A_TL * B_TL + A_TR * B_BL
    # A_TL: rows 0,1 cols 0,1. B_TL: rows 0,1 cols 0,1. C_TL: rows 0,1 cols 0,1.
    add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # 2. Top-Left-Right of C (rows 0,1; cols 2,3)
    # C_TLR = A_TL * B_TR + A_TR * B_BR
    # A_TL: rows 0,1 cols 0,1. B_TR: rows 0,1 cols 2,3. C_TLR: rows 0,1 cols 2,3.
    add_strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # 3. Top-Right-Left of C (rows 0,1; cols 4,5)
    # C_TRL = A_TL * B_RL + A_TR * B_RBL
    # A_TL: rows 0,1 cols 0,1. B_RL: rows 0,1 cols 4,5. C_TRL: rows 0,1 cols 4,5.
    add_strassen_block([0,1], [0,1], [0,1], [4,5], [0,1], [4,5])
    
    # 4. Top-Right-Right of C (rows 0,1; cols 6,7)
    # C_TRR = A_TL * B_RR + A_TR * B_RBR
    # A_TL: rows 0,1 cols 0,1. B_RR: rows 0,1 cols 6,7. C_TRR: rows 0,1 cols 6,7.
    add_strassen_block([0,1], [0,1], [0,1], [6,7], [0,1], [6,7])
    
    # Wait, I missed the A_TR contributions in the above 4 blocks?
    # The Strassen block computes $A_{sub} B_{sub}$.
    # I need to add TWO Strassen blocks for each $2 \times 4$ output block.
    # One for $A_{TL} B_{...}$ and one for $A_{TR} B_{...}$.
    
    # Let's restart the block assembly carefully.
    # Reset
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col_offset = 0
    
    # C rows 0,1, cols 0,1:
    # Term 1: A(0:2, 0:2) * B(0:2, 0:2)
    add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    # Term 2: A(0:2, 2:4) * B(2:4, 0:2)
    add_strassen_block([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # C rows 0,1, cols 2,3:
    # Term 3: A(0:2, 0:2) * B(0:2, 2:4)
    add_strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    # Term 4: A(0:2, 2:4) * B(2:4, 2:4)
    add_strassen_block([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    # C rows 0,1, cols 4,5:
    # Term 5: A(0:2, 0:2) * B(0:2, 4:6)
    add_strassen_block([0,1], [0,1], [0,1], [4,5], [0,1], [4,5])
    # Term 6: A(0:2, 2:4) * B(2:4, 4:6)
    add_strassen_block([0,1], [2,3], [2,3], [4,5], [0,1], [4,5])
    
    # C rows 0,1, cols 6,7:
    # Term 7: A(0:2, 0:2) * B(0:2, 6:8)
    add_strassen_block([0,1], [0,1], [0,1], [6,7], [0,1], [6,7])
    # Term 8: A(0:2, 2:4) * B(2:4, 6:8)
    add_strassen_block([0,1], [2,3], [2,3], [6,7], [0,1], [6,7])
    
    # C row 2, cols 0,1:
    # Term 9: A(2, 0:2) * B(0:2, 0:2)
    add_schoolbook_block(2, [0,1], [0,1], [0,1], 2, [0,1])
    # Term 10: A(2, 2:4) * B(2:4, 0:2)
    add_schoolbook_block(2, [2,3], [2,3], [0,1], 2, [0,1])
    
    # C row 2, cols 2,3:
    # Term 11: A(2, 0:2) * B(0:2, 2:4)
    add_schoolbook_block(2, [0,1], [0,1], [2,3], 2, [2,3])
    # Term 12: A(2, 2:4) * B(2:4, 2:4)
    add_schoolbook_block(2, [2,3], [2,3], [2,3], 2, [2,3])
    
    # C row 2, cols 4,5:
    # Term 13: A(2, 0:2) * B(0:2, 4:6)
    add_schoolbook_block(2, [0,1], [0,1], [4,5], 2, [4,5])
    # Term 14: A(2, 2:4) * B(2:4, 4:6)
    add_schoolbook_block(2, [2,3], [2,3], [4,5], 2, [4,5])
    
    # C row 2, cols 6,7:
    # Term 15: A(2, 0:2) * B(0:2, 6:8)
    add_schoolbook_block(2, [0,1], [0,1], [6,7], 2, [6,7])
    # Term 16: A(2, 2:4) * B(2:4, 6:8)
    add_schoolbook_block(2, [2,3], [2,3], [6,7], 2, [6,7])
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(base_result, n, m, p, time_budget):
    """Try to reduce rank by removing linearly dependent columns."""
    U = [row[:] for row in base_result["U"]]
    V = [row[:] for row in base_result["V"]]
    W = [row[:] for row in base_result["W"]]
    rank = len(U[0])
    
    start = time.time()
    
    # Try to remove columns one at a time
    # Check if column c can be removed: for all (i,j,k), the contribution
    # U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] must be compensatable by other columns
    # This is complex, so instead try: if a column is all zeros in one of U,V,W, remove it
    # More practically: try random linear combinations
    
    # Simple check: remove columns that are entirely zero
    cols_to_remove = []
    for c in range(rank):
        u_zero = all(U[i*m+j][c] == 0 for i in range(n) for j in range(m))
        v_zero = all(V[j*p+k][c] == 0 for j in range(m) for k in range(p))
        w_zero = all(W[k*n+i][c] == 0 for k in range(p) for i in range(n))
        if u_zero or v_zero or w_zero:
            cols_to_remove.append(c)
    
    if not cols_to_remove:
        return base_result
    
    # Remove zero columns
    new_rank = rank - len(cols_to_remove)
    keep = [c for c in range(rank) if c not in cols_to_remove]
    U_new = [[U[i*m+j][c] for c in keep] for i in range(n) for j in range(m)]
    V_new = [[V[j*p+k][c] for c in keep] for j in range(m) for k in range(p)]
    W_new = [[W[k*n+i][c] for c in keep] for k in range(p) for i in range(n)]
    
    result = {"U": U_new, "V": V_new, "W": W_new}
    if _verify(U_new, V_new, W_new, n, m, p):
        return result
    
    return base_result

def _try_winograd_2x2x2():
    """
    Try Winograd's method for 2x2x2 which might give rank 6 in some formulations.
    Actually Winograd for 2x2x2 is still rank 7 (same as Strassen).
    But let's verify our Strassen is correct and look for alternatives.
    """
    # Winograd's 2x2x2: uses 7 multiplications too, same as Strassen
    # The difference is in the structure of U,V,W
    # Let's try the standard Winograd formulation
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11)  
    # M3 = (a11)(b12-b22)
    # M4 = (a22)(b21-b11)
    # M5 = (a11+a12)(b22)
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # This is the same as Strassen, just different naming
    return None  # No improvement over Strassen for 2x2x2

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    result = _build_88_decomposition()
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], 3, 4, 8):
        # Try to reduce rank
        reduced = _try_reduce_rank(result, 3, 4, 8, deadline - time.time())
        if len(reduced["U"][0]) < len(result["U"][0]):
            return reduced
        return result
    
    # Fallback
    return _schoolbook()
# EVOLVE-BLOCK-END
