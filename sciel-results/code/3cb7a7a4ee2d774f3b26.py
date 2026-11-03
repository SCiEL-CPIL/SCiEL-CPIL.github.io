import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We will build the decomposition with rank 54.
    # Top 2 rows of C: 3 blocks of 2x2 output.
    # Each block is sum of two 2x2 multiplications (left and right halves of inner dim).
    # Each 2x2 multiplication uses Strassen (rank 7).
    # Total rank for top: 3 blocks * 2 mults * 7 = 42.
    
    # Bottom 1 row of C: 3 blocks of 1x2 output.
    # Each block is sum of two 1x2 by 2x2 multiplications.
    # Each 1x2 by 2x2 multiplication uses schoolbook (rank 4).
    # Total rank for bottom: 3 blocks * 2 mults * 4 = 24.
    
    # Total rank = 42 + 24 = 66? 
    # Wait, earlier I calculated 12 for bottom. Let's re-verify.
    # Bottom block: C[2, 0:2] = A[2, 0:2] @ B[0:2, 0:2] + A[2, 2:4] @ B[2:4, 0:2]
    # First term: 1x2 @ 2x2. Rank 4.
    # Second term: 1x2 @ 2x2. Rank 4.
    # Total per block: 8.
    # 3 blocks: 24.
    # Total Rank = 42 + 24 = 66.
    
    # Can we do better on 1x2 @ 2x2?
    # C_0 = A_0 B_00 + A_1 B_10
    # C_1 = A_0 B_01 + A_1 B_11
    # Rank 2?
    # M1 = A_0 B_00 + A_1 B_10? No.
    # If we use rank 2:
    # M1 = A_0 (B_00 + B_11)
    # M2 = A_1 (B_10 + B_01)
    # C_0 = M1 + M2 - A_0 B_11 - A_1 B_01? No.
    # It is known that rank of 1x2 x 2x2 is 4.
    
    # So Rank 66 is the target.
    
    U = np.zeros((n * m, 66), dtype=float)
    V = np.zeros((m * p, 66), dtype=float)
    W = np.zeros((p * n, 66), dtype=float)
    
    col = 0
    
    # Helper to add Strassen 2x2 block
    def add_strassen_2x2(a_i0, a_i1, a_j0, a_j1, b_j0, b_j1, b_k0, b_k1, c_i0, c_i1, c_k0, c_k1):
        nonlocal col
        # A indices
        a00 = a_i0 * m + a_j0
        a01 = a_i0 * m + a_j1
        a10 = a_i1 * m + a_j0
        a11 = a_i1 * m + a_j1
        
        # B indices
        b00 = b_j0 * p + b_k0
        b01 = b_j0 * p + b_k1
        b10 = b_j1 * p + b_k0
        b11 = b_j1 * p + b_k1
        
        # C indices (W indices: k*n + i)
        c00 = c_k0 * n + c_i0
        c01 = c_k1 * n + c_i0
        c10 = c_k0 * n + c_i1
        c11 = c_k1 * n + c_i1
        
        # M1 = (A00+A11)(B01+B11) -> C00(+1), C11(+1)
        U[a00, col] = 1; U[a11, col] = 1
        V[b01, col] = 1; V[b11, col] = 1
        W[c00, col] = 1; W[c11, col] = 1
        col += 1
        
        # M2 = (A10+A11)B00 -> C10(+1), C11(-1)
        U[a10, col] = 1; U[a11, col] = 1
        V[b00, col] = 1
        W[c10, col] = 1; W[c11, col] = -1
        col += 1
        
        # M3 = A00(B10-B11) -> C01(+1), C11(+1)
        U[a00, col] = 1
        V[b10, col] = 1; V[b11, col] = -1
        W[c01, col] = 1; W[c11, col] = 1
        col += 1
        
        # M4 = A11(B00-B10) -> C00(+1), C10(+1)
        U[a11, col] = 1
        V[b00, col] = 1; V[b10, col] = -1
        W[c00, col] = 1; W[c10, col] = 1
        col += 1
        
        # M5 = (A00+A10)B11 -> C00(-1), C01(+1)
        U[a00, col] = 1; U[a10, col] = 1
        V[b11, col] = 1
        W[c00, col] = -1; W[c01, col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11(+1)
        U[a10, col] = 1; U[a00, col] = -1
        V[b00, col] = 1; V[b01, col] = 1
        W[c11, col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00(+1)
        U[a01, col] = 1; U[a11, col] = -1
        V[b10, col] = 1; V[b11, col] = 1
        W[c00, col] = 1
        col += 1
        
    # Helper to add schoolbook 1x2 @ 2x2 block
    def add_schoolbook_1x2_2x2(a_i, a_j0, a_j1, b_j0, b_j1, b_k0, b_k1, c_i, c_k0, c_k1):
        nonlocal col
        a0 = a_i * m + a_j0
        a1 = a_i * m + a_j1
        
        b00 = b_j0 * p + b_k0
        b01 = b_j0 * p + b_k1
        b10 = b_j1 * p + b_k0
        b11 = b_j1 * p + b_k1
        
        c0 = c_k0 * n + c_i
        c1 = c_k1 * n + c_i
        
        # Term 1: A0 B00 -> C0
        U[a0, col] = 1
        V[b00, col] = 1
        W[c0, col] = 1
        col += 1
        
        # Term 2: A1 B10 -> C0
        U[a1, col] = 1
        V[b10, col] = 1
        W[c0, col] = 1
        col += 1
        
        # Term 3: A0 B01 -> C1
        U[a0, col] = 1
        V[b01, col] = 1
        W[c1, col] = 1
        col += 1
        
        # Term 4: A1 B11 -> C1
        U[a1, col] = 1
        V[b11, col] = 1
        W[c1, col] = 1
        col += 1

    # Top rows of C (i=0,1)
    # Inner split: j=0,1 and j=2,3
    # B cols split: k=0,1; k=2,3; k=4,5
    
    for k_start in [0, 2, 4]:
        # Left part: A[:, 0:2] @ B[0:2, k_start:k_start+2]
        # A rows 0,1; A cols 0,1
        # B rows 0,1; B cols k_start, k_start+1
        # C rows 0,1; C cols k_start, k_start+1
        add_strassen_2x2(
            a_i0=0, a_i1=1, a_j0=0, a_j1=1,
            b_j0=0, b_j1=1, b_k0=k_start, b_k1=k_start+1,
            c_i0=0, c_i1=1, c_k0=k_start, c_k1=k_start+1
        )
        
        # Right part: A[:, 2:4] @ B[2:4, k_start:k_start+2]
        # A rows 0,1; A cols 2,3
        # B rows 2,3; B cols k_start, k_start+1
        # C rows 0,1; C cols k_start, k_start+1
        add_strassen_2x2(
            a_i0=0, a_i1=1, a_j0=2, a_j1=3,
            b_j0=2, b_j1=3, b_k0=k_start, b_k1=k_start+1,
            c_i0=0, c_i1=1, c_k0=k_start, c_k1=k_start+1
        )

    # Bottom row of C (i=2)
    for k_start in [0, 2, 4]:
        # Left part: A[2, 0:2] @ B[0:2, k_start:k_start+2]
        add_schoolbook_1x2_2x2(
            a_i=2, a_j0=0, a_j1=1,
            b_j0=0, b_j1=1, b_k0=k_start, b_k1=k_start+1,
            c_i=2, c_k0=k_start, c_k1=k_start+1
        )
        
        # Right part: A[2, 2:4] @ B[2:4, k_start:k_start+2]
        add_schoolbook_1x2_2x2(
            a_i=2, a_j0=2, a_j1=3,
            b_j0=2, b_j1=3, b_k0=k_start, b_k1=k_start+1,
            c_i=2, c_k0=k_start, c_k1=k_start+1
        )

    return {"U": U, "V": V, "W": W}
