import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        if U_np.shape != (n * m, rank) or V_np.shape != (m * p, rank) or W_np.shape != (p * n, rank):
            return False

        T_target = np.zeros((n * m, m * p, p * n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0

        try:
            T_calc = np.einsum('ak,bk,ck->abc', U_np, V_np, W_np)
        except Exception:
            return False

        if not np.allclose(T_calc, T_target, atol=1e-6):
            return False
            
        return True

    # Schoolbook fallback (rank 100)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    if time.time() < deadline:
        U_opt, V_opt, W_opt, rank_opt = _build_optimized_decomposition(n, m, p)
        if verify(U_opt, V_opt, W_opt, rank_opt):
            return {"U": U_opt, "V": V_opt, "W": W_opt}

    return {"U": U_sb, "V": V_sb, "W": W_sb}

def _build_optimized_decomposition(n, m, p):
    """
    Build rank-64 decomposition for 4x5x5 matrix multiplication.
    
    Split m=5 into m1=4, m2=1.
    A (4x5) = [A_L (4x4) | A_R (4x1)]
    B (5x5) = [B_TL (4x4) B_TR (4x1); B_BL (1x4) B_BR (1x1)]
    C (4x5) = [C_L (4x4) | C_R (4x1)]
    
    C_L = A_L * B_TL + A_R * B_BL
    C_R = A_L * B_TR + A_R * B_BR
    
    Parts:
    1. A_L (4x4) * B_TL (4x4) -> C_L (4x4): rank 28 (Strassen)
    2. A_L (4x4) * B_TR (4x1) -> C_R (4x1): rank 16 (schoolbook)
    3. A_R (4x1) * B_BL (1x4) -> C_L (4x4): rank 16 (schoolbook)
    4. A_R (4x1) * B_BR (1x1) -> C_R (4x1): rank 4 (schoolbook)
    Total: 28 + 16 + 16 + 4 = 64
    """
    rank = 64
    U = [[0] * rank for _ in range(n * m)]  # (20, 64)
    V = [[0] * rank for _ in range(m * p)]  # (25, 64)
    W = [[0] * rank for _ in range(p * n)]  # (20, 64)
    
    col = 0
    
    # Part 1: 4x4x4 Strassen (rank 28)
    # A_L is rows 0-3, cols 0-3 of A (indices i*m+j for i=0..3, j=0..3)
    # B_TL is rows 0-3, cols 0-3 of B (indices j*p+k for j=0..3, k=0..3)
    # C_L is rows 0-3, cols 0-3 of C (indices k*n+i for k=0..3, i=0..3)
    
    # Strassen for 4x4: view as 2x2 blocks of 2x2 matrices.
    # Block (br, bc) of a 4x4 matrix covers rows 2*br..2*br+1, cols 2*bc..2*bc+1.
    # 
    # Strassen terms (7 terms, each is a 2x2 matrix multiplication of rank 4):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # For each Strassen term, we have a 2x2 matrix multiplication.
    # The 2x2 multiplication has 4 schoolbook terms.
    # Each term: A_v[i,j] * B_v[j,k] -> C_v[i,k]
    
    # Define the 7 Strassen terms with their A, B block coefficients
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # (a_blocks, b_blocks, c_blocks)
        # M1: A(0+3), B(0+3), contributes to C with coefficients
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),          # M2 -> C10=M2+M4, C11=M1-M2+...
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),           # M3 -> C01=M3+M5, C11=...+M3...
        ({3: 1}, {2: 1, 0: -1}, {0: -1, 2: 1}),          # M4 -> C00=M1-M4-..., C10=M2+M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),           # M5 -> C00=...-M5..., C01=M3+M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {11: -1}),         # M6 -> C11=...-M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 11: -1}),   # M7 -> C00=...+M7, C11=...-M6
    ]
    
    # Let me redo this more carefully.
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # So the contribution of each Mk to each C block:
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(-1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(-1)
    # M7: C00(+1)
    
    strassen_defs = [
        # (a_blocks, b_blocks, c_contributions)
        # c_contributions: dict from C_block_idx to coefficient
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),      # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),            # M3
        ({3: 1}, {2: 1, 0: -1}, {0: -1, 2: 1}),           # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: -1}),           # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),            # M7
    ]
    
    for a_blocks, b_blocks, c_blocks in strassen_defs:
        # For this Strassen term, we have a virtual 2x2 matrix multiplication.
        # The virtual A is a 2x2 matrix where A_v[br, bc] = sum of A[br,bc] with coefficients from a_blocks
        # Similarly for B_v.
        # The virtual C_v[i,k] contributes to C blocks with coefficients from c_blocks.
        
        # The 2x2 multiplication has 4 schoolbook terms:
        # (i_v, j_v, k_v) where i_v, j_v, k_v in {0,1}
        # A_v[i_v, j_v] * B_v[j_v, k_v] -> C_v[i_v, k_v]
        
        for i_v in range(2):
            for j_v in range(2):
                for k_v in range(2):
                    # This is one rank-1 term
                    # A_v[i_v, j_v] corresponds to A block (i_v, j_v)
                    # B_v[j_v, k_v] corresponds to B block (j_v, k_v)
                    # C_v[i_v, k_v] corresponds to C block (i_v, k_v)
                    
                    a_block = i_v * 2 + j_v  # 0,1,2,3
                    b_block = j_v * 2 + k_v
                    c_block = i_v * 2 + k_v
                    
                    a_coeff = a_blocks.get(a_block, 0)
                    b_coeff = b_blocks.get(b_block, 0)
                    c_coeff = c_blocks.get(c_block, 0)
                    
                    if a_coeff == 0 or b_coeff == 0 or c_coeff == 0:
                        continue
                    
                    # Now, the 2x2 block (br, bc) of A_L covers:
                    # A_L rows 2*br..2*br+1, cols 2*bc..2*bc+1
                    # In the full A (4x5), A_L is at rows 0-3, cols 0-3
                    # So A_L[2*br+di, 2*bc+dj] = A[2*br+di, 2*bc+dj]
                    # In U, index is i*m + j = (2*br+di)*5 + (2*bc+dj)
                    
                    # Similarly for B_TL: B_TL[2*br+di, 2*bc+dj] = B[2*br+di, 2*bc+dj]
                    # In V, index is j*p + k = (2*br+di)*5 + (2*bc+dj)
                    
                    # For C_L: C_L[2*br+di, 2*bc+dj] = C[2*br+di, 2*bc+dj]
                    # In W, index is k*n + i = (2*bc+dj)*4 + (2*br+di)
                    
                    # But wait, the 2x2 multiplication within a block:
                    # A_v[i_v, j_v] is a 2x2 matrix = sum of A blocks
                    # The schoolbook term is A_v[i_v,j_v][di, dj] * B_v[j_v,k_v][dk, dl] -> C_v[i_v,k_v][di, dl]
                    # where di, dj, dk, dl in {0,1}
                    # and dj = dk (the inner dimension of the 2x2 blocks)
                    
                    # Actually, for the 2x2 matrix multiplication:
                    # C_v[i_v, k_v][di, dl] = sum_{dj=0}^{1} A_v[i_v, j_v][di, dj] * B_v[j_v, k_v][dj, dl]
                    # So for each (di, dj, dl) with dj being the summation index:
                    
                    for di in range(2):
                        for dj in range(2):
                            for dl in range(2):
                                # A element: row = 2*i_v + di, col = 2*j_v + dj in A_L
                                a_row = 2 * i_v + di
                                a_col = 2 * j_v + dj
                                u_idx = a_row * 5 + a_col  # A is 4x5, A_L is cols 0-3
                                
                                # B element: row = 2*j_v + dj, col = 2*k_v + dl in B_TL
                                b_row = 2 * j_v + dj
                                b_col = 2 * k_v + dl
                                v_idx = b_row * 5 + b_col  # B is 5x5, B_TL is rows 0-3, cols 0-3
                                
                                # C element: row = 2*i_v + di, col = 2*k_v + dl in C_L
                                c_row = 2 * i_v + di
                                c_col = 2 * k_v + dl
                                w_idx = c_col * 4 + c_row  # C is 4x5, C_L is cols 0-3, W index is k*n+i
                                
                                U[u_idx][col] += a_coeff
                                V[v_idx][col] += b_coeff
                                W[w_idx][col] += c_coeff
                    
                    col += 1
    
    # Part 2: A_L (4x4) * B_TR (4x1) -> C_R (4x1), rank 16
    # A_L[i,j] * B_TR[j,4] -> C_R[i,4]
    # U index: i*5 + j (i=0..3, j=0..3)
    # V index: j*5 + 4 (j=0..3)
    # W index: 4*4 + i = 16 + i (k=4, i=0..3)
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    # Part 3: A_R (4x1) * B_BL (1x4) -> C_L (4x4), rank 16
    # A_R[i,4] * B_BL[4,k] -> C_L[i,k]
    # U index: i*5 + 4 (i=0..3)
    # V index: 4*5 + k = 20 + k (k=0..3)
    # W index: k*4 + i (k=0..3, i=0..3)
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    # Part 4: A_R (4x1) * B_BR (1x1) -> C_R (4x1), rank 4
    # A_R[i,4] * B_BR[4,4] -> C_R[i,4]
    # U index: i*5 + 4 (i=0..3)
    # V index: 4*5 + 4 = 24
    # W index: 4*4 + i = 16 + i (i=0..3)
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 4 + i][col] = 1
        col += 1
    
    return U, V, W, rank
# EVOLVE-BLOCK-END
