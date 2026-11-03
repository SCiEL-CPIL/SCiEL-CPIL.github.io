import numpy as np
import time

n, m, p = 4, 4, 5
TIME_LIMIT = 120


def _strassen_2x2x2_factors():
    """
    Return the 7 Strassen factors for 2x2x2 matrix multiplication.
    Each factor is (A_dict, B_dict, C_dict) where keys are (i,j) in {0,1}x{0,1}.
    """
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    factors = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(0,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    return factors


def _build_strassen_4x4x4(U, V, W, base_r=0):
    """
    Build Strassen recursion factors for 4x4x4 into U, V, W.
    Returns the next available rank index.
    
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    For the 4x4x4 part, we're computing C[0:4, 0:4] = A[0:4,0:4] * B[0:4,0:4].
    
    U index: i*m + j
    V index: j*p + k  (but for the 4x4 part, k is 0..3)
    W index: k*n + i
    """
    # Split A into 2x2 blocks: A[br*2:(br+1)*2, bc*2:(bc+1)*2]
    # Split B into 2x2 blocks: B[br*2:(br+1)*2, bc*2:(bc+1)*2]
    # Split C into 2x2 blocks: C[br*2:(br+1)*2, bc*2:(bc+1)*2]
    
    strassen_2x2 = _strassen_2x2x2_factors()
    
    r = base_r
    for br_A in range(2):  # block row of A
        for bc_A in range(2):  # block col of A
            for br_B in range(2):  # block row of B (= block col of A)
                for bc_B in range(2):  # block col of B
                    # We need to compute A_block * B_block where
                    # A_block = A[br_A*2:(br_A+1)*2, bc_A*2:(bc_A+1)*2]
                    # B_block = B[br_B*2:(br_B+1)*2, bc_B*2:(bc_B+1)*2]
                    # But this only contributes to C where the block positions match
                    # C[br_A*2:(br_A+1)*2, bc_B*2:(bc_B+1)*2] += A_block * B_block
                    # Wait, no. Standard matmul: C[i,k] = sum_j A[i,j]*B[j,k]
                    # Block structure: C_br,bc = sum_bc' A_br,bc' * B_bc',bc
                    # So C_block(br_A, bc_B) = sum_bc A_block(br_A, bc) * B_block(bc, bc_B)
                    
                    # For Strassen recursion on 4x4x4:
                    # We have 4x4 matrices. Split into 2x2 blocks.
                    # C[br, bc] = sum_b A[br, b] * B[b, bc] for b in {0,1}
                    # 
                    # Apply Strassen to the "block matrix" multiplication:
                    # Let A_blocks = [[A00, A01], [A10, A11]] (each 2x2)
                    # Let B_blocks = [[B00, B01], [B10, B11]] (each 2x2)
                    # C_blocks = A_blocks * B_blocks (block multiplication)
                    #
                    # Strassen on blocks:
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)B00
                    # M3 = A00(B10-B11)
                    # M4 = A11(B01-B10)
                    # M5 = (A00+A01)B11
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    #
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    #
                    # Each Mk is a 2x2 matrix product, decomposed by Strassen 2x2x2.
                    
                    pass
    
    # Actually, let me restructure. The Strassen recursion works as follows:
    # The 4x4x4 tensor is decomposed by applying Strassen at the block level.
    # There are 7 block-level products, each of which is a 2x2x2 tensor.
    # Each 2x2x2 tensor has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # For each of the 7 Strassen block products, we need to:
    # 1. Determine the linear forms on A (4x4) and B (4x4)
    # 2. For each of the 7 sub-products within that block product, determine the
    #    specific elements of A and B involved
    
    # Let's do it properly.
    # Block (br, bc) of A covers elements A[2*br+di, 2*bc+dj] for di,dj in {0,1}
    # Similarly for B and C.
    
    r = base_r
    for (A_blocks, B_blocks, C_blocks) in strassen_2x2:
        # A_blocks: dict (br, bc) -> coeff, for the linear form on A at block level
        # B_blocks: dict (br, bc) -> coeff
        # C_blocks: dict (br, bc) -> coeff
        
        # For each of the 7 sub-factors within this block product:
        for (sub_A, sub_B, sub_C) in strassen_2x2:
            # sub_A: dict (di, dj) -> coeff, within a 2x2 block
            # sub_B: dict (di, dj) -> coeff
            # sub_C: dict (di, dj) -> coeff
            
            # The full linear form on A for this term:
            # For each block (br, bc) in A_blocks with coeff ca:
            #   For each element offset (di, dj) in sub_A with coeff cs:
            #     Element A[2*br+di, 2*bc+dj] gets coeff ca * cs
            
            U_terms = {}
            for (br, bc), ca in A_blocks.items():
                for (di, dj), cs in sub_A.items():
                    elem = (2*br + di, 2*bc + dj)
                    U_terms[elem] = U_terms.get(elem, 0) + ca * cs
            
            V_terms = {}
            for (br, bc), cb in B_blocks.items():
                for (di, dj), cs in sub_B.items():
                    elem = (2*br + di, 2*bc + dj)
                    V_terms[elem] = V_terms.get(elem, 0) + cb * cs
            
            W_terms = {}
            for (br, bc), cw in C_blocks.items():
                for (di, dj), cs in sub_C.items():
                    elem = (2*br + di, 2*bc + dj)
                    W_terms[elem] = W_terms.get(elem, 0) + cw * cs
            
            # Now fill in U, V, W
            for (i, j), coeff in U_terms.items():
                U[i * m + j, r] += coeff
            for (i, j), coeff in V_terms.items():
                # B element (i, j) where i is row of B (= col of A), j is col of B
                # V index: i*p + j
                V[i * p + j, r] += coeff
            for (i, j), coeff in W_terms.items():
                # C element (i, j) where i is row of C, j is col of C
                # W index: j*n + i
                W[j * n + i, r] += coeff
            
            r += 1
    
    return r


def _verify(U, V, W, n, m, p, num_tests=5, tol=1e-9):
    """Verify the decomposition."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_exp = A @ B
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    a_val = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                    b_val = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                    C_comp[i, k] += W[k*n+i, r] * a_val * b_val
        if not np.allclose(C_comp, C_exp, atol=tol):
            return False
    return True


def construct():
    start = time.time()
    
    # Rank 49 for 4x4x4 (Strassen recursion) + 16 for 4x4x1 (schoolbook) = 65
    rank_total = 65
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Strassen 4x4x4 part (columns 0-3 of B)
    next_r = _build_strassen_4x4x4(U, V, W, base_r=0)
    # next_r should be 49
    
    # Schoolbook for column 4 of B
    for i in range(n):
        for j in range(m):
            r = next_r + i * m + j
            U[i * m + j, r] = 1
            V[j * p + 4, r] = 1
            W[4 * n + i, r] = 1  # k=4
    
    # Verify
    if time.time() - start < TIME_LIMIT - 10:
        try:
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, num_tests=5):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
