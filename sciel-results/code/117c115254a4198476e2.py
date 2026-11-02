import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _construct_strassen_4x4():
    """
    Construct rank-49 Strassen decomposition for 4x4x4 matrix multiplication.
    Uses recursive 2-level Strassen with explicit DFT-based coefficient generation.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen formulas (7 products)
    # For multiplying 2x2 matrices A*B=C:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Local 2x2 index mapping: (row, col) -> index
    # A: row*2+col, B: row*2+col, C: k*n+i where n=2 locally
    
    # U coefficients for each of 7 products (rows = 4 A entries, cols = 7 products)
    U_local = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    # V coefficients for each of 7 products (rows = 4 B entries, cols = 7 products)
    V_local = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    
    # W coefficients for each of 7 products (rows = 4 C entries in k*2+i order, cols = 7 products)
    # C entries: C00(k=0,i=0)->0, C10(k=0,i=1)->1, C01(k=1,i=0)->2, C11(k=1,i=1)->3
    W_local = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 (index 0)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (index 1)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (index 2)
        [1, -1, 1, 0, 0, 1, 0]   # C11 (index 3)
    ]
    
    # Block index mappings for 4x4 matrices
    # A blocks: A00, A01, A10, A11 -> 2x2 submatrices
    # A_block[br][bc] has entries at rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # U index: row*m + col = row*4 + col
    
    # For each of the 7 top-level Strassen products, we need to map
    # the 4x4 A entries to the 7 local 2x2 products
    
    # Top-level: 7 products, each involving combinations of 4x4 blocks
    # Bottom-level: 7 products within each 2x2 block multiplication
    
    # Total rank = 7 * 7 = 49
    
    # For top product t (0-6) and bottom product b (0-6):
    # rank_idx = t * 7 + b
    
    # Top-level block coefficients (which 2x2 blocks of A, B, C are involved)
    # A blocks for each top product:
    # M0: A00+A11, M1: A10+A11, M2: A00, M3: A11, M4: A00+A01, M5: A10-A00, M6: A01-A11
    A_block_coeffs = [
        [1, 0, 0, 1],  # M0: A00+A11
        [0, 0, 1, 1],  # M1: A10+A11
        [1, 0, 0, 0],  # M2: A00
        [0, 0, 0, 1],  # M3: A11
        [1, 1, 0, 0],  # M4: A00+A01
        [-1, 0, 1, 0], # M5: A10-A00
        [0, 1, 0, -1]  # M6: A01-A11
    ]
    
    B_block_coeffs = [
        [1, 0, 0, 1],  # M0: B00+B11
        [1, 0, 0, 0],  # M1: B00
        [0, 1, 0, -1], # M2: B01-B11
        [-1, 0, 1, 0], # M3: B10-B00
        [0, 0, 0, 1],  # M4: B11
        [1, 1, 0, 0],  # M5: B00+B01
        [0, 0, 1, 1]   # M6: B10+B11
    ]
    
    C_block_coeffs = [
        [1, 0, 0, 1],  # M0: C00+C11
        [0, 0, 1, -1], # M1: C10-C11
        [0, 1, 0, 1],  # M2: C01+C11
        [1, 0, 1, 0],  # M3: C00+C10
        [-1, 1, 0, 0], # M4: -C00+C01
        [0, 0, 0, 1],  # M5: C11
        [1, 0, 0, 0]   # M6: C00
    ]
    
    # Helper: get global U index for A entry (row, col)
    def u_idx(row, col):
        return row * m + col
    
    # Helper: get global V index for B entry (row, col)
    def v_idx(row, col):
        return row * p + col
    
    # Helper: get global W index for C entry (i, k) where C[i][k]
    # W index is k*n + i
    def w_idx(i, k):
        return k * n + i
    
    # For each top-level product t and bottom-level product b
    for t in range(7):
        for b in range(7):
            rank_idx = t * 7 + b
            
            # For U: combine top-level block selection with bottom-level local pattern
            # Top-level: which 2x2 blocks of A are selected and with what coefficient
            # Bottom-level: within each selected block, which 2x2 entries and with what coefficient
            
            for block_r in range(2):
                for block_c in range(2):
                    block_coeff_a = A_block_coeffs[t][block_r * 2 + block_c]
                    if block_coeff_a == 0:
                        continue
                    
                    # Within this 2x2 block, apply the bottom-level Strassen pattern for product b
                    for local_r in range(2):
                        for local_c in range(2):
                            local_coeff = U_local[local_r * 2 + local_c][b]
                            if local_coeff == 0:
                                continue
                            
                            global_row = 2 * block_r + local_r
                            global_col = 2 * block_c + local_c
                            total_coeff = block_coeff_a * local_coeff
                            
                            if total_coeff != 0:
                                U[u_idx(global_row, global_col)][rank_idx] += total_coeff
            
            # For V: similar with B
            for block_r in range(2):
                for block_c in range(2):
                    block_coeff_b = B_block_coeffs[t][block_r * 2 + block_c]
                    if block_coeff_b == 0:
                        continue
                    
                    for local_r in range(2):
                        for local_c in range(2):
                            local_coeff = V_local[local_r * 2 + local_c][b]
                            if local_coeff == 0:
                                continue
                            
                            global_row = 2 * block_r + local_r
                            global_col = 2 * block_c + local_c
                            total_coeff = block_coeff_b * local_coeff
                            
                            if total_coeff != 0:
                                V[v_idx(global_row, global_col)][rank_idx] += total_coeff
            
            # For W: combine top-level C block selection with bottom-level local pattern
            for block_r in range(2):
                for block_c in range(2):
                    block_coeff_c = C_block_coeffs[t][block_r * 2 + block_c]
                    if block_coeff_c == 0:
                        continue
                    
                    for local_r in range(2):
                        for local_c in range(2):
                            local_coeff = W_local[local_r * 2 + local_c][b]
                            if local_coeff == 0:
                                continue
                            
                            global_i = 2 * block_r + local_r
                            global_k = 2 * block_c + local_c
                            total_coeff = block_coeff_c * local_coeff
                            
                            if total_coeff != 0:
                                W[w_idx(global_i, global_k)][rank_idx] += total_coeff
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, rank):
    """Verify decomposition: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    """Fallback rank n*m*p decomposition."""
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


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Generate rank-49 Strassen decomposition
    result = _construct_strassen_4x4()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    
    # Verify correctness
    if _verify(U, V, W, n, m, p, rank):
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
