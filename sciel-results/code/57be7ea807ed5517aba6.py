import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition of 5x5x5 matrix multiplication.
    Uses Strassen's algorithm for the 4x4 sub-block (rank 49) 
    and schoolbook for the remaining parts (rank 37).
    Total rank: 86.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # We expect rank 86
    rank = 86
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get Strassen decomposition for N x N (N power of 2)
    def get_strassen_decomp(N):
        if N == 1:
            return [[1]], [[1]], [[1]]
        
        H = N // 2
        U_h, V_h, W_h = get_strassen_decomp(H)
        R_h = len(U_h[0])
        
        # Define the 7 Strassen multiplications
        # Each entry is (u_blocks, v_blocks, w_blocks)
        # u_blocks: list of (block_idx, coeff)
        # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
        
        # M1: (A00+A11)(B00+B11) -> C00(+), C11(+)
        m1 = ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
        # M2: (A10+A11)B00 -> C10(+), C11(-)
        m2 = ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])
        # M3: A00(B01-B11) -> C01(+), C11(+)
        m3 = ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])
        # M4: A11(B10-B00) -> C00(+), C10(+)
        m4 = ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])
        # M5: (A00+A01)B11 -> C00(-), C01(+)
        m5 = ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)])
        # M6: (A10-A00)(B00+B01) -> C11(+)
        m6 = ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
        # M7: (A01-A11)(B10+B11) -> C00(+), C11(+)
        m7 = ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)])
        
        mults = [m1, m2, m3, m4, m5, m6, m7]
        
        U_full = []
        V_full = []
        W_full = []
        
        for u_b, v_b, w_b in mults:
            for r in range(R_h):
                u_vec = [0] * (N * N)
                v_vec = [0] * (N * N)
                w_vec = [0] * (N * N)
                
                # Fill U
                for block_idx, coeff in u_b:
                    # Map sub-index to full-index
                    # Block 0 (00): rows 0-H, cols 0-H
                    # Block 1 (01): rows 0-H, cols H-N
                    # Block 2 (10): rows H-N, cols 0-H
                    # Block 3 (11): rows H-N, cols H-N
                    base_r = 0 if block_idx < 2 else H
                    base_c = 0 if block_idx % 2 == 0 else H
                    for i in range(H):
                        for j in range(H):
                            sub_idx = i * H + j
                            full_idx = (base_r + i) * N + (base_c + j)
                            u_vec[full_idx] += coeff * U_h[r][sub_idx]
                
                # Fill V
                for block_idx, coeff in v_b:
                    base_r = 0 if block_idx < 2 else H
                    base_c = 0 if block_idx % 2 == 0 else H
                    for i in range(H):
                        for j in range(H):
                            sub_idx = i * H + j
                            full_idx = (base_r + i) * N + (base_c + j)
                            v_vec[full_idx] += coeff * V_h[r][sub_idx]
                            
                # Fill W
                for block_idx, coeff in w_b:
                    base_r = 0 if block_idx < 2 else H
                    base_c = 0 if block_idx % 2 == 0 else H
                    for i in range(H):
                        for j in range(H):
                            sub_idx = i * H + j
                            # W index convention: k*n + i (col-major)
                            # Sub W is for HxH.
                            # Full W is for NxN.
                            # Sub (i,j) corresponds to C[i,j].
                            # Full C[i_full, j_full].
                            i_full = base_r + i
                            j_full = base_c + j
                            # Task W index: k*n + i => j_full * N + i_full
                            full_idx = j_full * N + i_full
                            w_vec[full_idx] += coeff * W_h[r][sub_idx]
                            
                U_full.append(u_vec)
                V_full.append(v_vec)
                W_full.append(w_vec)
                
        return U_full, V_full, W_full

    # Get 4x4 Strassen decomposition
    U_4, V_4, W_4 = get_strassen_decomp(4)
    
    col = 0
    # Add Strassen 4x4 terms
    for r in range(len(U_4[0])):
        # Map 4x4 indices to 5x5 indices
        # A (4x4) -> A (5x5) top-left
        for i in range(4):
            for j in range(4):
                U[i * 5 + j][col] = U_4[r][i * 4 + j]
                V[j * 5 + i][col] = V_4[r][j * 4 + i] # Note: V is B. B[j,i]
                # Wait, V_4 is for B (4x4).
                # In 5x5, B is at top-left.
                # V_4 index: j*4+i.
                # V_5 index: j*5+i.
                pass # Already done above? 
                # Let's re-verify V mapping.
                # V_4[r][j*4+i] is coefficient for B_{j,i}.
                # In 5x5, B_{j,i} is at V[j*5+i].
                # So V[j*5+i][col] = V_4[r][j*4+i].
                
                # W mapping
                # W_4[r][k*4+i] is coefficient for C_{i,k}.
                # In 5x5, C_{i,k} is at W[k*5+i].
                pass

        # Explicit loop for clarity and correctness
        for i in range(4):
            for j in range(4):
                U[i * 5 + j][col] = U_4[r][i * 4 + j]
                V[j * 5 + i][col] = V_4[r][j * 4 + i]
                # W index in 4x4 decomp: k*4 + i (col k, row i)
                # W index in 5x5: k*5 + i
                # Iterate over W_4 non-zeros? Or just map all.
                for k in range(4):
                    w_val = W_4[r][k * 4 + i]
                    if w_val != 0:
                        W[k * 5 + i][col] = w_val
        col += 1

    # Add Schoolbook terms for C[i,4] (i<4)
    # C[i,4] = sum_{j=0}^3 A[i,j] B[j,4]
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 5 + i][col] = 1
            col += 1

    # Add Schoolbook terms for C[4,k] (k<4)
    # C[4,k] = sum_{j=0}^3 A[4,j] B[j,k]
    for k in range(4):
        for j in range(4):
            U[4 * 5 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 5 + 4][col] = 1
            col += 1

    # Add terms for C[4,4]
    # C[4,4] = sum_{j=0}^3 A[4,j] B[j,4] + A[4,4] B[4,4]
    for j in range(4):
        U[4 * 5 + j][col] = 1
        V[j * 5 + 4][col] = 1
        W[4 * 5 + 4][col] = 1
        col += 1
        
    # Scalar term
    U[4 * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    W[4 * 5 + 4][col] = 1
    col += 1
    
    # Verify col == rank
    # 49 + 16 + 16 + 4 + 1 = 86
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
