import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We will build a decomposition of rank 65.
    # Part 1: 4x4x4 multiplication using Strassen (rank 49)
    # Part 2: 4x4x1 multiplication using schoolbook (rank 16)
    
    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # --- Part 1: 4x4x4 Strassen ---
    # Indices for A: 0..3 (i), 0..3 (j)
    # Indices for B: 0..3 (j), 0..3 (k) -> Note: B is 4x4 here, so k is 0..3
    # Indices for C: 0..3 (k), 0..3 (i) -> Output C is 4x4, so W index is k*n+i
    
    # Strassen splits A (4x4) into 2x2 blocks:
    # A11 (0:2, 0:2), A12 (0:2, 2:4)
    # A21 (2:4, 0:2), A22 (2:4, 2:4)
    # B (4x4) into 2x2 blocks:
    # B11 (0:2, 0:2), B12 (0:2, 2:4)
    # B21 (2:4, 0:2), B22 (2:4, 2:4)
    
    # Strassen intermediates M1..M7 are 2x2 matrices.
    # Each M is computed via 2x2x2 Strassen (rank 7).
    
    # Helper to add a 2x2x2 Strassen term to U, V, W
    # offset_i, offset_j for A block
    # offset_j, offset_k for B block
    # offset_k, offset_i for C block (W index)
    # signs for A and B parts in the Strassen formula
    
    def add_2x2_strassen(a_off_i, a_off_j, b_off_j, b_off_k, c_off_k, c_off_i, 
                         a_signs, b_signs, w_sign, col_idx):
        """
        Adds the 7 rank-1 terms for a 2x2x2 multiplication.
        a_signs: list of 7 tuples (row, col, sign) relative to the 2x2 block
        b_signs: list of 7 tuples (row, col, sign) relative to the 2x2 block
        w_sign: scalar sign for the output contribution
        """
        # The 7 Strassen products for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We define the coefficients for A and B for each of the 7 terms.
        # Term 1: A=(1,1,1,1), B=(1,1,1,1)
        # Term 2: A=(0,0,1,1), B=(1,0,0,0)
        # Term 3: A=(1,0,0,0), B=(0,1,-1,0)
        # Term 4: A=(0,0,0,1), B=(0,-1,1,0)
        # Term 5: A=(1,1,0,0), B=(0,0,0,1)
        # Term 6: A=(-1,0,1,0), B=(1,1,0,0)
        # Term 7: A=(0,-1,0,1), B=(0,0,1,1)
        
        # A vectors (4 components: 00, 01, 10, 11)
        A_coefs = [
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, -1, 0, 1]
        ]
        
        # B vectors (4 components: 00, 01, 10, 11)
        B_coefs = [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [0, 1, -1, 0],
            [0, -1, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        # C vectors (4 components: 00, 01, 10, 11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        C_coefs = [
            [1, 0, 0, 1],   # M1
            [0, 0, 1, -1],  # M2
            [0, 1, 0, 1],   # M3
            [1, 0, 1, 0],   # M4
            [-1, 1, 0, 0],  # M5
            [0, 0, 0, 1],   # M6
            [1, 0, 0, 0]    # M7
        ]
        
        for t in range(7):
            # Fill U
            for a_r in range(2):
                for a_c in range(2):
                    val = A_coefs[t][a_r * 2 + a_c]
                    if val != 0:
                        u_idx = (a_off_i + a_r) * m + (a_off_j + a_c)
                        U[u_idx, col_idx + t] += val * w_sign # w_sign is 1 for now, handled by C_coefs? 
                        # Actually, the scalar is product of A, B, C coeffs.
                        # Let's just accumulate.
                        
            # Fill V
            for b_r in range(2):
                for b_c in range(2):
                    val = B_coefs[t][b_r * 2 + b_c]
                    if val != 0:
                        v_idx = (b_off_j + b_r) * p + (b_off_k + b_c)
                        V[v_idx, col_idx + t] += val
                        
            # Fill W
            for c_r in range(2):
                for c_c in range(2):
                    val = C_coefs[t][c_r * 2 + c_c]
                    if val != 0:
                        w_idx = (c_off_k + c_r) * n + (c_off_i + c_c)
                        W[w_idx, col_idx + t] += val
                        
            # We need to normalize? No, the product U*V*W should sum to 1.
            # The coefficients are integers.
            
    # This helper is a bit complex to get right with signs.
    # Let's just hardcode the 49 terms explicitly for safety and speed.
    
    # Reset U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the 49 terms for 4x4x4 Strassen
    # We will iterate through the 4 block positions and the 7 Strassen steps.
    
    # Block offsets
    # A blocks: (0,0), (0,2), (2,0), (2,2)
    # B blocks: (0,0), (0,2), (2,0), (2,2)
    # C blocks: (0,0), (0,2), (2,0), (2,2)
    
    # Strassen formulas for 4x4x4:
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    # M2 = (A21+A22)B11 -> contributes to C21, C22
    # M3 = A11(B12-B22) -> contributes to C12, C22
    # M4 = A22(B21-B11) -> contributes to C11, C21
    # M5 = (A11+A12)B22 -> contributes to C11, C12
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    
    # Each M is a 2x2x2 multiplication, rank 7.
    
    # Let's define the 2x2x2 Strassen terms (rank 7) as a list of (U_vec, V_vec, W_vec)
    # where vectors are of length 4 (flattened 2x2).
    
    s2_U = [
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ]
    s2_V = [
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 1, -1, 0],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    s2_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Helper to add a block of 2x2x2 terms
    def add_block(a_off, b_off, c_off, scale_u, scale_v, scale_w, start_col):
        # a_off, b_off, c_off are (row, col) offsets for the 2x2 blocks
        for t in range(7):
            for r in range(2):
                for c in range(2):
                    # U
                    u_val = s2_U[t][r*2+c]
                    if u_val != 0:
                        u_idx = (a_off[0] + r) * m + (a_off[1] + c)
                        U[u_idx, start_col + t] += u_val * scale_u
                        
                    # V
                    v_val = s2_V[t][r*2+c]
                    if v_val != 0:
                        v_idx = (b_off[0] + r) * p + (b_off[1] + c)
                        V[v_idx, start_col + t] += v_val * scale_v
                        
                    # W
                    w_val = s2_W[t][r*2+c]
                    if w_val != 0:
                        w_idx = (c_off[0] + r) * n + (c_off[1] + c)
                        W[w_idx, start_col + t] += w_val * scale_w
                        
    # M1: (A11+A22)(B11+B22)
    # This is not a single 2x2x2 multiplication. It's a sum of two.
    # (A11+A22) is a 2x2 matrix. (B11+B22) is a 2x2 matrix.
    # Their product is a 2x2 matrix.
    # We need to decompose the multiplication of two 2x2 matrices.
    # But the inputs are sums.
    # U_vec for A11+A22: 1 at A11, 1 at A22.
    # V_vec for B11+B22: 1 at B11, 1 at B22.
    # This is NOT a rank-1 tensor in the 2x2x2 space?
    # Wait. The rank-1 tensor is $u \otimes v \otimes w$.
    # Here $u$ has support on A11 and A22.
    # $v$ has support on B11 and B22.
    # $w$ has support on C11 and C22.
    # This is ONE rank-1 tensor?
    # No, the multiplication $ (A11+A22)(B11+B22) $ is a bilinear operation.
    # To compute it, we need 7 multiplications (Strassen).
    # So we apply the 2x2x2 Strassen decomposition to the virtual matrices $A' = A11+A22$ and $B' = B11+B22$.
    # The U vectors will have 1s at the positions of A11 and A22.
    # Specifically, for each of the 7 terms, the U vector is $u_t \otimes (e_{11} + e_{22})$?
    # No.
    # Let $A'_{ij} = A_{ij} + A_{i+2, j+2}$.
    # $B'_{ij} = B_{ij} + B_{i+2, j+2}$.
    # $C'_{ij} = \sum_k A'_{ik} B'_{kj}$.
    # We decompose $C' = A' B'$ using 7 rank-1 terms.
    # Term $t$: $u_t \otimes v_t \otimes w_t$.
    # $u_t$ is a vector in $\mathbb{R}^4$ (for the 2x2 block $A'$).
    # To map to the full 4x4 A, we place $u_t$ at A11 and A22.
    # So $U_{full}$ has $u_t$ at A11 and $u_t$ at A22.
    
    # Let's implement this logic.
    
    def add_strassen_term(a_blocks, b_blocks, c_blocks, start_col):
        # a_blocks: list of (row_off, col_off) for A
        # b_blocks: list of (row_off, col_off) for B
        # c_blocks: list of (row_off, col_off) for C
        for t in range(7):
            for r in range(2):
                for c in range(2):
                    u_val = s2_U[t][r*2+c]
                    if u_val != 0:
                        for (ar, ac) in a_blocks:
                            u_idx = (ar + r) * m + (ac + c)
                            U[u_idx, start_col + t] += u_val
                            
                    v_val = s2_V[t][r*2+c]
                    if v_val != 0:
                        for (br, bc) in b_blocks:
                            v_idx = (br + r) * p + (bc + c)
                            V[v_idx, start_col + t] += v_val
                            
                    w_val = s2_W[t][r*2+c]
                    if w_val != 0:
                        for (cr, cc) in c_blocks:
                            w_idx = (cr + r) * n + (cc + c)
                            W[w_idx, start_col + t] += w_val

    # M1: A' = A11+A22, B' = B11+B22, C' = C11+C22
    # A11: (0,0), A22: (2,2)
    # B11: (0,0), B22: (2,2)
    # C11: (0,0), C22: (2,2)
    add_strassen_term([(0,0), (2,2)], [(0,0), (2,2)], [(0,0), (2,2)], 0)
    
    # M2: A' = A21+A22, B' = B11, C' = C21+C22
    # A21: (2,0), A22: (2,2)
    # B11: (0,0)
    # C21: (2,0), C22: (2,2)
    add_strassen_term([(2,0), (2,2)], [(0,0)], [(2,0), (2,2)], 7)
    
    # M3: A' = A11, B' = B12-B22, C' = C12+C22
    # A11: (0,0)
    # B12: (0,2), B22: (2,2) with sign -1
    # C12: (0,2), C22: (2,2)
    # We need to handle the sign -1 for B22.
    # We can call add_strassen_term and then flip signs for B22?
    # Or modify the function to accept signs.
    # Let's just do it manually for M3, M4, M6, M7 which have signs.
    
    # M3: A11 (B12 - B22)
    # U: A11
    # V: B12 (1), B22 (-1)
    # W: C12, C22
    for t in range(7):
        for r in range(2):
            for c in range(2):
                u_val = s2_U[t][r*2+c]
                if u_val != 0:
                    u_idx = (0 + r) * m + (0 + c)
                    U[u_idx, 14 + t] += u_val
                v_val = s2_V[t][r*2+c]
                if v_val != 0:
                    # B12
                    v_idx1 = (0 + r) * p + (2 + c)
                    V[v_idx1, 14 + t] += v_val
                    # B22 with -1
                    v_idx2 = (2 + r) * p + (2 + c)
                    V[v_idx2, 14 + t] -= v_val
                w_val = s2_W[t][r*2+c]
                if w_val != 0:
                    w_idx1 = (0 + r) * n + (2 + c)
                    W[w_idx1, 14 + t] += w_val
                    w_idx2 = (2 + r) * n + (2 + c)
                    W[w_idx2, 14 + t] += w_val

    # M4: A22 (B21 - B11)
    for t in range(7):
        for r in range(2):
            for c in range(2):
                u_val = s2_U[t][r*2+c]
                if u_val != 0:
                    u_idx = (2 + r) * m + (2 + c)
                    U[u_idx, 21 + t] += u_val
                v_val = s2_V[t][r*2+c]
                if v_val != 0:
                    v_idx1 = (2 + r) * p + (0 + c)
                    V[v_idx1, 21 + t] += v_val
                    v_idx2 = (0 + r) * p + (0 + c)
                    V[v_idx2, 21 + t] -= v_val
                w_val = s2_W[t][r*2+c]
                if w_val != 0:
                    w_idx1 = (0 + r) * n + (0 + c)
                    W[w_idx1, 21 + t] += w_val
                    w_idx2 = (2 + r) * n + (0 + c)
                    W[w_idx2, 21 + t] += w_val

    # M5: (A11+A12) B22
    add_strassen_term([(0,0), (0,2)], [(2,2)], [(0,0), (0,2)], 28)

    # M6: (A21-A11) (B11+B12)
    for t in range(7):
        for r in range(2):
            for c in range(2):
                u_val = s2_U[t][r*2+c]
                if u_val != 0:
                    u_idx1 = (2 + r) * m + (0 + c)
                    U[u_idx1, 35 + t] += u_val
                    u_idx2 = (0 + r) * m + (0 + c)
                    U[u_idx2, 35 + t] -= u_val
                v_val = s2_V[t][r*2+c]
                if v_val != 0:
                    v_idx1 = (0 + r) * p + (0 + c)
                    V[v_idx1, 35 + t] += v_val
                    v_idx2 = (0 + r) * p + (2 + c)
                    V[v_idx2, 35 + t] += v_val
                w_val = s2_W[t][r*2+c]
                if w_val != 0:
                    w_idx = (2 + r) * n + (2 + c)
                    W[w_idx, 35 + t] += w_val

    # M7: (A12-A22) (B21+B22)
    for t in range(7):
        for r in range(2):
            for c in range(2):
                u_val = s2_U[t][r*2+c]
                if u_val != 0:
                    u_idx1 = (0 + r) * m + (2 + c)
                    U[u_idx1, 42 + t] += u_val
                    u_idx2 = (2 + r) * m + (2 + c)
                    U[u_idx2, 42 + t] -= u_val
                v_val = s2_V[t][r*2+c]
                if v_val != 0:
                    v_idx1 = (2 + r) * p + (0 + c)
                    V[v_idx1, 42 + t] += v_val
                    v_idx2 = (2 + r) * p + (2 + c)
                    V[v_idx2, 42 + t] += v_val
                w_val = s2_W[t][r*2+c]
                if w_val != 0:
                    w_idx = (0 + r) * n + (0 + c)
                    W[w_idx, 42 + t] += w_val

    # Part 2: 4x4x1 Schoolbook
    # k=4
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1

    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
