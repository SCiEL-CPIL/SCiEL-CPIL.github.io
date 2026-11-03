"""5x5 matrix multiplication: rank-reduced decomposition via 4x4 Strassen + correction terms."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _build_2x2_strassen():
    """
    Return U, V, W for 2x2 matrix multiplication using Strassen (rank 7).
    U: 4x7 (rows indexed i*2+j), V: 4x7 (rows j*2+k), W: 4x7 (rows k*2+i)
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # Term 0: P1 = (a00+a11)(b00+b11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # Term 1: P2 = (a10+a11)*b00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # Term 2: P3 = a00*(b01-b11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # Term 3: P4 = a11*(b10-b00)
    U[3,3] = 1
    V[0,3] = -1; V[2,3] = 1
    W[0,3] = 1; W[2,3] = 1
    
    # Term 4: P5 = (a00+a01)*b11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # Term 5: P6 = (a10-a00)*(b00+b01)
    U[0,5] = -1; U[2,5] = 1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # Term 6: P7 = (a01-a11)*(b10+b11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _build_4x4_strassen():
    """
    Build 4x4 Strassen decomposition by composing two levels of 2x2 Strassen.
    Returns U, V, W each of shape (16, 49).
    """
    U2, V2, W2 = _build_2x2_strassen()
    
    U4 = np.zeros((16, 49), dtype=np.int8)
    V4 = np.zeros((16, 49), dtype=np.int8)
    W4 = np.zeros((16, 49), dtype=np.int8)
    
    # Big Strassen terms for 4x4:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For each big term b (0..6) and small term s (0..6):
    # The combined term index is r = b*7 + s
    
    # The u-vector for big term b gives coefficients for the 4x4 A matrix blocks.
    # The small term s gives coefficients for the 2x2 sub-product within that block.
    
    # Big term definitions (coefficients for 2x2 block structure):
    # Each big term involves two 2x2 blocks from A and two from B.
    
    # Let's define for each big term which blocks of A and B are involved and with what signs.
    # A blocks: A00 (top-left), A01 (top-right), A10 (bottom-left), A11 (bottom-right)
    # B blocks: B00, B01, B10, B11
    
    big_terms = [
        # (A_blocks, A_coeffs, B_blocks, B_coeffs)
        ([0, 3], [1, 1], [0, 3], [1, 1]),       # P1
        ([2, 3], [1, 1], [0], [1]),              # P2
        ([0], [1], [1, 3], [1, -1]),             # P3
        ([3], [1], [2, 0], [1, -1]),             # P4
        ([0, 1], [1, 1], [3], [1]),              # P5
        ([2, 0], [1, -1], [0, 1], [1, 1]),       # P6
        ([1, 3], [1, -1], [2, 3], [1, 1]),       # P7
    ]
    
    # For each big term, we need to figure out how the 2x2 Strassen applies.
    # The tricky part: for terms like P1 = (A00+A11)(B00+B11), the "effective" 2x2 matrices
    # are sums/differences of 2x2 blocks.
    
    # Actually, a cleaner approach: for each big term, the product is of two 2x2 matrices
    # where each element is a linear combination of the original 4x4 elements.
    
    # Let me think differently. For big term b, we have:
    # X_b = linear combination of A blocks (each 2x2)
    # Y_b = linear combination of B blocks (each 2x2)
    # The product X_b * Y_b is a 2x2 matrix product, decomposed via Strassen.
    
    # The u-vector for the combined term (b, s) is:
    # For each element (i,j) in the 4x4 A, the coefficient is:
    #   sum over blocks blk of A that contain (i,j) of [coeff_A[blk] * U2[local_i, local_j, s]]
    
    # Where local_i, local_j are the positions within the 2x2 block.
    
    # Block layout in 4x4:
    # Block 0 (A00): rows 0-1, cols 0-1 -> local (i%2, j%2) for i in {0,1}, j in {0,1}
    # Block 1 (A01): rows 0-1, cols 2-3 -> local (i%2, (j-2)%2)
    # Block 2 (A10): rows 2-3, cols 0-1 -> local ((i-2)%2, j%2)
    # Block 3 (A11): rows 2-3, cols 2-3 -> local ((i-2)%2, (j-2)%2)
    
    # For a given global position (i, j) in 4x4:
    # block_row = i // 2, block_col = j // 2
    # local_i = i % 2, local_j = j % 2
    # block_index = block_row * 2 + block_col
    
    for b in range(7):
        A_blocks, A_coeffs, B_blocks, B_coeffs = big_terms[b]
        
        for s in range(7):
            r = b * 7 + s
            
            # Build u-vector for A
            for i in range(4):
                for j in range(4):
                    block_idx = (i // 2) * 2 + (j // 2)
                    local_i = i % 2
                    local_j = j % 2
                    # Check if this block is in A_blocks for this big term
                    if block_idx in A_blocks:
                        coeff = A_coeffs[A_blocks.index(block_idx)]
                        U4[i * 4 + j, r] = coeff * U2[local_i * 2 + local_j, s]
            
            # Build v-vector for B
            for j in range(4):
                for k in range(4):
                    block_idx = (j // 2) * 2 + (k // 2)
                    local_j = j % 2
                    local_k = k % 2
                    if block_idx in B_blocks:
                        coeff = B_coeffs[B_blocks.index(block_idx)]
                        V4[j * 4 + k, r] = coeff * V2[local_j * 2 + local_k, s]
            
            # Build w-vector for C
            # The result block of the big term goes to specific C blocks.
            # C00 gets contributions from P1, P4, P5(neg), P7
            # C01 gets contributions from P3, P5
            # C10 gets contributions from P2, P4
            # C11 gets contributions from P1, P2(neg), P3, P6
            
            # For big term b, which C blocks does it contribute to and with what sign?
            c_contrib = {
                0: [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00: P1(+), P4(+), P5(-), P7(+)
                1: [(2, 1), (4, 1)],                      # C01: P3(+), P5(+)
                2: [(1, 1), (3, 1)],                      # C10: P2(+), P4(+)
                3: [(0, 1), (1, -1), (2, 1), (5, 1)],     # C11: P1(+), P2(-), P3(+), P6(+)
            }
            
            for c_block, contribs in c_contrib.items():
                for (term_idx, sign) in contribs:
                    if term_idx == b:
                        for ki in range(2):
                            for kj in range(2):
                                # Global position in C: row = c_block_row*2 + ki, col = c_block_col*2 + kj
                                c_row = (c_block // 2) * 2 + ki
                                c_col = (c_block % 2) * 2 + kj
                                # W index: k*n + i = c_col * 4 + c_row
                                w_idx = c_col * 4 + c_row
                                # Local position in 2x2 C block
                                local_k = kj
                                local_i = ki
                                W4[w_idx, r] += sign * W2[local_k * 2 + local_i, s]
    
    return U4, V4, W4

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    
    # Random test
    np.random.seed(42)
    for _ in range(3):
        A = np.random.randint(-5, 6, (n, m)).astype(float)
        B = np.random.randint(-5, 6, (m, p)).astype(float)
        C_true = A @ B
        
        # Compute using decomposition
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        C_comp = np.zeros((p, n), dtype=float)
        for r in range(rank):
            ar = U_arr[:, r] @ A_flat
            br = V_arr[:, r] @ B_flat
            C_comp += ar * br * W_arr[:, r].reshape((p, n))
        
        C_comp = C_comp.T  # C_comp is (p, n) = (k, i), we need (i, k)
        
        if not np.allclose(C_comp, C_true, atol=1e-9):
            return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
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

def _build_5x5_decomposition():
    """
    Build 5x5 decomposition using:
    - 4x4 Strassen (rank 49) for the main block
    - Correction terms for the 5th row/column
    """
    U4, V4, W4 = _build_4x4_strassen()
    
    # 5x5 = 4x4 block + correction terms
    # A = [[A44, A41], [A14, A11]] where A44 is 4x4, A41 is 4x1, A14 is 1x4, A11 is 1x1
    # B = [[B44, B41], [B14, B11]]
    # C = A*B:
    # C44 = A44*B44 + A41*B14  (4x4)
    # C41 = A44*B41 + A41*B11  (4x1)
    # C14 = A14*B44 + A11*B14  (1x4)
    # C11 = A14*B41 + A11*B11  (1x1)
    
    # The 4x4 Strassen handles A44*B44 with rank 49.
    # Additional terms needed:
    # 1. A41*B14: outer product (4x1)*(1x4) = rank 4
    # 2. A44*B41: (4x4)*(4x1) = rank 16 (each row dot with B41 col)
    # 3. A41*B11: (4x1)*(1x1) = rank 4
    # 4. A14*B44: (1x4)*(4x4) = rank 16
    # 5. A11*B14: (1x1)*(1x4) = rank 4
    # 6. A14*B41: (1x4)*(4x1) = rank 4
    # 7. A11*B11: (1x1)*(1x1) = rank 1
    
    # Total additional rank: 4+16+4+16+4+4+1 = 49
    # Total rank: 49 + 49 = 98
    
    total_rank = 49 + 49
    U = np.zeros((25, total_rank), dtype=np.int8)
    V = np.zeros((25, total_rank), dtype=np.int8)
    W = np.zeros((25, total_rank), dtype=np.int8)
    
    # First 49 columns: 4x4 Strassen
    # U4 is 16x49, rows 0-15 correspond to A44 elements (indices 0-15 in 5x5)
    U[0:16, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    col = 49
    
    # Now add correction terms.
    # In 5x5 indexing:
    # A[i,j] for i,j in 0..4, stored at U[i*5+j, col]
    # B[j,k] for j,k in 0..4, stored at V[j*5+k, col]
    # C[i,k] for i,k in 0..4, W[k*5+i, col]
    
    # A44: i in 0..3, j in 0..3 -> A[i,j] at index i*5+j
    # A41: i in 0..3, j=4 -> A[i,4] at index i*5+4
    # A14: i=4, j in 0..3 -> A[4,j] at index 4*5+j = 20+j
    # A11: i=4, j=4 -> A[4,4] at index 24
    
    # B44: j in 0..3, k in 0..3 -> B[j,k] at index j*5+k
    # B41: j in 0..3, k=4 -> B[j,4] at index j*5+4
    # B14: j=4, k in 0..3 -> B[4,k] at index 4*5+k = 20+k
    # B11: j=4, k=4 -> B[4,4] at index 24
    
    # C44: i in 0..3, k in 0..3 -> C[i,k] at W[k*5+i]
    # C41: i in 0..3, k=4 -> C[i,4] at W[4*5+i] = W[20+i]
    # C14: i=4, k in 0..3 -> C[4,k] at W[k*5+4]
    # C11: i=4, k=4 -> C[4,4] at W[4*5+4] = W[24]
    
    # Term 1: A41*B14 -> C44 (4x4)
    # For each i in 0..3, k in 0..3: C[i,k] += A[i,4]*B[4,k]
    for i in range(4):
        for k in range(4):
            U[i*5+4, col] = 1
            V[20+k, col] = 1
            W[k*5+i, col] = 1
            col += 1
    
    # Term 2: A44*B41 -> C41 (4x1)
    # For each i in 0..3: C[i,4] += sum_j A[i,j]*B[j,4] for j in 0..3
    for i in range(4):
        for j in range(4):
            U[i*5+j, col] = 1
            V[j*5+4, col] = 1
            W[4*5+i, col] = 1
            col += 1
    
    # Term 3: A41*B11 -> C41 (4x1)
    # For each i in 0..3: C[i,4] += A[i,4]*B[4,4]
    for i in range(4):
        U[i*5+4, col] = 1
        V[24, col] = 1
        W[4*5+i, col] = 1
        col += 1
    
    # Term 4: A14*B44 -> C14 (1x4)
    # For each k in 0..3: C[4,k] += sum_j A[4,j]*B[j,k] for j in 0..3
    for k in range(4):
        for j in range(4):
            U[20+j, col] = 1
            V[j*5+k, col] = 1
            W[k*5+4, col] = 1
            col += 1
    
    # Term 5: A11*B14 -> C14 (1x4)
    # For each k in 0..3: C[4,k] += A[4,4]*B[4,k]
    for k in range(4):
        U[24, col] = 1
        V[20+k, col] = 1
        W[k*5+4, col] = 1
        col += 1
    
    # Term 6: A14*B41 -> C11 (1x1)
    # C[4,4] += sum_j A[4,j]*B[j,4] for j in 0..3
    for j in range(4):
        U[20+j, col] = 1
        V[j*5+4, col] = 1
        W[24, col] = 1
        col += 1
    
    # Term 7: A11*B11 -> C11 (1x1)
    U[24, col] = 1
    V[24, col] = 1
    W[24, col] = 1
    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    start_time = time.time()
    
    deadline = start_time + 115  # 15s margin before 130s limit
    
    # First, get the schoolbook decomposition as fallback
    fallback = _schoolbook(n, m, p)
    
    # Try the 4x4 Strassen + correction approach (rank 98)
    try:
        if time.time() < deadline:
            result = _build_5x5_decomposition()
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return fallback

# EVOLVE-BLOCK-END
