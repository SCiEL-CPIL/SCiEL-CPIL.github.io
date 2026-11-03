"""Matrix multiplication 4,4,7 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 matrix multiplication (Rank 7).
    U: shape (4, 7), indices (i, j) -> i*2+j
    V: shape (4, 7), indices (j, k) -> j*2+k
    W: shape (4, 7), indices (k, i) -> k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    # A00=(0,0), A11=(1,1) -> U indices 0, 3
    # B00=(0,0), B11=(1,1) -> V indices 0, 3
    # C00=(0,0), C11=(1,1) -> W indices 0, 3 (Note: W is k*2+i, so C00 is k=0,i=0 -> 0; C11 is k=1,i=1 -> 3)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    # A10=(1,0)->2, A11=(1,1)->3
    # B00=(0,0)->0
    # C10=(0,1)->2? No. C10 is row 1, col 0. W index k*2+i. 
    # Wait, standard Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's verify indices for W (k, i) -> k*2+i
    # C00 (k=0, i=0) -> 0
    # C01 (k=0, i=1) -> 1
    # C10 (k=1, i=0) -> 2
    # C11 (k=1, i=1) -> 3
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # M2 contributes to C10 (2) and C11 (3)?
    # Formula: C10 = M2 + M4. C11 = M1 - M2 + ...
    # So M2 contributes +1 to C10 (idx 2) and -1 to C11 (idx 3).
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1  # B01=(0,1)->1, B11=(1,1)->3
    # M3 contributes to C01 (1) and C11 (3)
    # Formula: C01 = M3 + M5. C11 = ... + M3 ...
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1  # B10=(1,0)->2, B00=(0,0)->0
    # M4 contributes to C00 (0) and C10 (2)
    # Formula: C00 = ... + M4 ... C10 = M2 + M4
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00=(0,0)->0, A01=(0,1)->1
    V[3, 4] = 1  # B11=(1,1)->3
    # M5 contributes to C00 (0) and C01 (1)
    # Formula: C00 = ... - M5 ... C01 = M3 + M5
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # M6 contributes to C11 (3)
    # Formula: C11 = ... + M6
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # M7 contributes to C00 (0)
    # Formula: C00 = ... + M7
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Builds rank-49 decomposition for 4x4 * 4x4 using tensor product of two Strassen 2x2x2.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    U_s, V_s, W_s = _strassen_2x2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # The decomposition is a tensor product of the 2x2x2 decomposition with itself.
    # We are multiplying:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A B
    
    # The outer Strassen terms define how we group the 2x2 blocks of A and B.
    # Let's iterate over the outer Strassen terms (t_outer in 0..6) and inner (t_inner in 0..6).
    
    # Outer coefficients for A blocks (00, 01, 10, 11)
    # Map block index (0..3) to local 2x2 indices
    # Block 00: rows 0-1, cols 0-1
    # Block 01: rows 0-1, cols 2-3
    # Block 10: rows 2-3, cols 0-1
    # Block 11: rows 2-3, cols 2-3
    
    # The outer Strassen algorithm computes 7 intermediate 2x2 matrices M_0..M_6.
    # M_t = (sum a_ij A_ij) * (sum b_kl B_kl)
    # Then C_blocks are linear combos of M_t.
    
    # We need to be careful with the "effective" 2x2 multiplication.
    # The inner Strassen (U_s, V_s, W_s) computes X * Y = Z for 2x2 matrices X, Y, Z.
    # Here, the "entries" of X, Y, Z are themselves 2x2 blocks.
    
    # Let's define the outer coefficients clearly.
    # A_blocks: 0=00, 1=01, 2=10, 3=11
    # B_blocks: 0=00, 1=01, 2=10, 3=11
    
    a_coeffs = [
        {0: 1, 3: 1},   # M0: A00 + A11
        {2: 1, 3: 1},   # M1: A10 + A11
        {0: 1},         # M2: A00
        {3: 1},         # M3: A11
        {0: 1, 1: 1},   # M4: A00 + A01
        {2: 1, 0: -1},  # M5: A10 - A00
        {1: 1, 3: -1}   # M6: A01 - A11
    ]
    
    b_coeffs = [
        {0: 1, 3: 1},   # M0: B00 + B11
        {0: 1},         # M1: B00
        {1: 1, 3: -1},  # M2: B01 - B11
        {2: 1, 0: -1},  # M3: B10 - B00
        {3: 1},         # M4: B11
        {0: 1, 1: 1},   # M5: B00 + B01
        {2: 1, 3: 1}    # M6: B10 + B11
    ]
    
    # W coefficients: which M_t contributes to which C_block?
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Invert to get contribution of each M_t to each C_block
    w_contrib = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11
    ]
    
    def block_to_global(block_idx, local_i, local_j):
        rb = block_idx // 2
        cb = block_idx % 2
        gi = rb * 2 + local_i
        gj = cb * 2 + local_j
        return gi, gj
    
    for t_outer in range(7):
        for t_inner in range(7):
            t_global = t_outer * 7 + t_inner
            
            # Fill U
            for la in range(4): # la is local index in 2x2 block of A
                u_val = U_s[la, t_inner]
                if u_val != 0:
                    li = la // 2
                    lj = la % 2
                    for a_block_idx, a_coeff in a_coeffs[t_outer].items():
                        if a_coeff == 0: continue
                        gi, gj = block_to_global(a_block_idx, li, lj)
                        # U index is i*m + j
                        U[gi * m + gj, t_global] += u_val * a_coeff
                        
            # Fill V
            for lb in range(4): # lb is local index in 2x2 block of B
                v_val = V_s[lb, t_inner]
                if v_val != 0:
                    li = lb // 2
                    lj = lb % 2
                    for b_block_idx, b_coeff in b_coeffs[t_outer].items():
                        if b_coeff == 0: continue
                        gi, gj = block_to_global(b_block_idx, li, lj)
                        # V index is j*p + k
                        V[gi * p + gj, t_global] += v_val * b_coeff
                        
            # Fill W
            for lc in range(4): # lc is local index in 2x2 block of C
                w_val = W_s[lc, t_inner]
                if w_val != 0:
                    li = lc // 2
                    lj = lc % 2
                    for c_block_idx in range(4):
                        if t_outer in w_contrib[c_block_idx]:
                            c_coeff = w_contrib[c_block_idx][t_outer]
                            if c_coeff == 0: continue
                            gi, gj = block_to_global(c_block_idx, li, lj)
                            # W index is k*n + i
                            # gi is row i of C, gj is col k of C?
                            # Wait. block_to_global returns (row, col).
                            # For C, row is i, col is k.
                            # W index is k * n + i.
                            W[gj * n + gi, t_global] += w_val * c_coeff
                            
    return U, V, W

def _build_4x4x3_schoolbook():
    """
    Builds rank-48 decomposition for 4x4 * 4x3 (schoolbook).
    U: (16, 48), V: (12, 48), W: (12, 48)
    """
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n=4, m=4, p=7):
    """
    Fast vectorized verification.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Flatten A and B for vector ops
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # Compute a_t for all t: a_t = sum U[i,j,t] * A[i,j]
    # U is (n*m, rank). A_flat is (n*m,).
    # a_t vector = U.T @ A_flat
    a_vals = U.T @ A_flat
    
    # b_t vector = V.T @ B_flat
    b_vals = V.T @ B_flat
    
    # C_computed[i,k] = sum_t W[k*n+i, t] * a_t * b_t
    # W is (p*n, rank).
    # We can form a vector C_flat of length p*n.
    # C_flat[k*n+i] = sum_t W[k*n+i, t] * (a_t * b_t)
    # Let prod_t = a_t * b_t (vector of length rank)
    prod_vals = a_vals * b_vals
    
    C_flat = W @ prod_vals
    
    C_computed = C_flat.reshape((p, n)).T
    
    diff = np.max(np.abs(C_computed - C_expected))
    return diff < 1e-10, diff

def construct():
    start_time = time.time()
    
    # 1. Build 4x4x4 Strassen (Rank 49)
    U1, V1, W1 = _build_4x4x4_strassen()
    
    # 2. Build 4x4x3 Schoolbook (Rank 48)
    U2, V2, W2 = _build_4x4x3_schoolbook()
    
    # 3. Combine for 4x4x7
    n, m, p = 4, 4, 7
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Part 1: Columns 0-3 of B, Columns 0-3 of C
    U[:, :rank1] = U1
    
    # V1 is (16, 49) for 4x4 B. We need to map to 4x7 B.
    # V1 indices are j*4 + k.
    # New V indices are j*7 + k.
    # k in 0..3.
    for j in range(m):
        for k in range(4):
            V[j*7 + k, :rank1] = V1[j*4 + k, :]
            
    # W1 is (16, 49) for 4x4 C. Indices k*4 + i.
    # New W indices are k*7 + i.
    # k in 0..3.
    for k in range(4):
        for i in range(n):
            W[k*7 + i, :rank1] = W1[k*4 + i, :]
            
    # Part 2: Columns 4-6 of B, Columns 4-6 of C
    offset = rank1
    U[:, offset:] = U2
    
    # V2 is (12, 48) for 4x3 B. Indices j*3 + k.
    # New V indices are j*7 + (4+k).
    # k in 0..2.
    for j in range(m):
        for k in range(3):
            V[j*7 + (4 + k), offset:] = V2[j*3 + k, :]
            
    # W2 is (12, 48) for 4x3 C. Indices k*4 + i.
    # New W indices are (4+k)*7 + i.
    # k in 0..2.
    for k in range(3):
        for i in range(n):
            W[(4 + k)*7 + i, offset:] = W2[k*4 + i, :]
            
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    
    # 4. Verify
    valid, diff = _verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to pure schoolbook 4x4x7
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    return result

# EVOLVE-BLOCK-END
