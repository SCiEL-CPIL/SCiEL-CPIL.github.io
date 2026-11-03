"""Matrix multiplication 4x4x8 with Strassen-based rank-98 decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _build_2x2x2_strassen():
    """
    Build the rank-7 Strassen decomposition for 2x2x2.
    Index conventions: 
    U index = i*2+j
    V index = j*2+k
    W index = k*2+i
    """
    # Terms: (U_coeffs, V_coeffs, W_coeffs)
    # U, V, W are vectors of length 4
    terms = [
        ([1, 0, 0, -1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [0, 1, 0, 1], [0, 0, 0, 1]),
        ([1, 0, -1, 0], [1, 0, 1, 0], [1, 0, 0, 0]),
    ]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    return U, V, W

def _build_4x4x4_strassen2():
    """
    Build a rank-49 decomposition for 4x4x4 using Strassen on 2x2 blocks.
    U index = i*4+j
    V index = j*4+k
    W index = k*4+i
    """
    U2, V2, W2 = _build_2x2x2_strassen()
    
    # The 4x4x4 tensor is the Kronecker product of 2x2x2 tensors?
    # Not exactly, it's a block structure.
    # We can construct it by iterating over the 7 operations of the outer Strassen layer
    # and the 7 operations of the inner Strassen layer.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Outer Strassen coefficients for A, B, C
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11
    # B is 4x4, split into 2x2 blocks B00, B01, B10, B11
    # C is 4x4, split into 2x2 blocks C00, C01, C10, C11
    
    # Strassen operations for 2x2x2 (outer):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M5 + M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M4
    
    # We can define the coefficients for the outer layer.
    # Each outer operation 'op' involves a linear combination of A-blocks and B-blocks.
    # For each outer op, we perform a 2x2x2 Strassen multiplication of the resulting 2x2 matrices.
    
    # Outer A coefficients (4 blocks: 00, 01, 10, 11)
    # Outer B coefficients (4 blocks: 00, 01, 10, 11)
    # Outer C coefficients (4 blocks: 00, 01, 10, 11)
    
    outer_ops = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: [A00, A01, A10, A11]
        # B_coeffs: [B00, B01, B10, B11]
        # C_coeffs: [C00, C01, C10, C11]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # M0 -> C00, C11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M1 -> C10, C11
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M2 -> C01, C11
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 0, 1]), # M3 -> C00, C11 (Note: B10-B00 is [0,-1,1,0])
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]), # M4 -> C01, C11
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]), # M5 -> C00, C10 (Note: A10-A00 is [-1,0,1,0], B00+B01 is [1,1,0,0])
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M6 -> C00, C11 (Note: A01-A11 is [0,1,0,-1], B10+B11 is [0,0,1,1])
    ]
    # Wait, the C coefficients need to be mapped to the correct blocks.
    # Let's re-verify the C mapping for Strassen.
    # C00 = M0 + M3 - M5 + M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M4
    
    # My outer_ops list above has C_coeffs as [C00, C01, C10, C11] for that specific M.
    # M0: C00=1, C11=1. So [1, 0, 0, 1].
    # M1: C10=1, C11=-1. So [0, 0, 1, -1].
    # M2: C01=1, C11=1. So [0, 1, 0, 1].
    # M3: C00=1, C11=1. So [1, 0, 0, 1]. (Wait, M3 is A11(B10-B00). C00 += M3, C11 += M3? No.)
    # Let's check standard Strassen:
    # C00 = M0 - M1 + M3 + M6 ? No.
    # Standard:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 - M1 + M3 + M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 + M2 - M4 + M5 ? No.
    # C11 = M0 - M1 + M2 + M4 ?
    # Let's check C11:
    # A11 B11 + A10 B01 + A01 B10 + A00 B00 ? No, C11 = A10 B01 + A11 B11.
    # M0 = A00B00 + A00B11 + A11B00 + A11B11
    # M1 = A10B00 + A11B00
    # M2 = A00B01 - A00B11
    # M3 = A11B10 - A11B00
    # M4 = A00B11 + A01B11
    # M5 = A10B00 + A10B01 - A00B00 - A00B01
    # M6 = A01B10 + A01B11 - A11B10 - A11B11
    
    # C00 = A00B00 + A01B10
    # M0 - M1 + M3 + M6 = (A00B00 + A00B11 + A11B00 + A11B11) - (A10B00 + A11B00) + (A11B10 - A11B00) + (A01B10 + A01B11 - A11B10 - A11B11)
    # = A00B00 + A00B11 - A10B00 + A11B10 - A11B00 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A01B10 + A00B11 - A10B00 - A11B00 + A01B11
    # This is not C00.
    
    # Correct Strassen C formulas:
    # C00 = M0 - M1 + M3 + M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M4 ? No.
    # C11 = M0 + M2 - M4 + M5 ? No.
    # C11 = A10B01 + A11B11
    # M0 + M2 - M4 + M5 = (A00B00 + A00B11 + A11B00 + A11B11) + (A00B01 - A00B11) - (A00B11 + A01B11) + (A10B00 + A10B01 - A00B00 - A00B01)
    # = A11B11 + A10B01 + A11B00 + A00B01 - A00B11 - A01B11 - A00B01 + A10B00 - A00B00
    # This is not C11.
    
    # Let's use the known coefficients from the seed program, which worked.
    # The seed program used:
    # block_U, block_V, c_combos.
    # I will reuse the seed program's logic for the 4x4x4 construction to ensure correctness.

    block_U = [
        [1, 0, 0, -1], # M0: A00+A11? No, A00+A11 is [1,0,0,1]. 
        # Wait, the seed program's block_U[0] was [1, 0, 0, -1].
        # Let's check the seed program's terms.
        # Term 0: U=[1,0,0,-1], V=[1,0,0,1], W=[1,0,0,1]
        # This corresponds to (A00-A11)(B00+B11)?
        # Actually, Strassen can be written with different signs.
        # I will trust the seed program's 4x4x4 construction as it was verified.
        
    ]
    
    # I will copy the seed program's _build_4x4x4_strassen2 logic directly.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))

    U2, V2, W2 = _build_2x2x2_strassen()

    block_U = [
        [1, 0, 0, -1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [1, 0, -1, 0],
    ]
    block_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]
    c_combos = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    for op_idx in range(7):
        lc = block_U[op_idx]
        rc = block_V[op_idx]
        cc = c_combos[op_idx]
        for t in range(7):
            col = op_idx * 7 + t
            for a_blk in range(4):
                if lc[a_blk] == 0:
                    continue
                a_row = a_blk // 2
                a_col = a_blk % 2
                for li in range(2):
                    for lj in range(2):
                        u_idx = (a_row * 2 + li) * 4 + (a_col * 2 + lj)
                        U[u_idx, col] += lc[a_blk] * U2[2 * li + lj, t]
            for b_blk in range(4):
                if rc[b_blk] == 0:
                    continue
                b_row = b_blk // 2
                b_col = b_blk % 2
                for lj in range(2):
                    for lk in range(2):
                        v_idx = (b_row * 2 + lj) * 4 + (b_col * 2 + lk)
                        V[v_idx, col] += rc[b_blk] * V2[2 * lj + lk, t]
            for c_blk in range(4):
                c_coeff = cc[c_blk]
                if c_coeff == 0:
                    continue
                c_row = c_blk // 2
                c_col = c_blk % 2
                for lk in range(2):
                    for li in range(2):
                        w_idx = (c_row * 2 + lk) * 4 + (c_col * 2 + li)
                        W[w_idx, col] += c_coeff * W2[2 * lk + li, t]

    return U, V, W

def _build_4x4x8_rank98():
    """
    Rank-98 decomposition: Strassen for both halves.
    """
    U44, V44, W44 = _build_4x4x4_strassen2()

    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    # C1 = A*B1: columns 0..48
    U[:, 0:49] = U44
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 8 + k, c] = V44[j * 4 + k, c]
    W[0:16, 0:49] = W44

    # C2 = A*B2: columns 49..97
    U[:, 49:98] = U44
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 8 + (k + 4), 49 + c] = V44[j * 4 + k, c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[(k + 4) * 4 + i, 49 + c] = W44[k * 4 + i, c]

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Vectorized verification
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # We want to check for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # This can be done by iterating i, j, k or using einsum.
    # einsum('ir,jr,kr->ijk') won't work directly because indices are mixed.
    # Let's use a loop for i, j, k but vectorize the sum over r.
    
    for i in range(n):
        for j in range(m):
            u_vec = U[i*m + j] # (R,)
            for k in range(p):
                v_vec = V[j*p + k] # (R,)
                w_vec = W[k*n + i] # (R,)
                total = np.dot(u_vec, v_vec * w_vec)
                if abs(total - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = DIMENSIONS  # 4, 4, 8

    try:
        U, V, W = _build_4x4x8_rank98()
        if _verify_decomposition(U, V, W, n, m, p):
            if time.time() < deadline - 5.0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook decomposition (rank 128)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
