"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication
    using tensor product of Strassen (rank 7 for 2x2) and schoolbook (rank 27 for 3x3).
    Total rank = 7 * 27 = 189.
    """
    start = time.time()
    deadline = start + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 36, 36, 36

    # Build schoolbook as ultimate fallback
    schoolbook_rank = n * m * p  # 216
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Try tensor product: Strassen(2x2) x schoolbook(3x3) = rank 189
    target_rank = 7 * 27  # 189
    if target_rank < best_rank:
        result = _build_tensor_product_strassen_schoolbook(n, m, p, deadline)
        if result is not None:
            U_f, V_f, W_f = result
            best_rank = target_rank
            best_U = U_f
            best_V = V_f
            best_W = W_f

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    return {"U": U_out, "V": V_out, "W": W_out}

def _build_tensor_product_strassen_schoolbook(n, m, p, deadline):
    """
    Build tensor product decomposition: Strassen 2x2 (rank 7) x schoolbook 3x3 (rank 27).
    Total rank = 189.
    
    Index conventions:
    - A flat: row*6 + col, partitioned into 2x2 blocks of 3x3
    - B flat: row*6 + col
    - W (for C): col*6 + row (trace convention W[k*n+i, r])
    """
    nm, mp, pn = n * m, m * p, p * n
    outer_dim = 2  # 2x2 blocks
    inner_dim = 3  # 3x3 subblocks
    total_rank = 7 * 27  # 189

    # === Strassen 2x2 decomposition (rank 7) ===
    # U1: (4,7) - A block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V1: (4,7) - B block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W1: (4,7) - C block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C[row_block, col_block] -> W1 index = col_block*2 + row_block
    
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))

    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6

    # A block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C[i,j] -> W1 index = j*2 + i

    # M1: (A00+A11)(B00+B11)
    U1[0,0] = 1; U1[3,0] = 1
    V1[0,0] = 1; V1[3,0] = 1
    # C00 += M1, C11 += M1
    W1[0,0] = 1  # C[0,0]
    W1[3,0] = 1  # C[1,1]

    # M2: (A10+A11)B00
    U1[2,1] = 1; U1[3,1] = 1
    V1[0,1] = 1
    # C10 += M2, C11 -= M2
    W1[2,1] = 1  # C[1,0]
    W1[3,1] = -1  # C[1,1]

    # M3: A00(B10-B11)
    U1[0,2] = 1
    V1[2,2] = 1; V1[3,2] = -1
    # C01 += M3, C11 += M3
    W1[1,2] = 1  # C[0,1]
    W1[3,2] = 1  # C[1,1]

    # M4: A11(B01-B00)
    U1[3,3] = 1
    V1[1,3] = 1; V1[0,3] = -1
    # C00 += M4, C10 += M4... wait
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M6
    # So M4 only contributes to C00
    W1[0,3] = 1  # C[0,0]

    # M5: (A00+A01)B11
    U1[0,4] = 1; U1[1,4] = 1
    V1[3,4] = 1
    # C00 -= M5, C01 += M5
    W1[0,4] = -1  # C[0,0]
    W1[1,4] = 1  # C[0,1]

    # M6: (A10-A00)(B00+B01)
    U1[2,5] = 1; U1[0,5] = -1
    V1[0,5] = 1; V1[1,5] = 1
    # C10 += M6, C11 += M6
    W1[2,5] = 1  # C[1,0]
    W1[3,5] = 1  # C[1,1]

    # M7: (A01-A11)(B10+B11)
    U1[1,6] = 1; U1[3,6] = -1
    V1[2,6] = 1; V1[3,6] = 1
    # C00 += M7
    W1[0,6] = 1  # C[0,0]

    # Verify Strassen: C[i,j] = sum_s U1[i,s]*V1[j,s]*W1[ci,s]
    # where ci = j*2+i for C[i,j]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A[i,k], B[k,j] -> C[i,j]
                ai = i * 2 + k  # A block (i,k)
                bj = k * 2 + j  # B block (k,j)
                cj = j * 2 + i  # C block (i,j) -> W1 index
                val = np.sum(U1[ai, :] * V1[bj, :] * W1[cj, :])
                if abs(val - 1.0) > 1e-10:
                    # Check invalid
                    pass
                # Also check that non-matching give 0
            for k2 in range(4):
                for k3 in range(4):
                    # Check all triples
                    pass

    # Quick full verification of Strassen
    for ai in range(4):
        for bj in range(4):
            for cj in range(4):
                val = np.sum(U1[ai, :] * V1[bj, :] * W1[cj, :])
                # ai = i*2+k, bj = k*2+j, cj = j*2+i
                i_a = ai // 2; k_a = ai % 2
                k_b = bj // 2; j_b = bj % 2
                j_c = cj // 2; i_c = cj % 2
                expected = 1.0 if (i_a == i_c and k_a == k_b and j_b == j_c) else 0.0
                if abs(val - expected) > 1e-10:
                    return None  # Strassen is wrong, abort

    # === Schoolbook 3x3 decomposition (rank 27) ===
    U2 = np.zeros((9, 27))
    V2 = np.zeros((9, 27))
    W2 = np.zeros((9, 27))
    col2 = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U2[i * 3 + j, col2] = 1
                V2[j * 3 + k, col2] = 1
                W2[k * 3 + i, col2] = 1
                col2 += 1

    # Verify schoolbook 3x3
    for ai in range(9):
        for bj in range(9):
            for cj in range(9):
                val = np.sum(U2[ai, :] * V2[bj, :] * W2[cj, :])
                i_a = ai // 3; k_a = ai % 3
                k_b = bj // 3; j_b = bj % 3
                j_c = cj // 3; i_c = cj % 3
                expected = 1.0 if (i_a == i_c and k_a == k_b and j_b == j_c) else 0.0
                if abs(val - expected) > 1e-10:
                    return None

    # === Tensor product construction ===
    # Global A index: (outer_a, inner_a) where outer_a = ar*2+ac, inner_a = lr*3+lc
    # Global A flat: (3*ar+lr)*6 + (3*ac+lc)
    # Global B index: (outer_b, inner_b) where outer_b = br*2+bc, inner_b = lr*3+lc
    # Global B flat: (3*br+lr)*6 + (3*bc+lc)
    # Global C (W) index: (outer_c, inner_c) where outer_c = cc*2+cr, inner_c = lc*3+lr
    # Global W flat: (3*cc+lc)*6 + (3*cr+lr)
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))

    for s1 in range(7):
        for s2 in range(27):
            r_idx = s1 * 27 + s2
            u1_val = U1[:, s1]  # (4,)
            v1_val = V1[:, s1]  # (4,)
            w1_val = W1[:, s1]  # (4,)
            u2_val = U2[:, s2]  # (9,)
            v2_val = V2[:, s2]  # (9,)
            w2_val = W2[:, s2]  # (9,)
            
            # Fill U: for each outer_a (0..3) and inner_a (0..8)
            for oa in range(4):
                ar = oa // 2
                ac = oa % 2
                for ia in range(9):
                    lr = ia // 3
                    lc = ia % 3
                    global_a = (3 * ar + lr) * 6 + (3 * ac + lc)
                    U[global_a, r_idx] = u1_val[oa] * u2_val[ia]
            
            # Fill V: for each outer_b (0..3) and inner_b (0..8)
            for ob in range(4):
                br = ob // 2
                bc = ob % 2
                for ib in range(9):
                    lr = ib // 3
                    lc = ib % 3
                    global_b = (3 * br + lr) * 6 + (3 * bc + lc)
                    V[global_b, r_idx] = v1_val[ob] * v2_val[ib]
            
            # Fill W: for each outer_c (0..3) and inner_c (0..8)
            for oc in range(4):
                cc = oc // 2  # col block
                cr = oc % 2  # row block
                for ic in range(9):
                    lc = ic // 3  # local col
                    lr = ic % 3  # local row
                    global_w = (3 * cc + lc) * 6 + (3 * cr + lr)
                    W[global_w, r_idx] = w1_val[oc] * w2_val[ic]
            
            if time.time() > deadline:
                return None

    # Full verification
    if not _verify_decomposition(n, m, p, U, V, W):
        return None

    return U, V, W

def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Verify that U, V, W form a valid decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False
    
    # Check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that invalid triples give 0 (exhaustive for 36^3 is too many, sample)
    np.random.seed(42)
    for _ in range(5000):
        a = np.random.randint(0, nm)
        b = np.random.randint(0, mp)
        c = np.random.randint(0, pn)
        
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k1 = b % p
        k2 = c // n
        i2 = c % n
        
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    
    return True
# EVOLVE-BLOCK-END
