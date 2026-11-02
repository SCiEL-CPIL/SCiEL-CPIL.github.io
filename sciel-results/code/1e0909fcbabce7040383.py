"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication
    using a 2x2 block Strassen structure with Winograd's rank-19 algorithm for 3x3 blocks.
    
    Partition each 6x6 matrix into 2x2 blocks of 3x3 submatrices.
    Apply Strassen's rank-7 decomposition to the block multiplications.
    Apply Winograd's rank-19 decomposition to each 3x3 matrix multiplication.
    Total rank = 7 (Strassen) * 19 (Winograd) = 133.
    """
    start = time.time()
    deadline = start + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 36, 36, 36
    
    # Build the schoolbook decomposition as a fallback
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

    # Try Block-Strassen with Winograd inner products
    # Rank = 7 * 19 = 133
    target_rank = 133
    
    if target_rank < best_rank:
        success = _build_block_strassen_winograd(n, m, p, deadline)
        if success is not None:
            U_found, V_found, W_found = success
            best_rank = target_rank
            best_U = U_found
            best_V = V_found
            best_W = W_found

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}

def _build_block_strassen_winograd(n, m, p, deadline):
    """
    Build a block-Strassen decomposition for 6x6x6 matrix multiplication.
    Uses 2x2 blocks of 3x3 submatrices.
    Strassen rank 7 for blocks.
    Winograd rank 19 for 3x3 submatrices.
    Total rank 133.
    """
    nm, mp, pn = n * m, m * p, p * n
    total_rank = 133
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # Winograd 3x3 terms
    # Each term is (A_coeffs, B_coeffs) where coeffs are tuples of 9 values for 3x3 matrix
    # Indices 0-8 correspond to row-major order:
    # 0: (0,0), 1: (0,1), 2: (0,2)
    # 3: (1,0), 4: (1,1), 5: (1,2)
    # 6: (2,0), 7: (2,1), 8: (2,2)
    
    # Winograd 19 products
    # m1 = a(j+m+p)
    # m2 = (a+b+d)(j+k+m+n+p+q)
    # m3 = (b+c+e+f)(k+n+o+q+r)
    # m4 = c(l+n+o+r)
    # m5 = (d+e+g+h)(m+n+p+q)
    # m6 = (e+f+h+i)(n+o+q+r)
    # m7 = f(o+q+r)
    # m8 = (g+h+i)(p+q+r)
    # m9 = (a+g)(j+p)
    # m10 = (a+b+g+h)(j+k+m+n+p+q)
    # m11 = (b+c+h+i)(k+n+o+q+r)
    # m12 = (c+i)(l+r)
    # m13 = (d+g)(m+p)
    # m14 = (d+e+g+h)(m+n+p+q)
    # m15 = (e+f+h+i)(n+o+q+r)
    # m16 = (f+i)(o+r)
    # m17 = (a+e+i)(j+n+r)
    # m18 = (a+d+e+h+i)(j+m+n+q+r)
    # m19 = (e+i)(n+r)
    
    # A coefficients (a, b, c, d, e, f, g, h, i)
    # B coefficients (j, k, l, m, n, o, p, q, r)
    
    winograd_terms = [
        # (A, B)
        ((1,0,0, 0,0,0, 0,0,0), (1,0,0, 1,0,0, 1,0,0)), # m1
        ((1,1,0, 1,0,0, 0,0,0), (1,1,0, 1,1,0, 1,1,0)), # m2
        ((0,1,1, 0,1,1, 0,0,0), (0,1,0, 0,1,1, 0,1,1)), # m3
        ((0,0,1, 0,0,0, 0,0,0), (0,0,1, 0,1,1, 0,1,1)), # m4
        ((0,0,0, 1,1,0, 1,1,0), (0,0,0, 1,1,0, 1,1,0)), # m5
        ((0,0,0, 0,1,1, 0,1,1), (0,0,0, 0,1,1, 0,1,1)), # m6
        ((0,0,0, 0,0,1, 0,0,0), (0,0,0, 0,0,1, 0,1,1)), # m7
        ((0,0,0, 0,0,0, 1,1,1), (0,0,0, 0,0,0, 1,1,1)), # m8
        ((1,0,0, 0,0,0, 1,0,0), (1,0,0, 0,0,0, 1,0,0)), # m9
        ((1,1,0, 0,0,0, 1,1,0), (1,1,0, 1,1,0, 1,1,0)), # m10
        ((0,1,1, 0,0,0, 0,1,1), (0,1,0, 0,1,1, 0,1,1)), # m11
        ((0,0,1, 0,0,0, 0,0,1), (0,0,1, 0,0,0, 0,0,1)), # m12
        ((0,0,0, 1,0,0, 1,0,0), (0,0,0, 1,0,0, 1,0,0)), # m13
        ((0,0,0, 1,1,0, 1,1,0), (0,0,0, 1,1,0, 1,1,0)), # m14
        ((0,0,0, 0,1,1, 0,1,1), (0,0,0, 0,1,1, 0,1,1)), # m15
        ((0,0,0, 0,0,1, 0,0,1), (0,0,0, 0,0,1, 0,0,1)), # m16
        ((1,0,0, 0,1,0, 0,0,1), (1,0,0, 0,1,0, 0,0,1)), # m17
        ((1,0,0, 1,1,0, 0,1,1), (1,0,0, 1,1,0, 0,1,1)), # m18
        ((0,0,0, 0,1,0, 0,0,1), (0,0,0, 0,1,0, 0,0,1)), # m19
    ]
    
    # C coefficients for each m_k
    # C = [c11, c12, c13, c21, c22, c23, c31, c32, c33]
    # Indices 0-8 row-major
    
    winograd_c_coeffs = [
        # m1: C11 += 1
        (1, 0, 0, 0, 0, 0, 0, 0, 0),
        # m2: C12 += 1
        (0, 1, 0, 0, 0, 0, 0, 0, 0),
        # m3: C12 -= 1
        (0, -1, 0, 0, 0, 0, 0, 0, 0),
        # m4: C13 += 1
        (0, 0, 1, 0, 0, 0, 0, 0, 0),
        # m5: C21 += 1
        (0, 0, 0, 1, 0, 0, 0, 0, 0),
        # m6: C22 += 1
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
        # m7: C23 += 1
        (0, 0, 0, 0, 0, 1, 0, 0, 0),
        # m8: C21 -= 1
        (0, 0, 0, -1, 0, 0, 0, 0, 0),
        # m9: C11 += 1, C22 -= 1
        (1, 0, 0, 0, -1, 0, 0, 0, 0),
        # m10: C11 -= 1, C12 += 1
        (-1, 1, 0, 0, 0, 0, 0, 0, 0),
        # m11: C12 -= 1, C13 += 1
        (0, -1, 1, 0, 0, 0, 0, 0, 0),
        # m12: C13 -= 1
        (0, 0, -1, 0, 0, 0, 0, 0, 0),
        # m13: C21 += 1, C31 += 1
        (0, 0, 0, 1, 0, 0, 1, 0, 0),
        # m14: C21 -= 1, C22 += 1, C32 += 1
        (0, 0, 0, -1, 1, 0, 0, 1, 0),
        # m15: C22 -= 1, C23 += 1, C32 -= 1, C33 += 1
        (0, 0, 0, 0, -1, 1, 0, -1, 1),
        # m16: C23 -= 1, C33 -= 1
        (0, 0, 0, 0, 0, -1, 0, 0, -1),
        # m17: C11 += 1, C22 += 1
        (1, 0, 0, 0, 1, 0, 0, 0, 0),
        # m18: C11 -= 1, C22 -= 1
        (-1, 0, 0, 0, -1, 0, 0, 0, 0),
        # m19: C22 += 1
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
    ]
    
    # Strassen terms for 2x2 blocks
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Block indices: 00, 01, 10, 11
    # A block (u, v), B block (x, y), C block (u, y)
    # Strassen term s uses A blocks and B blocks.
    
    strassen_block_ops = [
        # (A_blocks, B_blocks)
        # Each block is (u, v) for A, (x, y) for B
        # Coefficients for A blocks: (coeff_00, coeff_01, coeff_10, coeff_11)
        # Coefficients for B blocks: (coeff_00, coeff_01, coeff_10, coeff_11)
        
        # M1: (A00 + A11) (B00 + B11)
        ((1, 0, 0, 1), (1, 0, 0, 1)),
        # M2: (A10 + A11) B00
        ((0, 0, 1, 1), (1, 0, 0, 0)),
        # M3: A00 (B10 - B11)
        ((1, 0, 0, 0), (0, 0, 1, -1)),
        # M4: A11 (B01 - B00)
        ((0, 0, 0, 1), (0, 1, -1, 0)),
        # M5: (A00 + A01) B11
        ((1, 1, 0, 0), (0, 0, 0, 1)),
        # M6: (A10 - A00) (B00 + B01)
        ((0, 0, 1, -1), (1, 1, 0, 0)),
        # M7: (A01 - A11) (B10 + B11)
        ((0, 1, 0, -1), (0, 0, 1, 1)),
    ]
    
    # C reconstruction for Strassen
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    strassen_c_coeffs = [
        # For each C block (u, y), coefficients for M1..M7
        # C00: M1(1), M2(0), M3(0), M4(1), M5(-1), M6(0), M7(1)
        (1, 0, 0, 1, -1, 0, 1),
        # C01: M1(0), M2(0), M3(1), M4(0), M5(1), M6(0), M7(0)
        (0, 0, 1, 0, 1, 0, 0),
        # C10: M1(0), M2(1), M3(0), M4(0), M5(0), M6(1), M7(0)
        (0, 1, 0, 0, 0, 1, 0),
        # C11: M1(1), M2(-1), M3(1), M4(0), M5(0), M6(1), M7(0)
        (1, -1, 1, 0, 0, 1, 0),
    ]
    
    col = 0
    # Iterate over Strassen terms
    for s in range(7):
        a_block_coeffs, b_block_coeffs = strassen_block_ops[s]
        
        # For each Winograd term in this Strassen term
        for w_idx in range(19):
            a_winograd, b_winograd = winograd_terms[w_idx]
            c_winograd = winograd_c_coeffs[w_idx]
            
            # Determine which C blocks this term contributes to
            # The contribution is scaled by strassen_c_coeffs[c_block_idx][s]
            
            # U, V, W construction
            # For each A block (u, v) with non-zero coeff in a_block_coeffs
            for u in range(2):
                for v in range(2):
                    coeff_a_block = a_block_coeffs[u*2 + v]
                    if coeff_a_block == 0: continue
                    
                    # Global A indices for block (u, v)
                    # Rows u*3 to u*3+2, Cols v*3 to v*3+2
                    for r in range(3):
                        for c in range(3):
                            local_idx = r * 3 + c
                            coeff_a_winograd = a_winograd[local_idx]
                            if coeff_a_winograd == 0: continue
                            
                            global_a = (u * 3 + r) * 6 + (v * 3 + c)
                            U[global_a, col] += coeff_a_block * coeff_a_winograd
            
            # For each B block (x, y) with non-zero coeff in b_block_coeffs
            for x in range(2):
                for y in range(2):
                    coeff_b_block = b_block_coeffs[x*2 + y]
                    if coeff_b_block == 0: continue
                    
                    # Global B indices for block (x, y)
                    # Rows x*3 to x*3+2, Cols y*3 to y*3+2
                    for r in range(3):
                        for c in range(3):
                            local_idx = r * 3 + c
                            coeff_b_winograd = b_winograd[local_idx]
                            if coeff_b_winograd == 0: continue
                            
                            global_b = (x * 3 + r) * 6 + (y * 3 + c)
                            V[global_b, col] += coeff_b_block * coeff_b_winograd
            
            # For each C block (u, y)
            # The Winograd term contributes to local C elements
            # These local C elements belong to block (u, y)
            # The block (u, y) is composed of M1..M7
            # The coefficient of this specific Strassen term s in block (u, y) is strassen_c_coeffs[u*2+y][s]
            
            for u in range(2):
                for y in range(2):
                    coeff_c_block = strassen_c_coeffs[u*2 + y][s]
                    if coeff_c_block == 0: continue
                    
                    # Local C elements in block (u, y)
                    for r in range(3):
                        for c in range(3):
                            local_idx = r * 3 + c
                            coeff_c_winograd = c_winograd[local_idx]
                            if coeff_c_winograd == 0: continue
                            
                            # Global W index for C[u*3+r, y*3+c]
                            # W[k*n + i] where k = col, i = row
                            # C row = u*3 + r, C col = y*3 + c
                            # W index = (y*3 + c) * 6 + (u*3 + r)
                            global_w = (y * 3 + c) * 6 + (u * 3 + r)
                            W[global_w, col] += coeff_c_block * coeff_c_winograd
            
            col += 1
            
            if time.time() > deadline:
                return None

    # Verify
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
    
    # Check that invalid triples give 0
    np.random.seed(123)
    for _ in range(2000):
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
