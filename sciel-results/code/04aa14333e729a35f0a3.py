"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _validate(U, V, W):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    import numpy as np
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    rank = U.shape[1]
    # The tensor is T[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check by verifying that for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta(i,j,k)
    # where delta = 1 if i<m (always true since n=2,m=4), j<m (always), k<p (always)
    # Actually the multiplication tensor has T[i,j,k] = 1 for all valid i,j,k.
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # Efficient check: for each c, the outer product of reshaped U[:,c], V[:,c], W[:,c]
    # U reshaped: (n, m) per column
    # V reshaped: (m, p) per column
    # W reshaped: (p, n) per column
    # The product at (i,j,k) is sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Build full tensor T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        # Use einsum or manual
        for i in range(n):
            for j in range(m):
                uval = Uc[i, j]
                if uval != 0:
                    for k in range(p):
                        T[i, j, k] += uval * Vc[j, k] * Wc[k, i]
    
    # T should be all ones
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    """
    Use block Strassen decomposition.
    
    The 2x4 x 4x6 multiplication can be viewed as 4 blocks of 2x2 x 2x2,
    each handled by rank-7 Strassen. Total rank = 4 * 7 = 28.
    
    Wait, that's not quite right. The structure is:
    A is 2x4, B is 4x6.
    Split into:
    A = [A_L | A_R] where A_L is 2x2, A_R is 2x2
    B = [B_T; B_B] where B_T is 4x3, B_B is 4x3
    
    Actually B is 4x6, so B_T is 4x3 (rows 0-1, cols 0-2), B_B is 4x3 (rows 2-3, cols 3-5)?
    No, that doesn't work dimensionally.
    
    Let me think again. A is 2x4, B is 4x6.
    C = A * B is 2x6.
    
    Split A columns into two halves: A_L (2x2), A_R (2x2)
    Split B rows into two halves: B_T (4x6 rows 0-1... no, B is 4x6, rows are 0-3, cols 0-5)
    
    Actually, the natural block decomposition for Strassen-like speedup:
    Split the intermediate dimension (4) into two parts of 2.
    A = [A1 | A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x6
    
    Then C = A1*B1 + A2*B2.
    
    Each product A1*B1 is 2x2 * 2x6, and A2*B2 is 2x2 * 2x6.
    
    For 2x2 * 2x6: split B1 into [B1a | B1b] where B1a, B1b are 2x3.
    Then A1*B1 = A1*B1a + A1*B1b, each 2x2 * 2x3.
    
    For 2x2 * 2x3: split B into [B_top; B_bottom] where B_top, B_bottom are 1x3? No.
    
    Let me use the standard approach: each 2x2x2 block has rank 7.
    The 2x4 x 4x6 can be decomposed into (2/2)*(4/2)*(6/2) = 1*2*3 = 6 blocks of 2x2x2?
    No, that's not how it works.
    
    Actually, the correct block decomposition:
    n=2, m=4, p=6.
    Split m into 2 blocks of 2, p into 3 blocks of 2.
    So we have 1 * 2 * 3 = 6 sub-problems of 2x2 x 2x2.
    Each has rank 7, total rank = 42.
    
    Let me implement this.
    """
    start_time = time.time()
    DEADLINE = 145  # 5 seconds margin
    
    n, m, p = DIMENSIONS
    
    # Block sizes: split m into m_blocks blocks of m_size, p into p_blocks of p_size
    m_size = 2
    p_size = 2
    n_size = 2  # n is already 2
    
    m_blocks = m // m_size  # 2
    p_blocks = p // p_size  # 3
    n_blocks = n // n_size  # 1
    
    # Each sub-block is n_size x m_size x p_size = 2x2x2, rank 7
    rank = n_blocks * m_blocks * p_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition: 7 terms
    # For local indices i,j,k in {0,1}:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)(B00)     -> C10 += M2
    # M3 = (A00)(B10+B11)     -> C01 += M3
    # M4 = (A01)(B01-B11)     -> C00 += M4, C11 -= M4
    # M5 = (A00+A01)(B11)     -> C11 += M5
    # M6 = (A10-A00)(B00+B01) -> C10 += M6
    # M7 = (A01-A11)(B10+B11) -> C01 += M7
    
    # For each M_s, we have coefficients u (for A, 4 entries), v (for B, 4 entries), w (for C, 4 entries)
    # A indices: (i,j) -> idx = i*2 + j
    # B indices: (j,k) -> idx = j*2 + k  
    # C indices: (k,i) -> idx = k*2 + i  (using trace convention W[k*n+i])
    
    strassen_terms = [
        # (u_dict, v_dict, w_dict)
        # M1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),       # A00,A11; B00,B11; C00,C11
        # M2
        ({2:1, 3:1}, {0:1}, {2:1}),                  # A10,A11; B00; C10
        # M3
        ({0:1}, {2:1, 3:1}, {1:1}),                  # A00; B10,B11; C01
        # M4
        ({1:1}, {1:1, 3:-1}, {0:1, 3:-1}),           # A01; B01,-B11; C01... wait
        # Let me re-derive Strassen more carefully.
    ]
    
    # Standard Strassen for 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b11-b12)  -- wait, this varies by source
    # Let me use the most common formulation:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10+b11)  
    # M4 = (a01-a11)(b10+b11)  -- hmm
    # 
    # Actually, let me use this well-known set:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10+b11)
    # M4 = (a01)(b01-b11)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M4 + M7
    
    strassen_terms = [
        # M1: u for A, v for B, w for C
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),       # M1: C00+=1, C11+=1
        # M2
        ({2:1, 3:1}, {0:1}, {2:1}),                  # M2: C10+=1
        # M3
        ({0:1}, {2:1, 3:1}, {1:1}),                  # M3: C01+=1
        # M4
        ({1:1}, {1:1, 3:-1}, {0:1, 3:-1}),           # M4: C00+=1, C11-=1
        # M5
        ({0:1, 1:1}, {3:1}, {0:-1, 3:1}),            # M5: C00-=1, C11+=1
        # M6
        ({2:1, 0:-1}, {0:1, 1:1}, {2:1}),            # M6: C10+=1
        # M7
        ({1:1, 3:-1}, {2:1, 3:1}, {1:-1, 3:1}),      # M7: C01-=1, C11+=1
    ]
    
    # Verify: C00 = M1(1) + M4(1) - M5(1) - M7(0) = M1+M4-M5
    # C01 = M3(1) + M5(0) - M7(-1) = M3+M7... wait
    # C01 = M3(1) + M5(0) + M7(-1) = M3 - M7? That's not right.
    # Let me re-check the Strassen formulas.
    
    # Standard Strassen:
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M6
    # C[1,1] = M1 - M4 + M7
    
    # With:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10+b11)
    # M4 = (a01)(b01-b11)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # So:
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M6
    # C[1,1] = M1 - M4 + M7
    
    strassen_terms = [
        # M1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2
        ({2:1, 3:1}, {0:1}, {2:1}),
        # M3
        ({0:1}, {2:1, 3:1}, {1:1}),
        # M4
        ({1:1}, {1:1, 3:-1}, {0:1, 3:-1}),
        # M5
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # M6
        ({2:1, 0:-1}, {0:1, 1:1}, {2:1}),
        # M7
        ({1:1, 3:-1}, {2:1, 3:1}, {0:-1, 3:1}),
    ]
    
    # Now build the full decomposition
    col = 0
    for ib in range(n_blocks):
        for jb in range(m_blocks):
            for kb in range(p_blocks):
                if time.time() - start_time > DEADLINE:
                    break
                # Global offsets
                i_off = ib * n_size
                j_off = jb * m_size
                k_off = kb * p_size
                
                for u_dict, v_dict, w_dict in strassen_terms:
                    if time.time() - start_time > DEADLINE:
                        break
                    for li, coeff_u in u_dict.items():
                        gi = i_off + (li // 2)
                        gj = j_off + (li % 2)
                        U[gi * m + gj][col] = coeff_u
                    
                    for lj, coeff_v in v_dict.items():
                        gj = j_off + (lj // 2)
                        gk = k_off + (lj % 2)
                        V[gj * p + gk][col] = coeff_v
                    
                    for lk, coeff_w in w_dict.items():
                        gk = k_off + (lk % 2)
                        gi = i_off + (lk // 2)
                        W[gk * n + gi][col] = coeff_w
                    
                    col += 1
    
    # Verify correctness
    if _validate(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
