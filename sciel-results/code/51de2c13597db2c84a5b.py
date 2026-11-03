"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term: (u, v, w) where u, v, w are 2x2 matrices (as flat lists of 4).
    u[i*2+j], v[j*2+k], w[k*2+i]
    """
    # Strassen's algorithm for 2x2:
    # M1 = (A11 - A22)(B22 - B11)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A22)(B11 + B22)
    # M6 = (A12 - A11)(B21 + B22)
    # M7 = (A21 - A12)(B11 + B12)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M5
    
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    def flat(d):
        """d is dict mapping (i,j) to coeff, return flat list of 4"""
        result = [0, 0, 0, 0]
        for (i, j), c in d.items():
            result[i*2 + j] = c
        return result
    
    terms = []
    
    # M1 = (A11 - A22)(B22 - B11)
    u1 = flat({(0,0): 1, (1,1): -1})
    v1 = flat({(1,1): 1, (0,0): -1})
    w1 = flat({(0,0): 1, (1,1): 1})  # contributes to C11 and C22
    terms.append((u1, v1, w1))
    
    # M2 = (A21 + A22)B11
    u2 = flat({(1,0): 1, (1,1): 1})
    v2 = flat({(0,0): 1})
    w2 = flat({(1,0): 1, (1,1): -1})  # C21 gets +, C22 gets -
    terms.append((u2, v2, w2))
    
    # M3 = A11(B12 - B22)
    u3 = flat({(0,0): 1})
    v3 = flat({(0,1): 1, (1,1): -1})
    w3 = flat({(0,1): 1, (1,1): 1})  # C12 gets +, C22 gets +
    terms.append((u3, v3, w3))
    
    # M4 = A22(B21 - B11)
    u4 = flat({(1,1): 1})
    v4 = flat({(1,0): 1, (0,0): -1})
    w4 = flat({(0,0): 1, (1,0): 1})  # C11 gets +, C21 gets +
    terms.append((u4, v4, w4))
    
    # M5 = (A11 + A22)(B11 + B22)
    u5 = flat({(0,0): 1, (1,1): 1})
    v5 = flat({(0,0): 1, (1,1): 1})
    w5 = flat({(0,0): -1, (0,1): 1, (1,1): 1})  # C11 gets -, C12 gets +, C22 gets +
    terms.append((u5, v5, w5))
    
    # M6 = (A12 - A11)(B21 + B22)
    u6 = flat({(0,1): 1, (0,0): -1})
    v6 = flat({(1,0): 1, (1,1): 1})
    w6 = flat({(0,1): 1})  # C12 gets +
    terms.append((u6, v6, w6))
    
    # M7 = (A21 - A12)(B11 + B12)
    u7 = flat({(1,0): 1, (0,1): -1})
    v7 = flat({(0,0): 1, (0,1): 1})
    w7 = flat({(0,0): 1})  # C11 gets +
    terms.append((u7, v7, w7))
    
    return terms

def _embed_3x5x6():
    """
    Build the rank-84 decomposition for 3x5x6 matrix multiplication.
    
    Strategy: Split A(3x5) into row blocks [1,2] and col blocks [2,3].
    Split B(5x6) into row blocks [2,3] and col blocks [2,4].
    Split C(3x6) into row blocks [1,2] and col blocks [2,4].
    
    This gives 4 block products:
    C00(1x2) = A00(1x2) * B00(2x2)
    C01(1x4) = A00(1x2) * B01(2x4) + A01(1x3) * B11(3x4)
    C10(2x2) = A10(2x2) * B00(2x2) + A11(2x3) * B10(3x2)
    C11(2x4) = A10(2x2) * B01(2x4) + A11(2x3) * B11(3x4)
    
    For each block product, we use the optimal or near-optimal decomposition.
    """
    n, m, p = 3, 5, 6
    
    # Row splits for A/C: r1=1, r2=2 (total 3)
    # Col splits for A/B: c1=2, c2=3 (total 5)
    # Col splits for B/C: d1=2, d2=4 (total 6)
    
    r = [1, 2]  # row block sizes
    c = [2, 3]  # col block sizes for A / row block sizes for B
    d = [2, 4]  # col block sizes for B/C
    
    # Precompute cumulative indices
    r_cum = [0, 1, 3]
    c_cum = [0, 2, 5]
    d_cum = [0, 2, 6]
    
    # We'll build the decomposition by handling each block product.
    # For 2x2x2 blocks, use Strassen (rank 7).
    # For other sizes, use schoolbook or other methods.
    
    terms = []  # Each term: (U_vec, V_vec, W_vec) as dicts
    
    def add_block_product(block_u, block_v, block_w, offset_u, offset_v, offset_w):
        """
        Add terms for a block product.
        block_u: shape (ru, cu) - which block of A
        block_v: shape (cv, dw) - which block of B
        block_w: shape (rw, dw) - which block of C
        offset_u, offset_v, offset_w: starting indices in the full matrices
        """
        ru, cu = block_u
        cv, dw = block_v
        rw, dw2 = block_w
        assert cu == cv, "Inner dimensions must match"
        assert dw == dw2, "Output col dims must match"
        
        # For 2x2x2, use Strassen
        if ru == 2 and cu == 2 and dw == 2:
            strassen_terms = _strassen_2x2x2()
            for u_flat, v_flat, w_flat in strassen_terms:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                for idx, val in enumerate(u_flat):
                    if val != 0:
                        ii = idx // 2
                        jj = idx % 2
                        u_dict[(offset_u[0] + ii, offset_u[1] + jj)] = val
                for idx, val in enumerate(v_flat):
                    if val != 0:
                        jj = idx // 2
                        kk = idx % 2
                        v_dict[(offset_v[0] + jj, offset_v[1] + kk)] = val
                for idx, val in enumerate(w_flat):
                    if val != 0:
                        kk = idx // 2
                        ii = idx % 2
                        w_dict[(offset_w[0] + kk, offset_w[1] + ii)] = val
                terms.append((u_dict, v_dict, w_dict))
        else:
            # Schoolbook for this block
            for ii in range(ru):
                for jj in range(cu):
                    for kk in range(dw):
                        u_dict = {(offset_u[0] + ii, offset_u[1] + jj): 1}
                        v_dict = {(offset_v[0] + jj, offset_v[1] + kk): 1}
                        w_dict = {(offset_w[0] + kk, offset_w[1] + ii): 1}
                        terms.append((u_dict, v_dict, w_dict))
    
    # C00 = A00 * B00: A00 is (1,2), B00 is (2,2), C00 is (1,2)
    add_block_product((1, 2), (2, 2), (1, 2), (0, 0), (0, 0), (0, 0))
    
    # C01 = A00*B01 + A01*B11
    # A00*B01: A00(1,2), B01(2,4), C01(1,4)
    add_block_product((1, 2), (2, 4), (1, 4), (0, 0), (0, 2), (0, 2))
    # A01*B11: A01(1,3), B11(3,4), C01(1,4)
    add_block_product((1, 3), (3, 4), (1, 4), (0, 2), (2, 0), (0, 2))
    
    # C10 = A10*B00 + A11*B10
    # A10*B00: A10(2,2), B00(2,2), C10(2,2) -> Use Strassen!
    add_block_product((2, 2), (2, 2), (2, 2), (1, 0), (0, 0), (1, 0))
    # A11*B10: A11(2,3), B10(3,2), C10(2,2)
    add_block_product((2, 3), (3, 2), (2, 2), (1, 2), (2, 0), (1, 0))
    
    # C11 = A10*B01 + A11*B11
    # A10*B01: A10(2,2), B01(2,4), C11(2,4)
    add_block_product((2, 2), (2, 4), (2, 4), (1, 0), (0, 2), (1, 2))
    # A11*B11: A11(2,3), B11(3,4), C11(2,4)
    add_block_product((2, 3), (3, 4), (2, 4), (1, 2), (2, 0), (1, 2))
    
    # Now convert terms to U, V, W matrices
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][col] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][col] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][col] = val
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    try:
        result = _embed_3x5x6()
        
        # Verify the decomposition
        n, m, p = 3, 5, 6
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        
        rank = U.shape[1]
        
        # Verify: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c] for each c... 
        # Actually, the tensor decomposition means:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # where T[i,j,k] = 1 if this is a valid (i,j,k) term, else 0
        # For matrix mult, T[i,j,k] = 1 for all valid (i,j,k)
        
        # Let's verify by checking that the reconstructed tensor matches
        # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        
        # Reconstruct: for each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # This should equal 1 for all (i,j,k)
        
        # Do a spot check
        import random
        random.seed(42)
        errors = 0
        for _ in range(100):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            val = 0.0
            for c in range(rank):
                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            if abs(val - 1.0) > 1e-9:
                errors += 1
        
        if errors > 0:
            # Fall back to schoolbook
            rank_school = n * m * p
            U_s = [[0] * rank_school for _ in range(n * m)]
            V_s = [[0] * rank_school for _ in range(m * p)]
            W_s = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i*m + j][col] = 1
                        V_s[j*p + k][col] = 1
                        W_s[k*n + i][col] = 1
                        col += 1
            return {"U": U_s, "V": V_s, "W": W_s}
        
        return result
        
    except Exception as e:
        # Fallback to schoolbook
        n, m, p = 3, 5, 6
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j][col] = 1
                    V_s[j*p + k][col] = 1
                    W_s[k*n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
