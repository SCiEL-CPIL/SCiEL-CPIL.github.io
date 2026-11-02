"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """
    Returns the 7 terms of Strassen's 2x2x2 matrix multiplication.
    Each term is a tuple (U_dict, V_dict, W_dict) where keys are (i,j), (j,k), (k,i).
    """
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2 = (A21+A22)B11
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    # M3 = A11(B21-B22)
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # M4 = A22(B12-B11)
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M5 = (A11+A12)B22
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    # M6 = (A21-A11)(B11+B12)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    # M7 = (A12-A22)(B21+B22)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    return terms

def _build_4x4x4_strassen():
    """
    Builds the rank 49 decomposition for 4x4 @ 4x4 -> 4x4.
    Returns U, V, W as lists of columns (vectors).
    """
    n, m, p = 4, 4, 4
    U_cols = []
    V_cols = []
    W_cols = []
    
    s2_terms = _strassen_2x2_terms()
    
    # Mapping of block (bi, bj) to global indices
    # A is 4x4, split into 2x2 blocks of size 2x2.
    # Block (0,0): rows 0,1 cols 0,1
    # Block (0,1): rows 0,1 cols 2,3
    # Block (1,0): rows 2,3 cols 0,1
    # Block (1,1): rows 2,3 cols 2,3
    
    def get_global_u(bi, bj, li, lj):
        # bi, bj: block index (0 or 1)
        # li, lj: local index within block (0 or 1)
        gi = bi * 2 + li
        gj = bj * 2 + lj
        return gi * m + gj
        
    def get_global_v(bj, bk, lj, lk):
        # bj: block row of B (0 or 1)
        # bk: block col of B (0 or 1)
        # lj: local row in B block
        # lk: local col in B block
        gj = bj * 2 + lj
        gk = bk * 2 + lk
        return gj * p + gk
        
    def get_global_w(bk, bi, lk, li):
        # bk: block col of C (0 or 1)
        # bi: block row of C (0 or 1)
        # lk: local col in C block
        # li: local row in C block
        gk = bk * 2 + lk
        gi = bi * 2 + li
        return gk * n + gi

    # Strassen combinations for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # A blocks: (0,0), (1,1)
    # B blocks: (0,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    # We need to add terms for each M_k.
    # Each M_k is a 2x2x2 matmul.
    # We iterate over the 7 terms of 2x2x2 Strassen.
    
    def add_strassen_block(u_blocks, v_blocks, w_blocks, w_coeffs):
        """
        u_blocks: list of (block_row, block_col) for A
        v_blocks: list of (block_row, block_col) for B
        w_blocks: list of (block_row, block_col) for C
        w_coeffs: list of coefficients for C blocks
        """
        for term in s2_terms:
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            # U part
            for (li, lj), cu in term['U'].items():
                # This term in 2x2x2 corresponds to local indices li, lj
                # We need to map this to the global A blocks specified in u_blocks
                # Wait, the 2x2x2 Strassen term U has entries for (0,0), (1,1) etc.
                # These correspond to the local 2x2 matrix.
                # The local 2x2 matrix is formed by summing/differencing the global blocks.
                # So for a specific global block (bi, bj), we add cu * U_local[li, lj] to U_global[bi*2+li, bj*2+lj].
                
                # Actually, it's easier to think:
                # The "A" matrix in the 2x2x2 sub-problem is a linear combination of global blocks.
                # A_sub = sum_{(bi,bj) in u_blocks} coeff * A_{bi,bj}
                # The U vector for the sub-problem has entries for local (li, lj).
                # So U_global[bi*2+li, bj*2+lj] += cu * coeff.
                
                # In our case, the "coeff" is 1 for the blocks involved in the sum/diff.
                # But we must be careful with signs.
                # The _strassen_2x2_terms() already includes the signs in U, V, W.
                # So for M1 = (A00+A11)(B00+B11):
                # A_sub = A00 + A11.
                # The U vector for the sub-problem has 1 at (0,0) and (1,1) (for M1 term 1).
                # So we add 1 * U_sub[li, lj] to U_global for block 00 and 1 * U_sub[li, lj] to U_global for block 11.
                
                for (bi, bj) in u_blocks:
                    gi = bi * 2 + li
                    gj = bj * 2 + lj
                    u_vec[gi*m + gj] += cu
                    
            # V part
            for (lj, lk), cv in term['V'].items():
                for (bj, bk) in v_blocks:
                    gj = bj * 2 + lj
                    gk = bk * 2 + lk
                    v_vec[gj*p + gk] += cv
                    
            # W part
            for (lk, li), cw in term['W'].items():
                for idx, (bi, bk) in enumerate(w_blocks):
                    gi = bi * 2 + li
                    gk = bk * 2 + lk
                    w_vec[gk*n + gi] += cw * w_coeffs[idx]
                    
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)

    # M1 = (A00+A11)(B00+B11)
    # A: 00, 11. B: 00, 11.
    # C: 00, 01, 10, 11 with coeffs 1, 1, 1, 1? No.
    # C00 = M1 + ...
    # C01 = ...
    # The W vector in the 2x2x2 term distributes to the 4 C blocks.
    # For M1, W_sub has 1 at (0,0) and (1,1).
    # This means C_sub[0,0] and C_sub[1,1] get the contribution.
    # C_sub is 2x2.
    # C00 (global) = C_sub[0,0]
    # C01 (global) = C_sub[0,1]
    # C10 (global) = C_sub[1,0]
    # C11 (global) = C_sub[1,1]
    
    # So for M1:
    # W_sub[0,0] -> C00. Coeff 1.
    # W_sub[1,1] -> C11. Coeff 1.
    # W_sub[0,1] -> C01. Coeff 0 (for M1).
    # W_sub[1,0] -> C10. Coeff 0 (for M1).
    
    # However, the _strassen_2x2_terms() W dict contains the distribution to C_sub entries.
    # We need to map C_sub entries to global C blocks.
    # C_sub[li, lk] corresponds to global block (bi, bk) where bi=li, bk=lk?
    # No, C_sub is the result of the 2x2x2 multiplication.
    # The result C_sub is a 2x2 matrix.
    # This 2x2 matrix IS the block C_{bi, bk} of the global C?
    # No. The 2x2x2 multiplication computes C_sub = A_sub * B_sub.
    # A_sub is 2x2, B_sub is 2x2.
    # The global C is 4x4.
    # The blocks of C are 2x2.
    # So C_sub corresponds to ONE block of C?
    # No. In Strassen for 4x4, we compute 7 products of 2x2 matrices.
    # Each product is a 2x2 matrix.
    # These 2x2 matrices are linear combinations of the 4 blocks of C.
    # Specifically:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So, for each M_k, we add its contribution to the specific C blocks.
    # The W vector from the 2x2x2 term tells us which entries of C_sub are non-zero.
    # C_sub[li, lk] is the (li, lk) entry of the 2x2 result.
    # This entry contributes to the global C block C_{bi, bk} at local position (li, lk)?
    # No. The entire 2x2 matrix M_k IS the value to be added to the C blocks.
    # So M_k[0,0] is added to C00[0,0], C01[0,0]? No.
    # M_k is a 2x2 matrix.
    # C00 is a 2x2 matrix.
    # C00 += coeff * M_k.
    # So every entry of M_k is added to the corresponding entry of C00.
    
    # Therefore, for a given M_k:
    # We need to add the W vector (which represents M_k) to the global W for the specified C blocks.
    # The W vector has indices (lk, li) corresponding to C_sub[lk, li]?
    # In _strassen_2x2_terms, W is indexed by (k, i) where k is col, i is row?
    # Let's check _strassen_2x2_terms.
    # W: {(0,0): 1, (1,1): 1}.
    # This means C_sub[0,0] and C_sub[1,1] are affected.
    # Global C blocks:
    # C00 corresponds to C_sub[0,0]? No.
    # The 2x2 matrix M_k is added to C00, C01, etc.
    # So if M_k has a non-zero at [0,0], it adds to C00[0,0], C01[0,0]?
    # No. C00 is a block. C00[0,0] is the top-left element of the top-left block.
    # So yes, M_k[0,0] contributes to C00[0,0], C01[0,0], C10[0,0], C11[0,0] depending on coefficients.
    
    # So, for each C block (bi, bk) with coefficient c:
    # We add c * W_sub[lk, li] to W_global[bk*2+lk, bi*2+li].
    
    # Let's re-implement add_strassen_block to handle this correctly.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term_global(u_blocks, v_blocks, c_blocks):
        """
        u_blocks: list of (bi, bj)
        v_blocks: list of (bj, bk)
        c_blocks: list of (bi, bk, coeff)
        """
        for term in s2_terms:
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            for (li, lj), cu in term['U'].items():
                for (bi, bj) in u_blocks:
                    gi = bi * 2 + li
                    gj = bj * 2 + lj
                    u_vec[gi*m + gj] += cu
                    
            for (lj, lk), cv in term['V'].items():
                for (bj, bk) in v_blocks:
                    gj = bj * 2 + lj
                    gk = bk * 2 + lk
                    v_vec[gj*p + gk] += cv
                    
            for (lk, li), cw in term['W'].items():
                for (bi, bk, coeff) in c_blocks:
                    gi = bi * 2 + li
                    gk = bk * 2 + lk
                    w_vec[gk*n + gi] += cw * coeff
                    
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)

    # M1 = (A00+A11)(B00+B11)
    add_term_global([(0,0), (1,1)], [(0,0), (1,1)], [(0,0,1), (1,1,1)])
    # M2 = (A10+A11)B00
    add_term_global([(1,0), (1,1)], [(0,0)], [(1,0,1), (1,1,-1)])
    # M3 = A00(B10-B11)
    add_term_global([(0,0)], [(1,0), (1,1)], [(0,1,1), (1,1,1)])
    # M4 = A11(B01-B00)
    add_term_global([(1,1)], [(0,0), (0,1)], [(0,0,1), (1,0,1)])
    # M5 = (A00+A01)B11
    add_term_global([(0,0), (0,1)], [(1,1)], [(0,0,-1), (0,1,1)])
    # M6 = (A10-A00)(B00+B01)
    add_term_global([(1,0), (0,0)], [(0,0), (0,1)], [(1,1,1)])
    # M7 = (A01-A11)(B10+B11)
    add_term_global([(0,1), (1,1)], [(1,0), (1,1)], [(0,0,1)])
    
    return U_cols, V_cols, W_cols

def _build_4x4x5():
    n, m, p = DIMENSIONS
    U_cols, V_cols, W_cols = _build_4x4x4_strassen()
    
    # Add rank 1 term for the 5th column (index 4)
    u_vec = np.ones(n*m)
    v_vec = np.zeros(m*p)
    w_vec = np.zeros(p*n)
    
    # V: j in 0..3, k=4
    for j in range(m):
        v_vec[j*p + 4] = 1
        
    # W: k=4, i in 0..3
    for i in range(n):
        w_vec[4*n + i] = 1
        
    U_cols.append(u_vec)
    V_cols.append(v_vec)
    W_cols.append(w_vec)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    
    try:
        result = _build_4x4x5()
        rank = len(result["U"][0])
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception as e:
        print(f"Error in new decomposition: {e}")
        
    # Fallback to schoolbook (rank 80)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
