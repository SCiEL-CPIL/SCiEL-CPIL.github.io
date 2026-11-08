"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank 49 for 4x4x4 part (Strassen)
    # Rank 48 for 4x4x3 part (Schoolbook)
    # Total rank 97
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_row, v_row, w_row, coeff):
        # u_row, v_row, w_row are lists of (index, coeff)
        for idx, c_u in u_row:
            U[idx][col] += c_u * coeff
        for idx, c_v in v_row:
            V[idx][col] += c_v * coeff
        for idx, c_w in w_row:
            W[idx][col] += c_w * coeff

    col = 0
    
    # Part 1: 4x4x4 using Strassen
    # Strassen terms for 2x2x2
    # u: [(idx, coeff)], v: [(idx, coeff)], w: [(idx, coeff)]
    # Local indices 0..3 for 2x2 matrix (row*2+col)
    strassen_2x2 = [
        # 1: (a+d)(e+h)
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)]
        },
        # 2: (c+d)e
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [(2, 1), (3, -1)]
        },
        # 3: a(f-h)
        {
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [(1, 1), (3, 1)]
        },
        # 4: d(g-e)
        {
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [(0, 1), (2, 1)]
        },
        # 5: (a+b)h
        {
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [(0, -1), (1, 1)]
        },
        # 6: (c-a)(e+f)
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [(3, 1)]
        },
        # 7: (b-d)(g+h)
        {
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [(0, 1)]
        }
    ]
    
    # Block multipliers for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Map block index to global indices
    # Block (bi, bj) corresponds to:
    # A: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # B: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # C: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1 (Note: C is p x n, so "row" is output col k, "col" is input row i)
    # Wait, C is p x n. In tensor decomposition, W is p*n x rank.
    # W[k*n + i, c].
    # So C[k, i].
    # Block C(bi, bj) has k in 2*bi..2*bi+1, i in 2*bj..2*bj+1.
    
    # For each of the 7 block multipliers M_k:
    # We determine which global C blocks it contributes to and with what sign.
    # Then for each of the 7 Strassen sub-terms of M_k:
    # We determine the local C entries it contributes to.
    # Combine to get global C entries.
    
    # M_k contributions to C blocks:
    # M1: C00 (+1), C11 (+1)
    # M2: C10 (+1), C11 (-1)
    # M3: C01 (+1), C11 (+1)
    # M4: C00 (+1), C10 (+1)
    # M5: C00 (-1), C01 (+1)
    # M6: C11 (+1)
    # M7: C00 (+1)
    
    m_contribs = [
        [(0, 0, 1), (1, 1, 1)], # M1: C00, C11
        [(1, 0, 1), (1, 1, -1)], # M2: C10, C11
        [(0, 1, 1), (1, 1, 1)], # M3: C01, C11
        [(0, 0, 1), (1, 0, 1)], # M4: C00, C10
        [(0, 0, -1), (0, 1, 1)], # M5: C00, C01
        [(1, 1, 1)], # M6: C11
        [(0, 0, 1)] # M7: C00
    ]
    
    # A block contributions for M_k
    # M1: A00 (+1), A11 (+1)
    # M2: A10 (+1), A11 (+1)
    # M3: A00 (+1)
    # M4: A11 (+1)
    # M5: A00 (+1), A01 (+1)
    # M6: A10 (+1), A00 (-1)
    # M7: A01 (+1), A11 (-1)
    
    a_contribs = [
        [(0, 0, 1), (1, 1, 1)], # M1
        [(1, 0, 1), (1, 1, 1)], # M2
        [(0, 0, 1)], # M3
        [(1, 1, 1)], # M4
        [(0, 0, 1), (0, 1, 1)], # M5
        [(1, 0, 1), (0, 0, -1)], # M6
        [(0, 1, 1), (1, 1, -1)] # M7
    ]
    
    # B block contributions for M_k
    # M1: B00 (+1), B11 (+1)
    # M2: B00 (+1)
    # M3: B01 (+1), B11 (-1)
    # M4: B10 (+1), B00 (-1)
    # M5: B11 (+1)
    # M6: B00 (+1), B01 (+1)
    # M7: B10 (+1), B11 (+1)
    
    b_contribs = [
        [(0, 0, 1), (1, 1, 1)], # M1
        [(0, 0, 1)], # M2
        [(0, 1, 1), (1, 1, -1)], # M3
        [(1, 0, 1), (0, 0, -1)], # M4
        [(1, 1, 1)], # M5
        [(0, 0, 1), (0, 1, 1)], # M6
        [(1, 0, 1), (1, 1, 1)] # M7
    ]
    
    for k in range(7):
        for t in range(7):
            # Check deadline
            if time.time() > deadline:
                return {"U": U, "V": "W": W} # Invalid, but better than timeout? No, just return best found.
                # Actually, if we haven't finished, we should return what we have?
                # The task requires exact decomposition. If incomplete, it's invalid (score 0).
                # So we must ensure we finish.
                # 97 terms is small. This loop is 49 iterations. It will be fast.
            
            s_term = strassen_2x2[t]
            
            # Build U, V, W vectors for this term
            u_vec = {}
            v_vec = {}
            w_vec = {}
            
            # Combine A contributions
            for a_bi, a_bj, a_sign in a_contribs[k]:
                # Local indices in 2x2 block
                for local_idx, local_coeff in s_term['u']:
                    # Map local_idx to global A index
                    # local_idx is 0..3 in 2x2 block
                    # row = local_idx // 2, col = local_idx % 2
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    g_row = 2 * a_bi + l_row
                    g_col = 2 * a_bj + l_col
                    g_idx = g_row * 4 + g_col
                    u_vec[g_idx] = u_vec.get(g_idx, 0) + a_sign * local_coeff
            
            # Combine B contributions
            for b_bi, b_bj, b_sign in b_contribs[k]:
                for local_idx, local_coeff in s_term['v']:
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    g_row = 2 * b_bi + l_row
                    g_col = 2 * b_bj + l_col
                    g_idx = g_row * 7 + g_col
                    v_vec[g_idx] = v_vec.get(g_idx, 0) + b_sign * local_coeff
            
            # Combine C contributions
            for c_bi, c_bj, c_sign in m_contribs[k]:
                for local_idx, local_coeff in s_term['w']:
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    # C is p x n. Block C(bi, bj) has k in 2*bi.., i in 2*bj..
                    g_k = 2 * c_bi + l_row
                    g_i = 2 * c_bj + l_col
                    g_idx = g_k * 4 + g_i
                    w_vec[g_idx] = w_vec.get(g_idx, 0) + c_sign * local_coeff
            
            # Add to U, V, W
            for idx, coeff in u_vec.items():
                if coeff != 0:
                    U[idx][col] = coeff
            for idx, coeff in v_vec.items():
                if coeff != 0:
                    V[idx][col] = coeff
            for idx, coeff in w_vec.items():
                if coeff != 0:
                    W[idx][col] = coeff
            
            col += 1
            
    # Part 2: 4x4x3 using Schoolbook
    # C[i, 4+k] = sum_j A[i, j] B[j, 4+k]
    # k in 0..2
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u_idx = i * 4 + j
                v_idx = j * 7 + (4 + k)
                w_idx = (4 + k) * 4 + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
