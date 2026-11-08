import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Recursive Strassen: 4x4 -> 7 block products of 2x2, each 2x2 -> 7 scalar products
    # Total: 7 * 7 = 49 rank-1 terms
    
    # For each of the 7 block-level Strassen terms, we have a 2x2 x 2x2 product
    # Each such product is decomposed into 7 scalar multiplications via Strassen
    
    # Block-level Strassen formulas:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # For each Pi = Li * Ri where Li, Ri are 2x2 matrices with entries being
    # linear combinations of A/B elements, apply Strassen to get 7 scalar products.
    
    # Strassen sub-product contribution to 2x2 result:
    # M1=(a+d)(e+h) -> C11:+1, C22:+1
    # M2=(c+d)e     -> C21:+1, C22:-1
    # M3=a(f-h)     -> C12:+1, C22:+1
    # M4=d(g-e)     -> C11:+1, C21:+1
    # M5=(a+b)h     -> C11:-1, C12:+1
    # M6=(c-a)(e+f) -> C22:+1
    # M7=(b-d)(g+h) -> C11:+1
    
    # Define the 7 block terms as (L_matrix, R_matrix, contribution_map)
    # L_matrix[r][c] = dict of {(global_row, global_col): coeff} for A
    # R_matrix[r][c] = dict of {(global_row, global_col): coeff} for B
    # contribution_map[(pr,pc)] = list of (out_row, out_col, coeff)
    
    def a_idx(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def b_idx(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def c_idx(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    # Build the 7 terms
    terms = []
    
    # Term 0: P1 = (A11+A22)(B11+B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L0[r][c] = {a_idx(0,0,r,c): 1, a_idx(1,1,r,c): 1}
            R0[r][c] = {b_idx(0,0,r,c): 1, b_idx(1,1,r,c): 1}
    contrib0 = {(r,c): [(c_idx(0,0,r,c)[0], c_idx(0,0,r,c)[1], 1),
                         (c_idx(1,1,r,c)[0], c_idx(1,1,r,c)[1], 1)]
                for r in range(2) for c in range(2)}
    terms.append((L0, R0, contrib0))
    
    # Term 1: P2 = (A21+A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L1[r][c] = {a_idx(1,0,r,c): 1, a_idx(1,1,r,c): 1}
            R1[r][c] = {b_idx(0,0,r,c): 1}
    contrib1 = {(r,c): [(c_idx(1,0,r,c)[0], c_idx(1,0,r,c)[1], 1),
                         (c_idx(1,1,r,c)[0], c_idx(1,1,r,c)[1], -1)]
                for r in range(2) for c in range(2)}
    terms.append((L1, R1, contrib1))
    
    # Term 2: P3 = A11(B12-B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L2[r][c] = {a_idx(0,0,r,c): 1}
            R2[r][c] = {b_idx(0,1,r,c): 1, b_idx(1,1,r,c): -1}
    contrib2 = {(r,c): [(c_idx(0,1,r,c)[0], c_idx(0,1,r,c)[1], 1),
                         (c_idx(1,1,r,c)[0], c_idx(1,1,r,c)[1], 1)]
                for r in range(2) for c in range(2)}
    terms.append((L2, R2, contrib2))
    
    # Term 3: P4 = A22(B21-B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L3[r][c] = {a_idx(1,1,r,c): 1}
            R3[r][c] = {b_idx(1,0,r,c): 1, b_idx(0,0,r,c): -1}
    contrib3 = {(r,c): [(c_idx(0,0,r,c)[0], c_idx(0,0,r,c)[1], 1),
                         (c_idx(1,0,r,c)[0], c_idx(1,0,r,c)[1], 1)]
                for r in range(2) for c in range(2)}
    terms.append((L3, R3, contrib3))
    
    # Term 4: P5 = (A11+A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L4[r][c] = {a_idx(0,0,r,c): 1, a_idx(0,1,r,c): 1}
            R4[r][c] = {b_idx(1,1,r,c): 1}
    contrib4 = {(r,c): [(c_idx(0,0,r,c)[0], c_idx(0,0,r,c)[1], -1),
                         (c_idx(0,1,r,c)[0], c_idx(0,1,r,c)[1], 1)]
                for r in range(2) for c in range(2)}
    terms.append((L4, R4, contrib4))
    
    # Term 5: P6 = (A21-A11)(B11+B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L5[r][c] = {a_idx(1,0,r,c): 1, a_idx(0,0,r,c): -1}
            R5[r][c] = {b_idx(0,0,r,c): 1, b_idx(0,1,r,c): 1}
    contrib5 = {(r,c): [(c_idx(1,1,r,c)[0], c_idx(1,1,r,c)[1], 1)]
                for r in range(2) for c in range(2)}
    terms.append((L5, R5, contrib5))
    
    # Term 6: P7 = (A12-A22)(B21+B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L6[r][c] = {a_idx(0,1,r,c): 1, a_idx(1,1,r,c): -1}
            R6[r][c] = {b_idx(1,0,r,c): 1, b_idx(1,1,r,c): 1}
    contrib6 = {(r,c): [(c_idx(0,0,r,c)[0], c_idx(0,0,r,c)[1], 1)]
                for r in range(2) for c in range(2)}
    terms.append((L6, R6, contrib6))
    
    # Strassen sub-product contributions to 2x2 output
    sub_contrib = [
        [(0,0,1), (1,1,1)],    # M1
        [(1,0,1), (1,1,-1)],   # M2
        [(0,1,1), (1,1,1)],    # M3
        [(0,0,1), (1,0,1)],    # M4
        [(0,0,-1), (0,1,1)],   # M5
        [(1,1,1)],             # M6
        [(0,0,1)]              # M7
    ]
    
    # For each term t and sub-product k, compute the linear combos and add to U,V,W
    for t in range(7):
        L_t, R_t, contrib_t = terms[t]
        
        a = L_t[0][0]; b = L_t[0][1]; c = L_t[1][0]; d = L_t[1][1]
        e = R_t[0][0]; f = R_t[0][1]; g = R_t[1][0]; h = R_t[1][1]
        
        # Build the 7 sub-products (x, y) pairs
        def add_dicts(d1, d2, sign=1):
            result = dict(d1)
            for key, val in d2.items():
                result[key] = result.get(key, 0) + sign * val
            return result
        
        sub_products = [
            (add_dicts(a, d), add_dicts(e, h)),          # M1
            (add_dicts(c, d), e),                         # M2
            (a, add_dicts(f, h, -1)),                     # M3
            (d, add_dicts(g, e, -1)),                     # M4
            (add_dicts(a, b), h),                         # M5
            (add_dicts(c, a, -1), add_dicts(e, f)),       # M6
            (add_dicts(b, d, -1), add_dicts(g, h))        # M7
        ]
        
        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k
            
            # Add to U
            for (row, col), coeff in x_k.items():
                u_idx = row * m + col
                U[u_idx][rank_idx] += coeff
            
            # Add to V
            for (row, col), coeff in y_k.items():
                v_idx = row * p + col
                V[v_idx][rank_idx] += coeff
            
            # Add to W
            for (pr, pc, sub_coeff) in sub_contrib[k]:
                if (pr, pc) in contrib_t:
                    for (out_row, out_col, block_coeff) in contrib_t[(pr, pc)]:
                        w_idx = out_col * n + out_row  # trace convention: W[k*n+i, r]
                        W[w_idx][rank_idx] += sub_coeff * block_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
