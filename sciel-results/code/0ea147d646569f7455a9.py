import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 4
    N_A = n * m  # 16
    N_B = m * p  # 16
    N_C = p * n  # 16
    
    # Initialize factor matrices
    rank = 49
    U = np.zeros((N_A, rank))
    V = np.zeros((N_B, rank))
    W = np.zeros((N_C, rank))
    
    # Helper to get indices
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i # Trace convention: W[k*n+i]
    
    # The 7 Strassen products for 2x2 matrices
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For 4x4, we split into 2x2 blocks of size 2x2.
    # Let A = [[A11, A12], [A21, A22]] where each is 2x2.
    # We compute 7 products M1..M7, each of which is a 2x2 matrix product.
    # Each 2x2 product is computed with 7 scalar multiplications.
    # Total 49 scalar multiplications.
    
    # We define the terms recursively.
    # Term (t, k) where t is high-level Strassen term (0..6) and k is low-level Strassen term (0..6).
    
    # High-level definitions:
    # t=0: M1 = (A11+A22)(B11+B22)
    # t=1: M2 = (A21+A22)B11
    # t=2: M3 = A11(B12-B22)
    # t=3: M4 = A22(B21-B11)
    # t=4: M5 = (A11+A12)B22
    # t=5: M6 = (A21-A11)(B11+B12)
    # t=6: M7 = (A12-A22)(B21+B22)
    
    # For each high-level term t, we have factors X_t and Y_t which are 2x2 matrices.
    # X_t = sum of some A-blocks, Y_t = sum of some B-blocks.
    # Then we compute M_t = X_t * Y_t using low-level Strassen.
    # The low-level Strassen terms k=0..6 correspond to the 7 multiplications for 2x2.
    # Let X_t = [[x00, x01], [x10, x11]] and Y_t = [[y00, y01], [y10, y11]].
    # Low-level terms:
    # k=0: (x00+x11)(y00+y11)
    # k=1: (x10+x11)y00
    # k=2: x00(y01-y11)
    # k=3: x11(y10-y00)
    # k=4: (x00+x01)y11
    # k=5: (x10-x00)(y00+y01)
    # k=6: (x01-x11)(y10+y11)
    
    # We need to express x_ij and y_ij in terms of the global A and B entries.
    # A-blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # Let's define the coefficients for each A entry (i,j) in each x_ij.
    # And for each B entry (j,k) in each y_ij.
    
    # This is getting complicated to hardcode manually without errors.
    # I will stick to the logic in the original code but clean it up.
    # The original code seems correct. I will just ensure it's robust.
    
    # Re-implementing the original logic with clearer variable names.
    
    def get_U_index(row, col):
        return row * m + col

    def get_V_index(row, col):
        return row * p + col

    def get_W_index(row, col):
        # row is i, col is k. W index is k*n + i.
        return col * n + row

    # a_elem(bi, bj, r, c) returns global (row, col) in A for block (bi,bj) local (r,c)
    # bi, bj are block indices (0 or 1).
    # r, c are local indices (0 or 1).
    # Global row = 2*bi + r
    # Global col = 2*bj + c
    def a_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def b_elem(bi, bj, r, c):
        # B blocks: B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
        # Row in B is j, Col in B is k.
        # Block (bi, bj) -> Global row = 2*bi + r, Global col = 2*bj + c
        return (2*bi + r, 2*bj + c)
    
    def c_elem(bi, bj, r, c):
        # C blocks: C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
        # Row in C is i, Col in C is k.
        # Block (bi, bj) -> Global row = 2*bi + r, Global col = 2*bj + c
        return (2*bi + r, 2*bj + c)

    # Define the 7 high-level terms
    # Each term is defined by:
    # X: dict of (local_row, local_col) -> coeff for A-blocks? No, X is a 2x2 matrix of A-blocks.
    # It's better to define X and Y as 2x2 matrices of dictionaries mapping (global_row, global_col) to coeff.
    
    # Term 0: P1 = (A11 + A22)(B11 + B22)
    # X0 = [[A11, 0], [0, A22]] ? No, X0 is the left factor.
    # X0 = A11 + A22 (as a 2x2 block matrix? No, A11 and A22 are 2x2 matrices).
    # In Strassen, X is a 2x2 matrix where each entry is a sum of A-blocks.
    # For P1: X = [[A11, 0], [0, A22]] is not right.
    # P1 = (A11 + A22)(B11 + B22).
    # Let X = [[A11, 0], [0, A22]]? No.
    # The formula is (A11+A22) * (B11+B22).
    # This is not a standard 2x2 block multiplication structure like C = AB.
    # It's a product of two specific combinations.
    # So X is a 2x2 matrix? No, X is just the matrix (A11+A22).
    # Wait, the recursive step treats the 4x4 matrix as 2x2 blocks of 2x2 matrices.
    # The multiplication is C = A B.
    # Strassen computes 7 products of 2x2 matrices.
    # M1 = (A11 + A22)(B11 + B22).
    # Here, (A11 + A22) is a 2x2 matrix. (B11 + B22) is a 2x2 matrix.
    # So X1 = A11 + A22. Y1 = B11 + B22.
    # Then we compute M1 = X1 Y1 using the base case Strassen (7 multiplications).
    
    # So for each high-level term t, we have X_t and Y_t which are 2x2 matrices.
    # We need to express X_t and Y_t in terms of the scalar entries of A and B.
    
    # Let's define X_t and Y_t as 2x2 arrays of dictionaries.
    # X_t[r][c] is a dict mapping (global_row, global_col) in A to coeff.
    # Y_t[r][c] is a dict mapping (global_row, global_col) in B to coeff.
    
    # Term 0: M1 = (A11 + A22)(B11 + B22)
    # X0 = A11 + A22
    # A11 is block (0,0), A22 is block (1,1).
    X0 = [[{}, {}], [{}, {}]]
    # Add A11 to X0
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(0, 0, r, c)
            X0[r][c][(gr, gc)] = 1
    # Add A22 to X0
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(1, 1, r, c)
            X0[r][c][(gr, gc)] = 1
            
    Y0 = [[{}, {}], [{}, {}]]
    # Add B11 to Y0
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(0, 0, r, c)
            Y0[r][c][(gr, gc)] = 1
    # Add B22 to Y0
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(1, 1, r, c)
            Y0[r][c][(gr, gc)] = 1
            
    # Contribution of M1 to C: C11 += M1, C22 += M1
    # C11 is block (0,0), C22 is block (1,1).
    contrib0 = {
        (0, 0): 1, # C11 += M1
        (1, 1): 1  # C22 += M1
    }
    
    # Term 1: M2 = (A21 + A22)B11
    X1 = [[{}, {}], [{}, {}]]
    # Add A21
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(1, 0, r, c)
            X1[r][c][(gr, gc)] = 1
    # Add A22
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(1, 1, r, c)
            X1[r][c][(gr, gc)] = 1
            
    Y1 = [[{}, {}], [{}, {}]]
    # Add B11
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(0, 0, r, c)
            Y1[r][c][(gr, gc)] = 1
            
    # Contribution of M2 to C: C21 += M2, C22 -= M2
    contrib1 = {
        (1, 0): 1, # C21 += M2
        (1, 1): -1 # C22 -= M2
    }
    
    # Term 2: M3 = A11(B12 - B22)
    X2 = [[{}, {}], [{}, {}]]
    # Add A11
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(0, 0, r, c)
            X2[r][c][(gr, gc)] = 1
            
    Y2 = [[{}, {}], [{}, {}]]
    # Add B12
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(0, 1, r, c)
            Y2[r][c][(gr, gc)] = 1
    # Subtract B22
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(1, 1, r, c)
            Y2[r][c][(gr, gc)] = -1
            
    # Contribution of M3 to C: C12 += M3, C22 += M3
    contrib2 = {
        (0, 1): 1, # C12 += M3
        (1, 1): 1  # C22 += M3
    }
    
    # Term 3: M4 = A22(B21 - B11)
    X3 = [[{}, {}], [{}, {}]]
    # Add A22
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(1, 1, r, c)
            X3[r][c][(gr, gc)] = 1
            
    Y3 = [[{}, {}], [{}, {}]]
    # Add B21
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(1, 0, r, c)
            Y3[r][c][(gr, gc)] = 1
    # Subtract B11
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(0, 0, r, c)
            Y3[r][c][(gr, gc)] = -1
            
    # Contribution of M4 to C: C11 += M4, C21 += M4
    contrib3 = {
        (0, 0): 1, # C11 += M4
        (1, 0): 1  # C21 += M4
    }
    
    # Term 4: M5 = (A11 + A12)B22
    X4 = [[{}, {}], [{}, {}]]
    # Add A11
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(0, 0, r, c)
            X4[r][c][(gr, gc)] = 1
    # Add A12
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(0, 1, r, c)
            X4[r][c][(gr, gc)] = 1
            
    Y4 = [[{}, {}], [{}, {}]]
    # Add B22
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(1, 1, r, c)
            Y4[r][c][(gr, gc)] = 1
            
    # Contribution of M5 to C: C11 -= M5, C12 += M5
    contrib4 = {
        (0, 0): -1, # C11 -= M5
        (0, 1): 1   # C12 += M5
    }
    
    # Term 5: M6 = (A21 - A11)(B11 + B12)
    X5 = [[{}, {}], [{}, {}]]
    # Add A21
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(1, 0, r, c)
            X5[r][c][(gr, gc)] = 1
    # Subtract A11
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(0, 0, r, c)
            X5[r][c][(gr, gc)] = -1
            
    Y5 = [[{}, {}], [{}, {}]]
    # Add B11
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(0, 0, r, c)
            Y5[r][c][(gr, gc)] = 1
    # Add B12
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(0, 1, r, c)
            Y5[r][c][(gr, gc)] = 1
            
    # Contribution of M6 to C: C22 += M6
    contrib5 = {
        (1, 1): 1   # C22 += M6
    }
    
    # Term 6: M7 = (A12 - A22)(B21 + B22)
    X6 = [[{}, {}], [{}, {}]]
    # Add A12
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(0, 1, r, c)
            X6[r][c][(gr, gc)] = 1
    # Subtract A22
    for r in range(2):
        for c in range(2):
            gr, gc = a_elem(1, 1, r, c)
            X6[r][c][(gr, gc)] = -1
            
    Y6 = [[{}, {}], [{}, {}]]
    # Add B21
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(1, 0, r, c)
            Y6[r][c][(gr, gc)] = 1
    # Add B22
    for r in range(2):
        for c in range(2):
            gr, gc = b_elem(1, 1, r, c)
            Y6[r][c][(gr, gc)] = 1
            
    # Contribution of M7 to C: C11 += M7
    contrib6 = {
        (0, 0): 1   # C11 += M7
    }
    
    X_terms = [X0, X1, X2, X3, X4, X5, X6]
    Y_terms = [Y0, Y1, Y2, Y3, Y4, Y5, Y6]
    contrib_terms = [contrib0, contrib1, contrib2, contrib3, contrib4, contrib5, contrib6]
    
    # Low-level Strassen definitions for 2x2 matrices
    # Given X (2x2) and Y (2x2), compute 7 products.
    # We need to fill U, V, W for each of these 7 products.
    # Each product is a rank-1 term in the final decomposition?
    # No, each of the 7 low-level products is itself a 2x2 matrix multiplication,
    # which is decomposed into 7 rank-1 tensors?
    # No. The low-level Strassen computes the 7 scalar multiplications required for a 2x2 product.
    # Each scalar multiplication corresponds to one rank-1 term in the tensor decomposition.
    
    # So for each high-level term t (0..6), we have X_t and Y_t (2x2 matrices of dicts).
    # We apply the 7 low-level Strassen formulas to X_t and Y_t.
    # This generates 7 rank-1 terms for each t.
    # Total 49 terms.
    
    # Low-level formulas for X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # The 7 products are:
    # P1 = (x00 + x11)(y00 + y11)
    # P2 = (x10 + x11)y00
    # P3 = x00(y01 - y11)
    # P4 = x11(y10 - y00)
    # P5 = (x00 + x01)y11
    # P6 = (x10 - x00)(y00 + y01)
    # P7 = (x01 - x11)(y10 + y11)
    
    # We need to sum the dictionaries to get the combined factors for each rank-1 term.
    
    def add_dicts(d1, d2):
        res = dict(d1)
        for k, v in d2.items():
            res[k] = res.get(k, 0) + v
        return res
        
    def sub_dicts(d1, d2):
        res = dict(d1)
        for k, v in d2.items():
            res[k] = res.get(k, 0) - v
        return res
        
    def neg_dict(d1):
        return {k: -v for k, v in d1.items()}
        
    # For each high-level term t
    for t in range(7):
        X = X_terms[t]
        Y = Y_terms[t]
        contrib = contrib_terms[t]
        
        # Extract x_ij, y_ij
        x00, x01, x10, x11 = X[0][0], X[0][1], X[1][0], X[1][1]
        y00, y01, y10, y11 = Y[0][0], Y[0][1], Y[1][0], Y[1][1]
        
        # Define the 7 low-level factor pairs (U_factor, V_factor)
        # Each is a pair of dicts.
        low_terms = []
        
        # 1: (x00+x11)(y00+y11)
        low_terms.append((add_dicts(x00, x11), add_dicts(y00, y11)))
        # 2: (x10+x11)y00
        low_terms.append((add_dicts(x10, x11), y00))
        # 3: x00(y01-y11)
        low_terms.append((x00, sub_dicts(y01, y11)))
        # 4: x11(y10-y00)
        low_terms.append((x11, sub_dicts(y10, y00)))
        # 5: (x00+x01)y11
        low_terms.append((add_dicts(x00, x01), y11))
        # 6: (x10-x00)(y00+y01)
        low_terms.append((sub_dicts(x10, x00), add_dicts(y00, y01)))
        # 7: (x01-x11)(y10+y11)
        low_terms.append((sub_dicts(x01, x11), add_dicts(y10, y11)))
        
        # The contribution of the resulting 2x2 matrix M_t to C is given by contrib.
        # M_t = [[M00, M01], [M10, M11]]
        # C_block(pr, pc) += coeff * M_t[pr, pc]
        
        # We need to map the entries of M_t to the global C indices.
        # M_t is computed by the 7 low-level products.
        # How do we distribute the 7 products to the 4 entries of M_t?
        # This is the tricky part. The low-level Strassen computes the 7 scalar products,
        # and then combines them to form the 4 entries of the 2x2 result.
        
        # Let's recall the 2x2 Strassen reconstruction:
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # So for each low-level term k (0..6), it contributes to specific entries of M_t.
        # And each entry of M_t contributes to specific blocks of C.
        
        recon_map = {
            0: [(0,0,1), (1,1,1)],   # P1 -> C11, C22
            1: [(1,0,1), (1,1,-1)],  # P2 -> C21, C22
            2: [(0,1,1), (1,1,1)],   # P3 -> C12, C22
            3: [(0,0,1), (1,0,1)],   # P4 -> C11, C21
            4: [(0,0,-1), (0,1,1)],  # P5 -> C11, C12
            5: [(1,1,1)],            # P6 -> C22
            6: [(0,0,1)]             # P7 -> C11
        }
        
        for k in range(7):
            u_dict, v_dict = low_terms[k]
            rank_idx = t * 7 + k
            
            # Fill U and V
            for (gr, gc), coeff in u_dict.items():
                u_idx = get_U_index(gr, gc)
                U[u_idx, rank_idx] += coeff
                
            for (gr, gc), coeff in v_dict.items():
                v_idx = get_V_index(gr, gc)
                V[v_idx, rank_idx] += coeff
                
            # Fill W
            # For each entry (mr, mc) of M_t that this term contributes to:
            for (mr, mc, m_coeff) in recon_map[k]:
                # M_t[mr, mc] contributes to C blocks via 'contrib'
                # M_t is the product X_t Y_t.
                # The entry M_t[mr, mc] is added to C blocks.
                # The 'contrib' dict maps block indices (br, bc) to coeff.
                # So C_block(br, bc) += contrib[(br, bc)] * M_t[br, bc]
                
                # Wait, the 'contrib' dict I defined earlier maps the *block indices* of C
                # to the coefficient of the *entire matrix* M_t.
                # So if contrib = {(0,0): 1, (1,1): 1}, it means:
                # C_block(0,0) += 1 * M_t
                # C_block(1,1) += 1 * M_t
                # This implies M_t[0,0] adds to C_block(0,0)[0,0], etc.
                
                # So for a specific entry M_t[mr, mc], it contributes to C_block(br, bc)
                # only if (br, bc) is in contrib?
                # No. The formula is C_block(br, bc) += coeff * M_t.
                # This means every entry of M_t adds to the corresponding entry of C_block(br, bc).
                # So M_t[mr, mc] adds to C_block(br, bc)[mr, mc] with coeff.
                
                # Therefore, for a given low-level term k, which contributes to M_t[mr, mc] with m_coeff,
                # we need to add m_coeff * contrib[(br, bc)] to W for the global index corresponding
                # to C_block(br, bc)[mr, mc].
                
                for (br, bc), c_coeff in contrib.items():
                    # The global row/col in C for block (br, bc) and local (mr, mc)
                    gr_c, gc_c = c_elem(br, bc, mr, mc)
                    w_idx = get_W_index(gr_c, gc_c)
                    W[w_idx, rank_idx] += m_coeff * c_coeff
    
    # Verify
    T = np.zeros((N_A, N_B, N_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx_val = i * m + j
                b_idx_val = j * p + k
                c_idx_val = k * n + i
                T[a_idx_val, b_idx_val, c_idx_val] = 1.0
                
    T_recon = np.einsum('as,bs,cs->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    
    if err > 1e-10:
        # Fallback to naive if error (should not happen)
        # But we return U, V, W anyway to satisfy interface, though it will be scored 0 if invalid.
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
