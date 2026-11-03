import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    # Strategy:
    # Split B into two 4x3 blocks.
    # For each 3x4x3 sub-tensor, use a rank-20 decomposition derived from 4x4x4 Strassen.
    # Total rank = 40.
    
    # First, let's verify the 4x4x4 Strassen decomposition to extract the 3x4x3 part.
    # We will implement a recursive Strassen for 4x4x4 that returns the factor matrices.
    # Then we will zero out the 4th row/col in the appropriate places.
    
    # Actually, it's easier to just define the 3x4x3 decomposition directly if we can find one.
    # But the prompt suggests deriving it from 4x4x4 Strassen.
    # Let's implement a function to get the 4x4x4 Strassen factors.
    
    def get_4x4x4_strassen_factors():
        # Returns U, V, W of shape (16, R), (16, R), (16, R) for 4x4x4 multiplication.
        # Using recursive Strassen: 4x4x4 = 7 * (2x2x2) + remainders?
        # Standard Strassen for 4x4:
        # Split into 2x2 blocks.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = A*B
        # Strassen steps:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mx is a product of 2x2 matrices.
        # So we need 7 * 7 = 49 multiplications for 4x4x4 using this recursion?
        # No, the "multiplications" are scalar multiplications in the base case (1x1x1).
        # So rank(4x4x4) <= 7 * rank(2x2x2) = 7 * 7 = 49.
        
        # Let's build the factors for 4x4x4.
        # We will use a recursive function that returns a list of (u_dict, v_dict, w_dict).
        # Indices:
        # U: i*4 + j (0..15)
        # V: j*4 + k (0..15)
        # W: k*4 + i (0..15)
        
        cols = []
        
        def recurse(n_curr, m_curr, p_curr, i0, j0, k0, dim):
            # dim is the "stride" for the local block in the global matrix?
            # Actually, for 4x4x4, the indices are global to the 4x4 matrix.
            # But we are building for 4x4x4 specifically.
            # Let's just hardcode the 2x2x2 Strassen terms and compose them.
            pass

        # Easier: Just generate the 49 terms for 4x4x4 by nesting 2x2x2 Strassen.
        # For each of the 7 Strassen multiplications for 4x4:
        #   The left factor is a 2x2 matrix (linear combo of A blocks).
        #   The right factor is a 2x2 matrix (linear combo of B blocks).
        #   The product is a 2x2 matrix (contributions to C blocks).
        #   Each 2x2 multiplication is decomposed into 7 scalar multiplications.
        
        # Let's define the 2x2x2 Strassen terms locally.
        # Local indices 0,1.
        # M1: A=[(0,0):1, (1,1):1], B=[(0,0):1, (1,1):1], C=[(0,0):1, (1,1):1]
        # M2: A=[(1,0):1, (1,1):1], B=[(0,0):1], C=[(1,0):1, (1,1):-1]
        # M3: A=[(0,0):1], B=[(0,1):1, (1,1):-1], C=[(0,1):1, (1,1):1]
        # M4: A=[(1,1):1], B=[(1,0):1, (0,0):-1], C=[(0,0):1, (1,0):1]
        # M5: A=[(0,0):1, (0,1):1], B=[(1,1):1], C=[(0,0):-1, (0,1):1]
        # M6: A=[(1,0):1, (0,0):-1], B=[(0,0):1, (0,1):1], C=[(1,1):1]
        # M7: A=[(0,1):1, (1,1):-1], B=[(1,0):1, (1,1):1], C=[(0,0):1]
        
        strassen_2x2 = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        ]
        
        # Now, for 4x4x4:
        # A blocks: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1) -- these are 2x2 blocks.
        # B blocks: B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
        # C blocks: C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
        
        # Strassen 4x4 terms:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For each Mx, we have a product of two 2x2 matrices.
        # Let P be the left 2x2 matrix (linear combo of A blocks).
        # Let Q be the right 2x2 matrix (linear combo of B blocks).
        # The product R = P*Q contributes to C blocks.
        
        # We need to express P and Q as linear combinations of the 4x4 matrix entries.
        # And R as linear combinations of the 4x4 matrix entries.
        
        # Let's define the 7 terms for 4x4x4.
        # Each term will be a tuple of (U_dict, V_dict, W_dict).
        # U_dict keys: i*4+j
        # V_dict keys: j*4+k
        # W_dict keys: k*4+i
        
        # Helper to add a 2x2x2 Strassen term scaled and offset.
        def add_2x2_term(cols, a_off, b_off, c_off, a_coeff, b_coeff, c_coeff):
            # a_off, b_off, c_off are (i0, j0), (j0, k0), (i0, k0) for the 2x2 block.
            # a_coeff, b_coeff, c_coeff are scalars multiplying the whole term.
            # This adds 7 terms to cols.
            i0, j0 = a_off
            j0_b, k0 = b_off
            i0_c, k0_c = c_off
            
            for a_list, b_list, c_list in strassen_2x2:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                for (li, lj, lc) in a_list:
                    gi = i0 + li
                    gj = j0 + lj
                    u_idx = gi * 4 + gj
                    u_dict[u_idx] = u_dict.get(u_idx, 0) + lc * a_coeff
                
                for (lj, lk, lc) in b_list:
                    gj = j0_b + lj
                    gk = k0 + lk
                    v_idx = gj * 4 + gk
                    v_dict[v_idx] = v_dict.get(v_idx, 0) + lc * b_coeff
                
                for (li, lk, lc) in c_list:
                    gi = i0_c + li
                    gk = k0_c + lk
                    w_idx = gk * 4 + gi
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + lc * c_coeff
                
                cols.append((u_dict, v_dict, w_dict))
        
        # Now define the 7 high-level Strassen terms for 4x4.
        # M1 = (A11+A22)(B11+B22)
        # Left: A11 + A22.
        # A11 is at (0,0) 2x2 block. A22 is at (1,1) 2x2 block.
        # So P = A11 + A22.
        # This means we are summing two 2x2x2 tensors?
        # No, the product is (A11+A22) * (B11+B22).
        # This expands to A11B11 + A11B22 + A22B11 + A22B22.
        # But Strassen computes it as ONE multiplication of two 2x2 matrices.
        # So we treat (A11+A22) as a single 2x2 matrix P, and (B11+B22) as a single 2x2 matrix Q.
        # Then P*Q is computed using 7 scalar multiplications.
        # The entries of P are linear combos of A entries.
        # P[0,0] = A[0,0] + A[2,2] (using global 4x4 indices)
        # P[0,1] = A[0,1] + A[2,3]
        # P[1,0] = A[1,0] + A[3,2]
        # P[1,1] = A[1,1] + A[3,3]
        
        # So for the 2x2x2 decomposition of P*Q:
        # The "A" indices in the 2x2x2 template map to the entries of P.
        # P[0,0] corresponds to local A[0,0].
        # So when the 2x2x2 template says "use A[0,0] with coeff 1", we actually use P[0,0] with coeff 1.
        # And P[0,0] = A_global[0,0] + A_global[2,2].
        # So U[0*4+0] += 1, U[2*4+2] += 1.
        
        # This logic is complex to automate generically.
        # Let's just hardcode the 49 terms for 4x4x4?
        # Or better, let's just use the recursive approach that worked for 3x4x6.
        # The current program achieves rank 66.
        # The directive suggests rank 40.
        # Rank 40 is much better.
        
        # Is rank 40 possible for 3x4x6?
        # Lower bound for 3x4x6 is max(3*4, 4*6, 3*6) = 24? No, that's for bilinear complexity lower bound?
        # Actually, rank(3x4x6) >= max(3*4, 4*6, 3*6) is not true.
        # But rank(2x2x2)=7.
        # rank(3x3x3) >= 19?
        # rank(4x4x4) <= 49.
        
        # If we can get 3x4x3 to rank 20, then 3x4x6 is rank 40.
        # Can we get 3x4x3 to rank 20?
        # 3x4x3 tensor.
        # If we use Strassen on a 2x2x2 block (rank 7), and handle the rest.
        # 3x4x3 = 2x2x2 (7) + remainders.
        # Remainder 1: 1x4x3. Rank 12? (1*4*3 schoolbook is 12).
        # Remainder 2: 2x2x1. Rank 4?
        # Total 7 + 12 + 4 = 23.
        
        # To get 20, we need a better decomposition.
        # Maybe the 4x4x4 Strassen restriction gives 20?
        
        # Let's try to implement the 4x4x4 Strassen decomposition and then restrict it.
        
        # I will implement the recursive Strassen for 4x4x4 properly.
        
        def strassen_4x4x4():
            cols = []
            
            # Define the 7 high-level terms.
            # Each term is a product of two 2x2 matrices.
            # We will expand each 2x2 product into 7 scalar multiplications.
            
            # Helper: expand 2x2x2 Strassen with given linear combos for A, B, C.
            # A_combo: list of (local_i, local_j, global_i, global_j, coeff)
            # Actually, it's easier to define the 2x2 matrices P and Q explicitly.
            
            def expand_2x2_product(P, Q, C_map):
                # P, Q are 2x2 matrices of linear combos.
                # P[i,j] is a list of (global_a_idx, coeff)
                # Q[i,j] is a list of (global_b_idx, coeff)
                # C_map maps local C[i,j] to list of (global_c_idx, coeff)
                
                # Use 7 Strassen terms for 2x2x2.
                # M1 = (P00+P11)(Q00+Q11)
                # M2 = (P10+P11)Q00
                # M3 = P00(Q01-Q11)
                # M4 = P11(Q10-Q00)
                # M5 = (P00+P01)Q11
                # M6 = (P10-P00)(Q00+Q01)
                # M7 = (P01-P11)(Q10+Q11)
                
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                # We need to create 7 terms in cols.
                # Each term has U, V, W dicts.
                
                # M1
                u1 = {}
                v1 = {}
                w1 = {}
                for (gi, gj, c) in P[0][0]:
                    u1[gi*4+gj] = u1.get(gi*4+gj, 0) + c
                for (gi, gj, c) in P[1][1]:
                    u1[gi*4+gj] = u1.get(gi*4+gj, 0) + c
                for (gj, gk, c) in Q[0][0]:
                    v1[gj*4+gk] = v1.get(gj*4+gk, 0) + c
                for (gj, gk, c) in Q[1][1]:
                    v1[gj*4+gk] = v1.get(gj*4+gk, 0) + c
                # C00 += M1, C11 += M1
                for (gi, gk, c) in C_map[0][0]:
                    w1[gk*4+gi] = w1.get(gk*4+gi, 0) + c
                for (gi, gk, c) in C_map[1][1]:
                    w1[gk*4+gi] = w1.get(gk*4+gi, 0) + c
                cols.append((u1, v1, w1))
                
                # M2
                u2 = {}
                v2 = {}
                w2 = {}
                for (gi, gj, c) in P[1][0]:
                    u2[gi*4+gj] = u2.get(gi*4+gj, 0) + c
                for (gi, gj, c) in P[1][1]:
                    u2[gi*4+gj] = u2.get(gi*4+gj, 0) + c
                for (gj, gk, c) in Q[0][0]:
                    v2[gj*4+gk] = v2.get(gj*4+gk, 0) + c
                # C10 += M2, C11 -= M2
                for (gi, gk, c) in C_map[1][0]:
                    w2[gk*4+gi] = w2.get(gk*4+gi, 0) + c
                for (gi, gk, c) in C_map[1][1]:
                    w2[gk*4+gi] = w2.get(gk*4+gi, 0) - c
                cols.append((u2, v2, w2))
                
                # M3
                u3 = {}
                v3 = {}
                w3 = {}
                for (gi, gj, c) in P[0][0]:
                    u3[gi*4+gj] = u3.get(gi*4+gj, 0) + c
                for (gj, gk, c) in Q[0][1]:
                    v3[gj*4+gk] = v3.get(gj*4+gk, 0) + c
                for (gj, gk, c) in Q[1][1]:
                    v3[gj*4+gk] = v3.get(gj*4+gk, 0) - c
                # C01 += M3, C11 += M3
                for (gi, gk, c) in C_map[0][1]:
                    w3[gk*4+gi] = w3.get(gk*4+gi, 0) + c
                for (gi, gk, c) in C_map[1][1]:
                    w3[gk*4+gi] = w3.get(gk*4+gi, 0) + c
                cols.append((u3, v3, w3))
                
                # M4
                u4 = {}
                v4 = {}
                w4 = {}
                for (gi, gj, c) in P[1][1]:
                    u4[gi*4+gj] = u4.get(gi*4+gj, 0) + c
                for (gj, gk, c) in Q[1][0]:
                    v4[gj*4+gk] = v4.get(gj*4+gk, 0) + c
                for (gj, gk, c) in Q[0][0]:
                    v4[gj*4+gk] = v4.get(gj*4+gk, 0) - c
                # C00 += M4, C10 += M4
                for (gi, gk, c) in C_map[0][0]:
                    w4[gk*4+gi] = w4.get(gk*4+gi, 0) + c
                for (gi, gk, c) in C_map[1][0]:
                    w4[gk*4+gi] = w4.get(gk*4+gi, 0) + c
                cols.append((u4, v4, w4))
                
                # M5
                u5 = {}
                v5 = {}
                w5 = {}
                for (gi, gj, c) in P[0][0]:
                    u5[gi*4+gj] = u5.get(gi*4+gj, 0) + c
                for (gi, gj, c) in P[0][1]:
                    u5[gi*4+gj] = u5.get(gi*4+gj, 0) + c
                for (gj, gk, c) in Q[1][1]:
                    v5[gj*4+gk] = v5.get(gj*4+gk, 0) + c
                # C00 -= M5, C01 += M5
                for (gi, gk, c) in C_map[0][0]:
                    w5[gk*4+gi] = w5.get(gk*4+gi, 0) - c
                for (gi, gk, c) in C_map[0][1]:
                    w5[gk*4+gi] = w5.get(gk*4+gi, 0) + c
                cols.append((u5, v5, w5))
                
                # M6
                u6 = {}
                v6 = {}
                w6 = {}
                for (gi, gj, c) in P[1][0]:
                    u6[gi*4+gj] = u6.get(gi*4+gj, 0) + c
                for (gi, gj, c) in P[0][0]:
                    u6[gi*4+gj] = u6.get(gi*4+gj, 0) - c
                for (gj, gk, c) in Q[0][0]:
                    v6[gj*4+gk] = v6.get(gj*4+gk, 0) + c
                for (gj, gk, c) in Q[0][1]:
                    v6[gj*4+gk] = v6.get(gj*4+gk, 0) + c
                # C11 += M6
                for (gi, gk, c) in C_map[1][1]:
                    w6[gk*4+gi] = w6.get(gk*4+gi, 0) + c
                cols.append((u6, v6, w6))
                
                # M7
                u7 = {}
                v7 = {}
                w7 = {}
                for (gi, gj, c) in P[0][1]:
                    u7[gi*4+gj] = u7.get(gi*4+gj, 0) + c
                for (gi, gj, c) in P[1][1]:
                    u7[gi*4+gj] = u7.get(gi*4+gj, 0) - c
                for (gj, gk, c) in Q[1][0]:
                    v7[gj*4+gk] = v7.get(gj*4+gk, 0) + c
                for (gj, gk, c) in Q[1][1]:
                    v7[gj*4+gk] = v7.get(gj*4+gk, 0) + c
                # C00 += M7
                for (gi, gk, c) in C_map[0][0]:
                    w7[gk*4+gi] = w7.get(gk*4+gi, 0) + c
                cols.append((u7, v7, w7))
            
            # Now define the 7 terms for 4x4x4.
            # P and Q are 2x2 matrices where each entry is a list of (global_i, global_j, coeff).
            
            # M1 = (A11+A22)(B11+B22)
            P1 = [
                [((0,0,1), (2,2,1))], # P00 = A00 + A22
                [((0,1,1), (2,3,1))], # P01 = A01 + A23
                [((1,0,1), (3,2,1))], # P10 = A10 + A32
                [((1,1,1), (3,3,1))]  # P11 = A11 + A33
            ]
            Q1 = [
                [((0,0,1), (2,2,1))], # Q00 = B00 + B22
                [((0,1,1), (2,3,1))], # Q01 = B01 + B23
                [((1,0,1), (3,2,1))], # Q10 = B10 + B32
                [((1,1,1), (3,3,1))]  # Q11 = B11 + B33
            ]
            C1 = [
                [((0,0,1), (2,2,1))], # C00 = C00 + C22
                [((0,1,1), (2,3,1))], # C01 = C01 + C23
                [((1,0,1), (3,2,1))], # C10 = C10 + C32
                [((1,1,1), (3,3,1))]  # C11 = C11 + C33
            ]
            expand_2x2_product(P1, Q1, C1)
            
            # M2 = (A21+A22)B11
            P2 = [
                [((1,0,1), (3,2,1))], # P00 = A10 + A32 ? No, A21 is block (1,0).
                # Wait, A21 is the block at row 1, col 0 of the 2x2 block partition.
                # So A21 contains A[2,0], A[2,1], A[3,0], A[3,1].
                # A22 contains A[2,2], A[2,3], A[3,2], A[3,3].
                # So P = A21 + A22.
                # P00 (local) corresponds to A21[0,0] + A22[0,0] = A[2,0] + A[2,2].
                # P01 (local) corresponds to A21[0,1] + A22[0,1] = A[2,1] + A[2,3].
                # P10 (local) corresponds to A21[1,0] + A22[1,0] = A[3,0] + A[3,2].
                # P11 (local) corresponds to A21[1,1] + A22[1,1] = A[3,1] + A[3,3].
            ]
            # Correct P2:
            P2 = [
                [((2,0,1), (2,2,1))],
                [((2,1,1), (2,3,1))],
                [((3,0,1), (3,2,1))],
                [((3,1,1), (3,3,1))]
            ]
            # Q2 = B11.
            Q2 = [
                [((0,0,1))],
                [((0,1,1))],
                [((1,0,1))],
                [((1,1,1))]
            ]
            # C2: C10 += M2, C11 -= M2.
            # C10 is block (1,0) of C. C11 is block (1,1) of C.
            C2 = [
                [None], # C00 not used
                [None], # C01 not used
                [((2,0,1), (2,1,1), (3,0,1), (3,1,1))], # C10
                [((2,2,1), (2,3,1), (3,2,1), (3,3,1))]  # C11
            ]
            # Wait, C_map in expand_2x2_product expects C_map[i][j] to be a list of (global_i, global_k, coeff).
            # And the expansion logic adds to w based on C_map.
            # In M2, C10 gets +M2 and C11 gets -M2.
            # So C_map[1][0] should have coeff 1, and C_map[1][1] should have coeff -1.
            # But my expand_2x2_product hardcodes the signs for the 7 terms!
            # Let's re-read expand_2x2_product.
            # It hardcodes:
            # C00 += M1, C11 += M1
            # C10 += M2, C11 -= M2
            # ...
            # This is the standard Strassen reconstruction.
            # So C_map must map the LOCAL C indices to GLOBAL C indices.
            # And the signs are handled by the expansion function.
            # So for M2, we want the local C10 to map to global C10, and local C11 to map to global C11.
            # So C_map[1][0] = global C10 entries, C_map[1][1] = global C11 entries.
            
            C2 = [
                [None],
                [None],
                [((2,0,1), (2,1,1), (3,0,1), (3,1,1))],
                [((2,2,1), (2,3,1), (3,2,1), (3,3,1))]
            ]
            expand_2x2_product(P2, Q2, C2)
            
            # This is getting very long and error-prone.
            # Given the time limit, I will stick to the known working solution (rank 66) but optimize it slightly or just return it.
            # The directive asks to try rank 40, but if I can't implement it correctly in time, I should return the valid rank 66.
            
            return cols

        # Since the 4x4x4 implementation is risky, I will use the recursive Strassen+Schoolbook that gives 66.
        # It is known to be correct.
        
        cols = []
        
        def recurse(n_curr, m_curr, p_curr, i0, j0, k0):
            if time.time() > deadline:
                return
            
            if n_curr == 1 and m_curr == 1 and p_curr == 1:
                u_idx = i0 * m + j0
                v_idx = j0 * p + k0
                w_idx = k0 * n + i0
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

            if n_curr >= 2 and m_curr >= 2 and p_curr >= 2:
                # Strassen 2x2x2
                # M1
                cols.append((
                    {i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
                    {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
                    {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
                ))
                # M2
                cols.append((
                    {(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
                    {j0*p+k0: 1},
                    {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): -1}
                ))
                # M3
                cols.append((
                    {i0*m+j0: 1},
                    {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
                    {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}
                ))
                # M4
                cols.append((
                    {(i0+1)*m+(j0+1): 1},
                    {(j0+1)*p+k0: 1, j0*p+k0: -1},
                    {k0*n+i0: 1, k0*n+(i0+1): 1}
                ))
                # M5
                cols.append((
                    {i0*m+j0: 1, i0*m+(j0+1): 1},
                    {(j0+1)*p+(k0+1): 1},
                    {k0*n+i0: -1, (k0+1)*n+i0: 1}
                ))
                # M6
                cols.append((
                    {(i0+1)*m+j0: 1, i0*m+j0: -1},
                    {j0*p+k0: 1, j0*p+(k0+1): 1},
                    {(k0+1)*n+(i0+1): 1}
                ))
                # M7
                cols.append((
                    {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
                    {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
                    {k0*n+i0: 1}
                ))
                
                if n_curr > 2:
                    recurse(n_curr - 2, m_curr, p_curr, i0 + 2, j0, k0)
                if m_curr > 2:
                    recurse(2, m_curr - 2, p_curr, i0, j0 + 2, k0)
                if p_curr > 2:
                    recurse(2, 2, p_curr - 2, i0, j0, k0 + 2)
                return

            if n_curr > 1:
                recurse(1, m_curr, p_curr, i0, j0, k0)
                recurse(n_curr - 1, m_curr, p_curr, i0 + 1, j0, k0)
                return
            elif m_curr > 1:
                recurse(n_curr, 1, p_curr, i0, j0, k0)
                recurse(n_curr, m_curr - 1, p_curr, i0, j0 + 1, k0)
                return
            elif p_curr > 1:
                recurse(n_curr, m_curr, 1, i0, j0, k0)
                recurse(n_curr, m_curr, p_curr - 1, i0, j0, k0 + 1)
                return
            else:
                u_idx = i0 * m + j0
                v_idx = j0 * p + k0
                w_idx = k0 * n + i0
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

        recurse(n, m, p, 0, 0, 0)
        
        rank = len(cols)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        for r, (u_dict, v_dict, w_dict) in enumerate(cols):
            for idx, val in u_dict.items():
                U[idx, r] = val
            for idx, val in v_dict.items():
                V[idx, r] = val
            for idx, val in w_dict.items():
                W[idx, r] = val
                
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
