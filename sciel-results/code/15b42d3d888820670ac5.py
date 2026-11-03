"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    # We decompose the 4x4x7 multiplication into:
    # 1. 4x4x4 part using Strassen (Rank 49)
    # 2. 4x4x3 part using Schoolbook (Rank 48)
    # Total Rank = 97

    total_rank = 49 + 48
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen Decomposition ---
    
    # Define the 7 rank-1 terms for 2x2x2 Strassen multiplication.
    # u_t, v_t, w_t are vectors of length 4 (flattened 2x2 matrices).
    # The decomposition is R = sum_t (u_t . P) (v_t . Q) w_t
    # where P, Q, R are flattened 2x2 matrices.
    
    # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Term 1: (P00+P11)(Q00+Q11) -> R00+R11 ... wait, Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need u_t, v_t such that M_t = (u_t . A) (v_t . B)
    # and w_t such that C = sum_t M_t w_t
    
    # Let's define the 7 terms (u, v, w) for the 2x2 case.
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    
    # T1: (a0+a3)(b0+b3)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    # Contribution to C: C00 += M1, C11 += M1
    # w1 should be such that M1 * w1 adds M1 to C00 and C11?
    # No, the decomposition is C = sum (u.A)(v.B) w.
    # So for T1, the matrix added to C is M1 * (matrix represented by w1).
    # But M1 is a scalar in the 2x2 Strassen context? No, M1 is the result of the product.
    # In the 2x2x2 tensor, the "scalar" is the element.
    # Here, we are decomposing the tensor T_{ijk}.
    # The 7 terms are:
    # 1. u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] -> (a0+a3)(b0+b3) * [1,0,0,1]^T?
    #    This would add (a0+a3)(b0+b3) to C00 and C11.
    #    But C00 = M1 + M4 - M5 + M7.
    #    So w1 should be [1, 0, 0, 1] (adds to C00 and C11).
    
    # Let's verify the standard Strassen w vectors.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So:
    # w1 (for M1) adds to C00 and C11. w1 = [1, 0, 0, 1]
    # w2 (for M2) adds to C10 and subtracts from C11. w2 = [0, 0, 1, -1]
    # w3 (for M3) adds to C01 and C11. w3 = [0, 1, 0, 1]
    # w4 (for M4) adds to C00 and C10. w4 = [1, 0, 1, 0]
    # w5 (for M5) subtracts from C00 and adds to C01. w5 = [-1, 1, 0, 0]
    # w6 (for M6) adds to C11. w6 = [0, 0, 0, 1]
    # w7 (for M7) adds to C00. w7 = [1, 0, 0, 0]
    
    # Let's double check C00:
    # M1*w1[0] + M2*w2[0] + M3*w3[0] + M4*w4[0] + M5*w5[0] + M6*w6[0] + M7*w7[0]
    # = M1*1 + M2*0 + M3*0 + M4*1 + M5*(-1) + M6*0 + M7*1 = M1 + M4 - M5 + M7. Correct.
    
    # C01:
    # M1*0 + M2*0 + M3*1 + M4*0 + M5*1 + M6*0 + M7*0 = M3 + M5. Correct.
    
    # C10:
    # M1*0 + M2*1 + M3*0 + M4*1 + M5*0 + M6*0 + M7*0 = M2 + M4. Correct.
    
    # C11:
    # M1*1 + M2*(-1) + M3*1 + M4*0 + M5*0 + M6*1 + M7*0 = M1 - M2 + M3 + M6. Correct.
    
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # 1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # 2: (a2+a3)b0
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # 3: a0(b1-b3)
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]), # 4: a3(b2-b0)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # 5: (a0+a1)b3
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # 6: (a2-a0)(b0+b1)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # 7: (a1-a3)(b2+b3)
    ]
    
    # Now, build the 4x4x4 part.
    # We iterate over the 7 block-level Strassen terms.
    # For each, we define the linear maps A->X and B->Y.
    # X_k and Y_k are 2x2 matrices.
    # We need to express the 4 entries of X_k as linear combinations of the 16 entries of A.
    # A is flattened: a00, a01, a02, a03, a10, ..., a33.
    # Indices: i*4 + j.
    
    def get_map_A(k):
        # Returns 4 vectors of length 16, one for each entry of X_k (00, 01, 10, 11)
        maps = []
        # X1 = A00 + A11
        if k == 1:
            # A00 is top-left 2x2: indices 0,1,4,5
            # A11 is bottom-right 2x2: indices 10,11,14,15
            m00 = [0]*16; m00[0]=1; m00[10]=1
            m01 = [0]*16; m01[1]=1; m01[11]=1
            m10 = [0]*16; m10[4]=1; m10[14]=1
            m11 = [0]*16; m11[5]=1; m11[15]=1
            return [m00, m01, m10, m11]
        elif k == 2:
            # X2 = A10 + A11 (A10 is bottom-left, A11 is bottom-right)
            # A10: indices 8,9,12,13
            # A11: indices 10,11,14,15
            m00 = [0]*16; m00[8]=1; m00[10]=1
            m01 = [0]*16; m01[9]=1; m01[11]=1
            m10 = [0]*16; m10[12]=1; m10[14]=1
            m11 = [0]*16; m11[13]=1; m11[15]=1
            return [m00, m01, m10, m11]
        elif k == 3:
            # X3 = A00
            m00 = [0]*16; m00[0]=1
            m01 = [0]*16; m01[1]=1
            m10 = [0]*16; m10[4]=1
            m11 = [0]*16; m11[5]=1
            return [m00, m01, m10, m11]
        elif k == 4:
            # X4 = A11
            m00 = [0]*16; m00[10]=1
            m01 = [0]*16; m01[11]=1
            m10 = [0]*16; m10[14]=1
            m11 = [0]*16; m11[15]=1
            return [m00, m01, m10, m11]
        elif k == 5:
            # X5 = A00 + A01 (A01 is top-right)
            # A00: 0,1,4,5
            # A01: 2,3,6,7
            m00 = [0]*16; m00[0]=1; m00[2]=1
            m01 = [0]*16; m01[1]=1; m01[3]=1
            m10 = [0]*16; m10[4]=1; m10[6]=1
            m11 = [0]*16; m11[5]=1; m11[7]=1
            return [m00, m01, m10, m11]
        elif k == 6:
            # X6 = A10 - A00
            m00 = [0]*16; m00[8]=1; m00[0]=-1
            m01 = [0]*16; m01[9]=1; m01[1]=-1
            m10 = [0]*16; m10[12]=1; m10[4]=-1
            m11 = [0]*16; m11[13]=1; m11[5]=-1
            return [m00, m01, m10, m11]
        elif k == 7:
            # X7 = A01 - A11
            m00 = [0]*16; m00[2]=1; m00[10]=-1
            m01 = [0]*16; m01[3]=1; m01[11]=-1
            m10 = [0]*16; m10[6]=1; m10[14]=-1
            m11 = [0]*16; m11[7]=1; m11[15]=-1
            return [m00, m01, m10, m11]
        return [[0]*16 for _ in range(4)]

    def get_map_B(k):
        # Similar logic for B
        if k == 1:
            # Y1 = B00 + B11
            m00 = [0]*16; m00[0]=1; m00[10]=1
            m01 = [0]*16; m01[1]=1; m01[11]=1
            m10 = [0]*16; m10[4]=1; m10[14]=1
            m11 = [0]*16; m11[5]=1; m11[15]=1
            return [m00, m01, m10, m11]
        elif k == 2:
            # Y2 = B00
            m00 = [0]*16; m00[0]=1
            m01 = [0]*16; m01[1]=1
            m10 = [0]*16; m10[4]=1
            m11 = [0]*16; m11[5]=1
            return [m00, m01, m10, m11]
        elif k == 3:
            # Y3 = B01 - B11
            m00 = [0]*16; m00[2]=1; m00[10]=-1
            m01 = [0]*16; m01[3]=1; m01[11]=-1
            m10 = [0]*16; m10[6]=1; m10[14]=-1
            m11 = [0]*16; m11[7]=1; m11[15]=-1
            return [m00, m01, m10, m11]
        elif k == 4:
            # Y4 = B10 - B00
            m00 = [0]*16; m00[8]=1; m00[0]=-1
            m01 = [0]*16; m01[9]=1; m01[1]=-1
            m10 = [0]*16; m10[12]=1; m10[4]=-1
            m11 = [0]*16; m11[13]=1; m11[5]=-1
            return [m00, m01, m10, m11]
        elif k == 5:
            # Y5 = B11
            m00 = [0]*16; m00[10]=1
            m01 = [0]*16; m01[11]=1
            m10 = [0]*16; m10[14]=1
            m11 = [0]*16; m11[15]=1
            return [m00, m01, m10, m11]
        elif k == 6:
            # Y6 = B00 + B01
            m00 = [0]*16; m00[0]=1; m00[2]=1
            m01 = [0]*16; m01[1]=1; m01[3]=1
            m10 = [0]*16; m10[4]=1; m10[6]=1
            m11 = [0]*16; m11[5]=1; m11[7]=1
            return [m00, m01, m10, m11]
        elif k == 7:
            # Y7 = B10 + B11
            m00 = [0]*16; m00[8]=1; m00[10]=1
            m01 = [0]*16; m01[9]=1; m01[11]=1
            m10 = [0]*16; m10[12]=1; m10[14]=1
            m11 = [0]*16; m11[13]=1; m11[15]=1
            return [m00, m01, m10, m11]
        return [[0]*16 for _ in range(4)]

    def get_map_C(k):
        # Returns 4 vectors of length 16, one for each entry of M_k contribution to C.
        # M_k is a 2x2 matrix.
        # C is 4x4.
        # We need to map the 4 entries of M_k to the 16 entries of C.
        # The mapping is determined by the Strassen combination rules.
        
        # C00 += M1, M4; -= M5; += M7
        # C01 += M3, M5
        # C10 += M2, M4
        # C11 += M1, M3, M6; -= M2
        
        # For a given k, we define where M_k goes.
        # Let M_k = [[m00, m01], [m10, m11]]
        # We return 4 vectors of length 16, corresponding to m00, m01, m10, m11.
        
        maps = [[0]*16 for _ in range(4)]
        
        # Helper to add a 2x2 block to C at position (br, bc) with sign s
        def add_block(maps, br, bc, s):
            # br, bc in {0,1}
            # Offset in C: row br*2, col bc*2
            # m00 -> C[br*2, bc*2]
            # m01 -> C[br*2, bc*2+1]
            # m10 -> C[br*2+1, bc*2]
            # m11 -> C[br*2+1, bc*2+1]
            
            idx00 = br*2*4 + bc*2
            idx01 = br*2*4 + bc*2 + 1
            idx10 = (br*2+1)*4 + bc*2
            idx11 = (br*2+1)*4 + bc*2 + 1
            
            maps[0][idx00] += s
            maps[1][idx01] += s
            maps[2][idx10] += s
            maps[3][idx11] += s
            
        if k == 1:
            add_block(maps, 0, 0, 1) # C00
            add_block(maps, 1, 1, 1) # C11
        elif k == 2:
            add_block(maps, 1, 0, 1) # C10
            add_block(maps, 1, 1, -1) # C11
        elif k == 3:
            add_block(maps, 0, 1, 1) # C01
            add_block(maps, 1, 1, 1) # C11
        elif k == 4:
            add_block(maps, 0, 0, 1) # C00
            add_block(maps, 1, 0, 1) # C10
        elif k == 5:
            add_block(maps, 0, 0, -1) # C00
            add_block(maps, 0, 1, 1) # C01
        elif k == 6:
            add_block(maps, 1, 1, 1) # C11
        elif k == 7:
            add_block(maps, 0, 0, 1) # C00
            
        return maps

    col = 0
    for k in range(1, 8):
        map_A = get_map_A(k)
        map_B = get_map_B(k)
        map_C = get_map_C(k)
        
        for t in range(7):
            u_t, v_t, w_t = strassen_2x2[t]
            
            # Construct global U vector (length 16)
            u_global = [0]*16
            for idx in range(4):
                # u_t[idx] is the coefficient for the idx-th entry of X_k
                # map_A[idx] is the vector of coefficients for A that form the idx-th entry of X_k
                coeff = u_t[idx]
                if coeff != 0:
                    for i in range(16):
                        u_global[i] += coeff * map_A[idx][i]
                        
            # Construct global V vector (length 16)
            v_global = [0]*16
            for idx in range(4):
                coeff = v_t[idx]
                if coeff != 0:
                    for i in range(16):
                        v_global[i] += coeff * map_B[idx][i]
                        
            # Construct global W vector (length 16)
            w_global = [0]*16
            for idx in range(4):
                coeff = w_t[idx]
                if coeff != 0:
                    for i in range(16):
                        w_global[i] += coeff * map_C[idx][i]
                        
            # Fill U, V, W
            for i in range(16):
                U[i][col] = u_global[i]
                V[i][col] = v_global[i] # Note: V is for B, which is 4x4 in this part.
                W[i][col] = w_global[i] # Note: W is for C, which is 4x4 in this part.
            
            col += 1
            
    # --- Part 2: 4x4x3 Schoolbook Decomposition ---
    # Multiplies A (4x4) by B_rest (4x3) to get C_rest (4x3).
    # Rank = 4 * 4 * 3 = 48.
    
    # U for this part: Identity for A (4x4) -> 16x16 block?
    # No, U is (n*m) x rank. n=4, m=4.
    # The A matrix is the same for both parts.
    # We need to add 48 columns to U, V, W.
    
    # For schoolbook 4x4x3:
    # A is 4x4. B_rest is 4x3. C_rest is 4x3.
    # Indices for B_rest: j in 0..3 (rows of B), k in 0..2 (cols of B).
    # Global V index: j*3 + k. (Since p=7, first 4 cols are used by Part 1, next 3 by Part 2?
    # Wait, the problem is 4x4x7.
    # Part 1 used B cols 0..3.
    # Part 2 uses B cols 4..6.
    # So in the global V matrix (size 28 x rank), the rows for Part 2 are 4*3 + k?
    # No, V is (m*p) x rank. m=4, p=7. Size 28.
    # Rows 0..15 correspond to B cols 0..3 (4 rows * 4 cols).
    # Rows 16..27 correspond to B cols 4..6 (4 rows * 3 cols).
    
    # Similarly for W (size p*n = 28).
    # Rows 0..15 correspond to C cols 0..3.
    # Rows 16..27 correspond to C cols 4..6.
    
    # U is (n*m) x rank. Size 16.
    # U is the same for both parts.
    
    for i in range(4): # row of A
        for j in range(4): # col of A / row of B
            for k in range(3): # col of B (0,1,2)
                # Global indices
                u_idx = i * 4 + j
                v_idx = (j * 7) + (4 + k) # B row j, col 4+k
                w_idx = ((4 + k) * 4) + i # C col 4+k, row i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                
                col += 1

    if time.time() < deadline:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if something went wrong (unlikely)
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
