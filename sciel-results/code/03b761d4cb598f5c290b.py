import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    # n=4 (rows of A), m=4 (cols of A / rows of B), p=5 (cols of B)
    # Wait, prompt says: A is 4x4, B is 4x5.
    # So A is n x m? No, A is n x p?
    # Prompt: "multiplying a 4 by 4 matrix by a 4 by 5 matrix"
    # Usually A is (n x k), B is (k x m).
    # Here A is 4x4, B is 4x5.
    # So k=4. n=4, m=5.
    # But the code says DIMENSIONS = [4, 4, 5].
    # And U shape (n*m, rank).
    # If n=4, m=4, p=5:
    # U (16, R), V (20, R), W (20, R).
    # Seed: U[i*m+j], V[j*p+k], W[k*n+i].
    # If m=4, p=5, n=4:
    # U index i*4+j. (i in 0..3, j in 0..3).
    # V index j*5+k. (j in 0..3, k in 0..4).
    # W index k*4+i. (k in 0..4, i in 0..3).
    # This implies:
    # A indices: i, j? No.
    # Let's check the multiplication: C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # If A is 4x4 (i, j), B is 4x5 (j, k).
    # Then U is A factors (i, j). V is B factors (j, k). W is C factors (k, i).
    # This matches the seed indices:
    # U[i*m+j] -> i (row A), j (col A). m=4.
    # V[j*p+k] -> j (row B), k (col B). p=5.
    # W[k*n+i] -> k (row C), i (col C). n=4.
    # So A is 4x4, B is 4x5, C is 4x5.
    # The tensor T_{i, j, k} = 1 if A_{i, j} B_{j, k} contributes to C_{i, k}.
    # This is the standard matrix multiplication tensor.
    
    # We split B into B_left (4x4) and B_right (4x1).
    # Part 1: A (4x4) * B_left (4x4) -> C_left (4x4). Rank 49.
    # Part 2: A (4x4) * B_right (4x1) -> C_right (4x1). Rank 16.
    
    R1 = 49
    R2 = 16
    R = R1 + R2
    
    U = [[0] * R for _ in range(n * m)]  # 16 x 65
    V = [[0] * R for _ in range(m * p)]  # 20 x 65
    W = [[0] * R for _ in range(p * n)]  # 20 x 65
    
    # Part 1: 4x4x4 Strassen
    # We need the factors for 4x4x4.
    # We construct them from 2x2x2 Strassen factors.
    
    # 2x2x2 Strassen factors (u, v, w) for each of the 7 terms.
    # u, v, w are 2x2 matrices.
    # Term 1: P1 = (A00+A11)(B00+B11)
    # u1 = [[1, 0], [0, 1]]? No.
    # P1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11.
    # u1 = [[1, 0], [0, 1]]? No, u1 is the coefficient matrix for A.
    # u1 = [[1, 0], [0, 1]]? No.
    # u1[i, j] is coeff of A[i, j] in the first factor.
    # First factor: a00 + a11. So u1 = [[1, 0], [0, 1]].
    # Second factor: b00 + b11. So v1 = [[1, 0], [0, 1]].
    # w1: coeff of P1 in C.
    # C00 = P1 - P4 + P6. C01 = P3 + P5. C10 = P2 + P4. C11 = P1 - P3 + P7.
    # w1 = [[1, 0], [0, 1]].
    
    # Let's define the 7 terms for 2x2.
    # u, v, w are 2x2 lists.
    strassen_2x2 = [
        {
            'u': [[1, 0], [0, 1]],  # a00+a11
            'v': [[1, 0], [0, 1]],  # b00+b11
            'w': [[1, 0], [0, 1]]   # C00, C11
        },
        {
            'u': [[0, 0], [1, 1]],  # a10+a11
            'v': [[1, 0], [0, 0]],  # b00
            'w': [[0, 0], [1, 0]]   # C10
        },
        {
            'u': [[1, 0], [0, 0]],  # a00
            'v': [[0, 0], [1, -1]], # b10-b11
            'w': [[0, 1], [0, 0]]   # C01
        },
        {
            'u': [[0, 0], [0, 1]],  # a11
            'v': [[0, 1], [-1, 0]], # b01-b00
            'w': [[-1, 0], [0, 1]]  # -C00, +C10
        },
        {
            'u': [[1, 1], [0, 0]],  # a00+a01
            'v': [[0, 0], [0, 1]],  # b11
            'w': [[0, 1], [0, 0]]   # C01
        },
        {
            'u': [[-1, 0], [1, 0]], # a10-a00
            'v': [[1, 1], [0, 0]],  # b00+b01
            'w': [[0, 0], [1, 0]]   # C10
        },
        {
            'u': [[0, 1], [-1, 0]], # a01-a11
            'v': [[0, 0], [1, 1]],  # b10+b11
            'w': [[0, 0], [0, 1]]   # C11
        }
    ]
    
    # Now build 4x4x4 factors.
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11.
    # B is 4x4, split into B00, B01, B10, B11.
    # C is 4x4, split into C00, C01, C10, C11.
    # The 7 high-level terms M1..M7 are 2x2 multiplications.
    # Each M_i is computed using the 7 low-level Strassen terms.
    # Total 49 terms.
    
    # High-level Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # We need to map the high-level u, v, w to the 4x4 matrices.
    # For each high-level term i (0..6), we have 7 low-level terms j (0..6).
    # Total 49 terms.
    
    col = 0
    for hi in range(7):
        # Determine the high-level u, v, w matrices (2x2 blocks)
        # These are sparse matrices of block coefficients.
        # u_hi is 2x2 matrix of 2x2 blocks.
        # v_hi is 2x2 matrix of 2x2 blocks.
        # w_hi is 2x2 matrix of 2x2 blocks.
        
        # We can define the high-level coefficients.
        # Let's use the same logic as 2x2 but with blocks.
        
        # Instead of hardcoding, we can reuse the strassen_2x2 structure.
        # The high-level u, v, w are determined by the same formulas.
        # u_hi = strassen_2x2[hi]['u'] (as block matrix)
        # v_hi = strassen_2x2[hi]['v'] (as block matrix)
        # w_hi = strassen_2x2[hi]['w'] (as block matrix)
        
        # Now, for each low-level term j:
        for lo in range(7):
            # The low-level u, v, w are 2x2 matrices.
            # u_lo = strassen_2x2[lo]['u']
            # v_lo = strassen_2x2[lo]['v']
            # w_lo = strassen_2x2[lo]['w']
            
            # The total u for this term (4x4) is the product of the block structure.
            # u_total[i, j] = sum_{block_i, block_j} u_hi[block_i, block_j] * u_lo[i%2, j%2]?
            # No, the indices are:
            # A is 4x4. Index (r, c). r = 2*br + i, c = 2*bc + j.
            # u_total[r, c] = u_hi[br, bc] * u_lo[i, j].
            
            u_4x4 = [[0]*4 for _ in range(4)]
            v_4x4 = [[0]*4 for _ in range(4)]
            w_4x4 = [[0]*4 for _ in range(4)]
            
            for br in range(2):
                for bc in range(2):
                    uh = strassen_2x2[hi]['u'][br][bc]
                    vh = strassen_2x2[hi]['v'][br][bc]
                    wh = strassen_2x2[hi]['w'][br][bc]
                    
                    if uh != 0:
                        for i in range(2):
                            for j in range(2):
                                ul = strassen_2x2[lo]['u'][i][j]
                                u_4x4[2*br+i][2*bc+j] = uh * ul
                                
                    if vh != 0:
                        for i in range(2):
                            for j in range(2):
                                vl = strassen_2x2[lo]['v'][i][j]
                                v_4x4[2*br+i][2*bc+j] = vh * vl
                                
                    if wh != 0:
                        for i in range(2):
                            for j in range(2):
                                wl = strassen_2x2[lo]['w'][i][j]
                                w_4x4[2*br+i][2*bc+j] = wh * wl
            
            # Now map u_4x4, v_4x4, w_4x4 to U, V, W columns.
            # U index: i*m + j. m=4.
            # V index: j*p + k. p=4 (for this part).
            # W index: k*n + i. n=4.
            
            for i in range(4):
                for j in range(4):
                    if u_4x4[i][j] != 0:
                        U[i*4+j][col] = u_4x4[i][j]
                        
            for j in range(4):
                for k in range(4):
                    if v_4x4[j][k] != 0:
                        V[j*4+k][col] = v_4x4[j][k]
                        
            for k in range(4):
                for i in range(4):
                    if w_4x4[k][i] != 0:
                        W[k*4+i][col] = w_4x4[k][i]
                        
            col += 1
            
    # Part 2: 4x4x1 Schoolbook
    # A (4x4) * B_right (4x1) -> C_right (4x1).
    # B_right corresponds to j=4 in B (since B is 4x5, cols 0..4).
    # So B indices: j=0..3 for left, j=4 for right.
    # In V, index is j*p + k. p=5.
    # For B_right, j=4. V index = 4*5 + k = 20 + k.
    # In W, index is k*n + i. n=4.
    # In U, index is i*m + j. m=4.
    # Wait, U is for A. A is 4x4. Indices i, j (j is col of A, which is row of B).
    # So U index i*4 + j.
    
    for i in range(4):
        for k in range(4):
            # Term: A[i, k] * B[k, 4] -> C[i, 4]
            # U[i, k] = 1
            # V[4, k] = 1 -> V index 4*5 + k = 20 + k
            # W[4, i] = 1 -> W index 4*4 + i = 16 + i
            
            U[i*4+k][col] = 1
            V[20+k][col] = 1
            W[16+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
