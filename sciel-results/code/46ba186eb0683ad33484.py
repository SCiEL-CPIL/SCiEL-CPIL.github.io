import time
import sys

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block matrix multiplication
    # We decompose A (2x4) into two 2x2 blocks: A_left (cols 0,1), A_right (cols 2,3)
    # B (4x4) into four 2x2 blocks: B_11, B_12, B_21, B_22
    # C (2x4) into two 2x2 blocks: C_left, C_right
    
    # For each half (left/right), we apply Strassen to compute C_half = A_half * B_half
    # where A_half is 2x2, B_half is 2x2 (stacked from B's columns)
    
    # Actually, let's think more carefully.
    # A is 2x4, B is 4x4, C is 2x4.
    # We can view this as computing two 2x2 matrix multiplications:
    # C_left = A_left * B_left, where A_left = A[:, 0:2], B_left = B[:, 0:2]
    # C_right = A_right * B_right, where A_right = A[:, 2:4], B_right = B[:, 2:4]
    
    # Each 2x2 * 2x2 multiplication can be done with rank 7 (Strassen).
    # But the A matrices are different for left and right, and the B matrices are different.
    # So total rank would be 7 + 7 = 14? No, that's not right either.
    
    # Let me reconsider. The current program uses rank 28. Let me verify it's correct
    # and try to find a better decomposition.
    
    # Actually, let me just keep the current Strassen-based approach but verify correctness.
    # The current score is 1.1429, which means rank 28 is valid.
    
    # Let me try a different approach: use the fact that M_{2,4,4} can be decomposed
    # using Kronecker product structure.
    
    # M_{2,4,4} = M_{2,2,2} ⊗ M_{1,2,2}
    # Rank(M_{2,2,2}) = 7 (Strassen)
    # Rank(M_{1,2,2}) = ? 
    # M_{1,2,2} has tensor T_{i,j,k} = 1 if i=j else 0, where i is row of A (1), j is col of A / row of B (2), k is col of B (2).
    # This is essentially two dot products of length 2.
    # Rank(M_{1,2,2}) = 2 (since the two outputs are independent linear combinations)
    # Actually no, let me think again.
    
    # For M_{1,2,2}: A is 1x2 = [a, b], B is 2x2 = [[c,d],[e,f]], C is 1x2 = [ac+be, ad+bf]
    # The tensor has entries T_{a,c}, T_{a,f}, T_{b,e}, T_{b,d} = 1 (in appropriate indexing)
    # This is a 2x2 matrix multiplication where A is 1x2.
    # Rank is 4 (schoolbook) or can it be less?
    # The four bilinear forms are ac, ad, be, bf. These are linearly independent.
    # So rank is at least 4? No, rank of a tensor is the minimum number of rank-1 tensors.
    # Each term like ac is a rank-1 term (a * c). So we need at least 4 rank-1 terms? 
    # But we can combine: a(c+d) + b(e+f) gives 2 terms but only produces [a(c+d), b(e+f)] not [ac+be, ad+bf].
    # So rank(M_{1,2,2}) = 4.
    
    # Therefore R(M_{2,4,4}) ≤ R(M_{2,2,2}) * R(M_{1,2,2}) = 7 * 4 = 28.
    # This matches the current solution!
    
    # Can we do better? R(M_{2,4,4}) is known to be at most 28 and at least 24 (I think).
    # Let's keep rank 28 for now and make sure the code is correct.
    
    # Let me rewrite the Strassen decomposition more carefully.
    
    # Block structure:
    # A = [A11 A12] where A11, A12 are 2x2 (rows 0-1, cols 0-1 and cols 2-3)
    # B = [B11 B12; B21 B22] where each is 2x2
    # C = [C11 C12] where C11, C12 are 2x2
    
    # C11 = A11 * B11 + A12 * B21
    # C12 = A11 * B12 + A12 * B22
    
    # Using Strassen for each product:
    # For A11 * B11: 7 multiplications
    # For A12 * B21: 7 multiplications  
    # For A11 * B12: 7 multiplications
    # For A12 * B22: 7 multiplications
    # Total: 28 multiplications
    
    # This is the same as the Kronecker product approach.
    
    # Let me implement this more cleanly.
    
    # For a 2x2 * 2x2 Strassen decomposition:
    # M1 = (a+b)(e+f)
    # M2 = (a+b)(g+h) 
    # M3 = a(d-f)
    # M4 = c(b-d)
    # M5 = (a+c)(e+f)
    # M6 = (b+c)(g+h)
    # M7 = (a-c)(e+d)
    
    # Where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # Result = [[M1+M4-M5+M7, M2-M4+M5-M6], [M3-M1+M6-M7, M2-M3+M4+M5]]
    
    def strassen_coeffs():
        """Return alpha, beta, w coefficients for 2x2 Strassen."""
        # alpha[s][i][j]: coefficient of A[i][j] in the s-th Strassen factor
        # beta[s][i][j]: coefficient of B[i][j] in the s-th Strassen factor
        # w[s][i][j]: coefficient of M_s in C[i][j]
        
        alpha = [
            {},  # unused
            {(0,0): 1, (0,1): 1},  # M1: (a+b)
            {(0,0): 1, (0,1): 1},  # M2: (a+b)
            {(0,0): 1},            # M3: a
            {(1,0): 1, (1,1): -1}, # M4: c(b-d) -> c * (b-d), so c with coeff 1, and we handle (b-d) in beta
            {(0,0): 1, (1,0): 1},  # M5: (a+c)
            {(0,1): 1, (1,0): 1},  # M6: (b+c)
            {(0,0): 1, (1,0): -1}, # M7: (a-c)
        ]
        
        beta = [
            {},  # unused
            {(0,0): 1, (1,0): 1},  # M1: (e+f)
            {(1,0): 1, (1,1): 1},  # M2: (g+h)
            {(0,1): 1, (1,1): -1}, # M3: (d-f)
            {(0,1): 1, (1,1): -1}, # M4: (b-d) -- wait, this is for A, not B
            {(0,0): 1, (1,0): 1},  # M5: (e+f)
            {(1,0): 1, (1,1): 1},  # M6: (g+h)
            {(0,0): 1, (0,1): 1},  # M7: (e+d)
        ]
        
        # Fix M4: M4 = c(b-d), so alpha[4] = {(1,0): 1}, beta[4] = {(0,1): 1, (1,1): -1}
        alpha[4] = {(1,0): 1}
        beta[4] = {(0,1): 1, (1,1): -1}
        
        w = [
            {},  # unused
            {(0,0): 1, (1,1): 1},  # M1
            {(0,1): 1, (1,0): -1}, # M2
            {(1,1): -1, (0,0): 1}, # M3
            {(1,0): 1, (0,1): -1}, # M4
            {(0,0): -1, (0,1): 1}, # M5
            {(1,0): -1, (1,1): 1}, # M6
            {(0,0): 1, (1,1): -1}, # M7
        ]
        
        # Verify: C[0][0] = M1 + M4 - M5 + M7
        # C[0][1] = M2 - M4 + M5 - M6
        # C[1][0] = M3 - M1 + M6 - M7
        # C[1][1] = M2 - M3 + M4 + M5
        
        # Let me re-verify with standard Strassen:
        # p1 = (a00 + a11) * (b00 + b11)
        # p2 = (a10 + a11) * b00
        # p3 = a00 * (b11 - b10)
        # p4 = a11 * (b01 - b00)
        # p5 = (a00 + a01) * b10
        # p6 = (a10 - a00) * (b00 + b01)
        # p7 = (a01 - a11) * (b10 + b11)
        
        # c00 = p1 + p4 - p5 + p7
        # c01 = p3 + p5
        # c10 = p2 + p6
        # c11 = p1 - p2 + p3 + p6
        
        # Let me use this standard form instead.
        
        alpha_std = [
            {},  # 0
            {(0,0): 1, (1,1): 1},  # p1: (a00+a11)
            {(1,0): 1, (1,1): 1},  # p2: (a10+a11)
            {(0,0): 1},            # p3: a00
            {(1,1): 1},            # p4: a11
            {(0,0): 1, (0,1): 1},  # p5: (a00+a01)
            {(1,0): 1, (0,0): -1}, # p6: (a10-a00)
            {(0,1): 1, (1,1): -1}, # p7: (a01-a11)
        ]
        
        beta_std = [
            {},  # 0
            {(0,0): 1, (1,1): 1},  # p1: (b00+b11)
            {(0,0): 1},            # p2: b00
            {(1,1): 1, (1,0): -1}, # p3: (b11-b10)
            {(0,1): 1, (0,0): -1}, # p4: (b01-b00)
            {(1,0): 1},            # p5: b10
            {(0,0): 1, (0,1): 1},  # p6: (b00+b01)
            {(1,0): 1, (1,1): 1},  # p7: (b10+b11)
        ]
        
        w_std = [
            {},  # 0
            {(0,0): 1, (1,1): 1},  # p1
            {(1,0): 1, (1,1): -1}, # p2
            {(0,1): 1, (1,1): 1},  # p3
            {(0,0): 1, (1,0): 1},  # p4
            {(0,0): -1, (0,1): 1}, # p5
            {(1,0): 1, (1,1): 1},  # p6
            {(0,0): 1},            # p7
        ]
        
        return alpha_std, beta_std, w_std
    
    alpha, beta, w = strassen_coeffs()
    
    # Now apply to our problem:
    # We have 4 products: A11*B11, A12*B21, A11*B12, A12*B22
    # Each uses 7 Strassen multiplications, total 28.
    
    # For each product, we need to map the 2x2 blocks to the full matrices.
    
    # A11: rows 0-1, cols 0-1 of A (which is 2x4)
    # A12: rows 0-1, cols 2-3 of A
    # B11: rows 0-1, cols 0-1 of B (which is 4x4)
    # B21: rows 2-3, cols 0-1 of B
    # B12: rows 0-1, cols 2-3 of B
    # B22: rows 2-3, cols 2-3 of B
    
    # C11: rows 0-1, cols 0-1 of C (which is 2x4)
    # C12: rows 0-1, cols 2-3 of C
    
    # Product 1: A11 * B11 -> contributes to C11
    # Product 2: A12 * B21 -> contributes to C11
    # Product 3: A11 * B12 -> contributes to C12
    # Product 4: A12 * B22 -> contributes to C12
    
    # For each product, we use 7 Strassen terms.
    
    # Product 1: A11 * B11
    for s in range(1, 8):
        col = 0 * 7 + (s - 1)  # columns 0-6
        for (i, j), val in alpha[s].items():
            # A11[i][j] is at A[i][j]
            u_idx = i * m + j
            U[u_idx][col] += val
        for (i, j), val in beta[s].items():
            # B11[i][j] is at B[i][j]
            v_idx = i * p + j
            V[v_idx][col] += val
        for (i, j), val in w[s].items():
            # C11[i][j] is at C[i][j]
            w_idx = j * n + i
            W[w_idx][col] += val
    
    # Product 2: A12 * B21
    for s in range(1, 8):
        col = 1 * 7 + (s - 1)  # columns 7-13
        for (i, j), val in alpha[s].items():
            # A12[i][j] is at A[i][j+2]
            u_idx = i * m + (j + 2)
            U[u_idx][col] += val
        for (i, j), val in beta[s].items():
            # B21[i][j] is at B[i+2][j]
            v_idx = (i + 2) * p + j
            V[v_idx][col] += val
        for (i, j), val in w[s].items():
            # C11[i][j] is at C[i][j]
            w_idx = j * n + i
            W[w_idx][col] += val
    
    # Product 3: A11 * B12
    for s in range(1, 8):
        col = 2 * 7 + (s - 1)  # columns 14-20
        for (i, j), val in alpha[s].items():
            # A11[i][j] is at A[i][j]
            u_idx = i * m + j
            U[u_idx][col] += val
        for (i, j), val in beta[s].items():
            # B12[i][j] is at B[i][j+2]
            v_idx = i * p + (j + 2)
            V[v_idx][col] += val
        for (i, j), val in w[s].items():
            # C12[i][j] is at C[i][j+2]
            w_idx = (j + 2) * n + i
            W[w_idx][col] += val
    
    # Product 4: A12 * B22
    for s in range(1, 8):
        col = 3 * 7 + (s - 1)  # columns 21-27
        for (i, j), val in alpha[s].items():
            # A12[i][j] is at A[i][j+2]
            u_idx = i * m + (j + 2)
            U[u_idx][col] += val
        for (i, j), val in beta[s].items():
            # B22[i][j] is at B[i+2][j+2]
            v_idx = (i + 2) * p + (j + 2)
            V[v_idx][col] += val
        for (i, j), val in w[s].items():
            # C12[i][j] is at C[i][j+2]
            w_idx = (j + 2) * n + i
            W[w_idx][col] += val
    
    return {"U": U, "V": V, "W": W}
