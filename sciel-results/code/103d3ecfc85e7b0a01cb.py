"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin from 150s limit
    n, m, p = DIMENSIONS
    
    # We want to find a rank-minimal decomposition for multiplying
    # A (n x m) by B (m x p) to get C (n x p)
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # where C = sum_r U_r * V_r * W_r (in appropriate tensor form)
    
    # The tensor T[i, j, k] = 1 if the output C[i,k] depends on A[i,j]*B[j,k]
    # Actually, the matrix multiplication tensor has entries T[i,j,k] = delta
    # Let me think about this more carefully.
    
    # For matrix multiplication: C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor is T[i, j, k] = 1 for all i, j, k (it's the all-ones tensor in some sense)
    # No wait, the tensor for matrix multiplication is:
    # T[i, j, k] represents the coefficient of A[i,j] * B[j,k] in C[i,k]
    # So T[i, j, k] = 1 if the output index (i,k) corresponds to input indices (i,j) and (j,k)
    # 
    # In the factorization:
    # U is (n*m) x rank: U[i*m+j, r]
    # V is (m*p) x rank: V[j*p+k, r]  
    # W is (p*n) x rank: W[k*n+i, r]
    # 
    # The condition is: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]
    # Wait, that's not right either.
    
    # Let me reconsider. The matrix multiplication tensor is T in R^(n*m) x R^(m*p) x R^(n*p)
    # T[i*m+j, j*p+k, k*n+i] = 1 if and only if the three indices are consistent
    # i.e., the middle index j is the same in all three.
    # 
    # Actually, T[a, b, c] where:
    # a = i*m + j (index into A)
    # b = j*p + k (index into B)  
    # c = k*n + i (index into C, using trace convention W[k*n+i])
    # 
    # T[a, b, c] = 1 iff the j in a equals the j in b, and the k in b equals the k in c,
    # and the i in a equals the i in c.
    
    # For a rank-r decomposition: T = sum_{r=1}^R U_r \otimes V_r \otimes W_r
    # where U_r in R^(n*m), V_r in R^(m*p), W_r in R^(n*p)
    # 
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # So we need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # for all (i,j) in A indices, (j',k) in B indices, (k',i') in C indices.
    
    # The schoolbook solution has rank n*m*p with:
    # For each (i,j,k): U[i*m+j, r(i,j,k)] = 1, V[j*p+k, r(i,j,k)] = 1, W[k*n+i, r(i,j,k)] = 1
    
    # Let's try to find a lower rank solution.
    # 
    # For n=m=p=1, rank is 1.
    # For n=m=p=2, Strassen gives rank 7.
    # For n=m=p=4, the best known is around 49 (7^2 from two levels of Strassen).
    
    # For 4x4x8, we can potentially use a Strassen-like approach.
    # Let's try to construct a solution using block matrix multiplication.
    
    # Split B into two 4x4 blocks: B = [B1 | B2] where B1 is cols 0-3, B2 is cols 4-7.
    # Then C = A * B = [A*B1 | A*B2]
    # 
    # If we have a rank-r decomposition for 4x4x4, we can use it twice:
    # One for A*B1 and one for A*B2.
    # This gives rank 2r.
    # 
    # For 4x4x4 using two levels of Strassen: rank = 7^2 = 49.
    # So for 4x4x8: rank = 98.
    
    # Let's implement the 2x2 Strassen decomposition first, then extend to 4x4.
    
    def get_strassen_2x2():
        """Return U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7)."""
        # Strassen's 7 products for 2x2 matrices:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # U indices: i*2 + j for i,j in {0,1}
        # V indices: j*2 + k for j,k in {0,1}
        # W indices: k*2 + i for k,i in {0,1} (trace convention)
        
        U = [[0]*7 for _ in range(4)]  # 2x2 = 4 rows
        V = [[0]*7 for _ in range(4)]  # 2x2 = 4 rows
        W = [[0]*7 for _ in range(4)]  # 2x2 = 4 rows
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11: i=0,j=0 -> U[0,0], A22: i=1,j=1 -> U[3,0]
        # B11: j=0,k=0 -> V[0,0], B22: j=1,k=1 -> V[3,0]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C11 gets +M1, C22 gets +M1
        # C11: k=0,i=0 -> W[0,0], C22: k=1,i=1 -> W[3,0]
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21 + A22)B11
        # A21: i=1,j=0 -> U[2,1], A22: i=1,j=1 -> U[3,1]
        # B11: j=0,k=0 -> V[0,1]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21: k=0,i=1 -> W[2,1]
        W[2][1] = 1
        
        # M3 = A11(B12 - B22)
        # A11: i=0,j=0 -> U[0,2]
        # B12: j=0,k=1 -> V[1,2], B22: j=1,k=1 -> V[3,2]
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C12: k=1,i=0 -> W[1,2], C22: k=1,i=1 -> W[3,2]
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A22(B21 - B11)
        # A22: i=1,j=1 -> U[3,3]
        # B21: j=1,k=0 -> V[2,3], B11: j=0,k=0 -> V[0,3]
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C11: k=0,i=0 -> W[0,3], C21: k=0,i=1 -> W[2,3]
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A11 + A12)B22
        # A11: i=0,j=0 -> U[0,4], A12: i=0,j=1 -> U[1,4]
        # B22: j=1,k=1 -> V[3,4]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C11: k=0,i=0 -> W[0,4], C12: k=1,i=0 -> W[1,4]
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A21 - A11)(B11 + B12)
        # A21: i=1,j=0 -> U[2,5], A11: i=0,j=0 -> U[0,5]
        # B11: j=0,k=0 -> V[0,5], B12: j=0,k=1 -> V[1,5]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C22: k=1,i=1 -> W[3,5]
        W[3][5] = 1
        
        # M7 = (A12 - A22)(B21 + B22)
        # A12: i=0,j=1 -> U[1,6], A22: i=1,j=1 -> U[3,6]
        # B21: j=1,k=0 -> V[2,6], B22: j=1,k=1 -> V[3,6]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C11: k=0,i=0 -> W[0,6], C22: k=1,i=1 -> W[3,6]
        W[0][6] = 1; W[3][6] = -1
        
        return U, V, W
    
    def combine_strassen(U1, V1, W1, U2, V2, W2):
        """Combine two 2x2x2 Strassen decompositions to get a 4x4x4 decomposition.
        
        This uses the fact that if we have rank-r decompositions for two 2x2 multiplications,
        we can combine them to get a 4x4 multiplication with rank 2r... but actually,
        we need to be more careful.
        
        For 4x4x4, we can think of it as 2x2 blocks where each element is 2x2.
        The Strassen algorithm applies at the block level, giving 7 block multiplications,
        each of which is a 2x2 multiplication with rank 7.
        Total rank = 7 * 7 = 49.
        """
        # U1, V1, W1 are for 2x2x2 with 4 rows each and 7 columns.
        # We need to create U4, V4, W4 for 4x4x4 with 16 rows each and 49 columns.
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        # Strassen at the 4x4 level (2x2 blocks of 2x2):
        # The 7 Strassen formulas again, but now each "element" is a 2x2 matrix.
        
        # For each Strassen product s (0-6), and for each sub-product t (0-6):
        # The combined index is s*7 + t
        
        for s in range(7):
            for t in range(7):
                col = s * 7 + t
                # U1[i1*2+j1, s] gives the coefficient for the s-th Strassen product
                # in the first (block) level.
                # U1[i2*2+j2, t] gives the coefficient for the t-th Strassen product
                # in the second (element) level.
                
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                # Global i = i1*2 + i2, global j = j1*2 + j2
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                U4[i*4 + j][col] += U1[i1*2 + j1][s] * U1[i2*2 + j2][t]
                
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                V4[j*4 + k][col] += V1[j1*2 + k1][s] * V1[j2*2 + k2][t]
                
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                W4[k*4 + i][col] += W1[k1*2 + i1][s] * W1[k2*2 + i2][t]
        
        return U4, V4, W4
    
    # Get the 2x2x2 Strassen decomposition
    U2, V2, W2 = get_strassen_2x2()
    
    # Get the 4x4x4 decomposition (rank 49)
    U4, V4, W4 = combine_strassen(U2, V2, W2, U2, V2, W2)
    
    # Now for 4x4x8: split B into two 4x4 blocks
    # Rank = 98
    U = [[0]*98 for _ in range(16)]  # n*m = 4*4 = 16
    V = [[0]*98 for _ in range(32)]  # m*p = 4*8 = 32
    W = [[0]*98 for _ in range(32)]  # p*n = 8*4 = 32
    
    # First half (cols 0-48): multiply A by B1 (first 4 columns of B)
    for i in range(16):
        for r in range(49):
            U[i][r] = U4[i][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*8 + k][r] = V4[j*4 + k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k*4 + i][r] = W4[k*4 + i][r]
    
    # Second half (cols 49-97): multiply A by B2 (last 4 columns of B)
    for i in range(16):
        for r in range(49):
            U[i][r + 49] = U4[i][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*8 + (k + 4)][r + 49] = V4[j*4 + k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[(k + 4)*4 + i][r + 49] = W4[k*4 + i][r]
    
    # Verify the decomposition
    if time.time() < deadline - 10:
        # Check a few random entries
        import random
        random.seed(42)
        correct = True
        for _ in range(100):
            i = random.randint(0, 3)
            j = random.randint(0, 3)
            k = random.randint(0, 7)
            # T[i*m+j, j*p+k, k*n+i] should be 1
            a = i * 4 + j
            b = j * 8 + k
            c = k * 4 + i
            val = 0
            for r in range(98):
                val += U[a][r] * V[b][r] * W[c][r]
            if val != 1:
                correct = False
                break
            # Check a non-matching entry
            i2 = (i + 1) % 4
            a2 = i2 * 4 + j
            c2 = k * 4 + i2
            val2 = 0
            for r in range(98):
                val2 += U[a2][r] * V[b][r] * W[c2][r]
            if val2 != 0:
                correct = False
                break
        if not correct:
            # Fall back to schoolbook
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
