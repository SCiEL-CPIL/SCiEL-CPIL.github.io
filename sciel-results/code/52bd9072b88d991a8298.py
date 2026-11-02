"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    def is_valid(U, V, W, rank):
        """Check if U, V, W form a valid tensor decomposition using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
        # Use einsum: for each c, outer product of reshaped vectors
        T = np.zeros((n, m, p))
        for c in range(rank):
            Uc = U_arr[:, c].reshape(n, m)
            Vc = V_arr[:, c].reshape(m, p)
            Wc = W_arr[:, c].reshape(p, n)
            # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
            # This is a tensor contraction
            T += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
        
        return np.allclose(T, 1.0, atol=1e-9)
    
    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def strassen_2x2():
        """
        Strassen decomposition for 2x2 x 2x2 -> 2x2 matrix multiplication.
        Returns U, V, W with rank 7 for the 2x2x2x2 multiplication tensor.
        
        For A (2x2), B (2x2), C = A*B:
        A = [[A11, A12], [A21, A22]]
        B = [[B11, B12], [B21, B22]]
        
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B12-B22)
        M4 = A22(B21-B11)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        Here U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        where indices are: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k]
        """
        # Each M_c is a product of two 2x2 matrices.
        # For each c, we need to express:
        #   sum_{i,j} U[i,j] * (linear combo of A) * sum_{j,k} V[j,k] * (linear combo of B) 
        #   contributes to C[i,k] with coefficient W[k,i]
        
        # Actually, the Strassen decomposition works as follows:
        # For each term M_c = X_c * Y_c where X_c is a linear combination of A entries
        # and Y_c is a linear combination of B entries.
        # Then C = sum_c (contribution of M_c to each C[i,k])
        
        # Let me define the 7 products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For each M_c, X_c is a 2x2 matrix (linear combo of A) and Y_c is a 2x2 matrix (linear combo of B).
        # The contribution to C is: C += W_coeff * X_c * Y_c where W_coeff relates to how M_c contributes to each C[i,k].
        
        # Actually, in the tensor decomposition framework:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # where i,j are input indices for A, j,k for B, k,i for C.
        
        # For Strassen, each "multiplication" M_c involves a specific 2x2 matrix X_c (from A)
        # and Y_c (from B), and contributes to C with a specific pattern.
        
        # Let me think of it differently. The Strassen algorithm computes:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M_c = (linear combo of A entries) * (linear combo of B entries)
        # But each M_c is a scalar product of two 2x2 matrices? No, each M_c is a 2x2 matrix product.
        
        # Wait, I think I'm overcomplicating this. In the standard Strassen algorithm,
        # the 7 "multiplications" are actually products of 2x2 matrices, not scalars.
        # So M1 = (A11+A22) * (B11+B22) is a 2x2 matrix product.
        
        # To decompose this into the tensor form, each 2x2 matrix product itself needs a decomposition.
        # But that would give rank 7*4 = 28, not 7.
        
        # Actually, the key insight is that in the tensor decomposition of matrix multiplication,
        # the rank refers to the number of scalar multiplications needed. For 2x2x2x2,
        # the rank is 7 because Strassen reduces it from 8 scalar multiplications to 7.
        
        # But wait, for 2x2 matrices, each entry of A and B is a scalar, so we have 4x4=16 scalar products
        # in the schoolbook, and Strassen reduces this to 7 scalar multiplications of larger expressions.
        
        # Hmm, I think the confusion is that for 2x2x2x2, the "rank" in tensor decomposition
        # refers to the bilinear complexity, which is 7 for Strassen.
        
        # Let me just construct the explicit U, V, W matrices for the 2x2 case.
        
        # The multiplication tensor T for 2x2 x 2x2 -> 2x2 has T[i,j,k] = 1 for all i,j,k in {0,1}
        # (since C[i,k] = sum_j A[i,j] B[j,k]).
        
        # We need U (4x7), V (4x7), W (4x7) such that:
        # sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all i,j,k
        
        # Using Strassen's 7 products:
        # Let me denote the 7 intermediate products as m1,...,m7.
        # Each m_c is formed from specific linear combinations of A and B entries.
        
        # Actually, let me look up the explicit Strassen tensor decomposition.
        # The 7 rank-1 tensors for 2x2x2x2 are:
        
        # For each c in 1..7, we have vectors u_c (in R^4 for A), v_c (in R^4 for B), w_c (in R^4 for C)
        # such that T = sum_c u_c ⊗ v_c ⊗ w_c
        
        # Standard Strassen:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        # Each m_c is a scalar (product of two linear combinations).
        # So for m1: u1 has a11 and a22 set to 1, v1 has b11 and b22 set to 1.
        # m1 contributes to c11 with coeff +1, c22 with coeff +1.
        
        # So w1 has c11 and c22 set to 1.
        
        # Let me construct this:
        # Index convention: A[i,j] at position i*2+j, B[j,k] at position j*2+k, C[i,k] at position i*2+k
        # But the task uses W[k*n+i, r] for the output, so C[i,k] is at position k*2+i.
        
        # For 2x2: U shape (4,7), V shape (4,7), W shape (4,7)
        # U[i*2+j, c] = coefficient of A[i,j] in the c-th Strassen product
        # V[j*2+k, c] = coefficient of B[j,k] in the c-th Strassen product
        # W[k*2+i, c] = coefficient of C[i,k] in the c-th Strassen product (note: C[i,k] at pos k*2+i)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1 = (a11+a22)(b11+b22)
        # U: a11 (pos 0) = 1, a22 (pos 3) = 1
        # V: b11 (pos 0) = 1, b22 (pos 3) = 1
        # W: c11 (pos 0*2+0=0) = 1, c22 (pos 1*2+1=3) = 1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (a21+a22)b11
        # U: a21 (pos 2) = 1, a22 (pos 3) = 1
        # V: b11 (pos 0) = 1
        # W: c21 (pos 0*2+1=1) = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # m3 = a11(b12-b22)
        # U: a11 (pos 0) = 1
        # V: b12 (pos 1) = 1, b22 (pos 3) = -1
        # W: c12 (pos 0*2+1=1) = 1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # m4 = a22(b21-b11)
        # U: a22 (pos 3) = 1
        # V: b21 (pos 2) = 1, b11 (pos 0) = -1
        # W: c11 (pos 0) = 1, c21 (pos 1) = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # m5 = (a11+a12)b22
        # U: a11 (pos 0) = 1, a12 (pos 1) = 1
        # V: b22 (pos 3) = 1
        # W: c11 (pos 0) = -1, c12 (pos 1) = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # m6 = (a21-a11)(b11+b12)
        # U: a21 (pos 2) = 1, a11 (pos 0) = -1
        # V: b11 (pos 0) = 1, b12 (pos 1) = 1
        # W: c22 (pos 3) = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # m7 = (a12-a22)(b21+b22)
        # U: a12 (pos 1) = 1, a22 (pos 3) = -1
        # V: b21 (pos 2) = 1, b22 (pos 3) = 1
        # W: c11 (pos 0) = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        # Verify: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should be 1 for all i,j,k
        # Let's verify numerically
        T = np.zeros((2, 2, 2))
        for c in range(7):
            Uc = U[:, c].reshape(2, 2)
            Vc = V[:, c].reshape(2, 2)
            Wc = W[:, c].reshape(2, 2)
            T += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
        
        if not np.allclose(T, 1.0, atol=1e-10):
            # Fallback: use schoolbook for 2x2
            U = np.zeros((4, 4))
            V = np.zeros((4, 4))
            W = np.zeros((4, 4))
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U[i*2+j, col] = 1
                        V[j*2+k, col] = 1
                        W[k*2+i, col] = 1
                        col += 1
            return U.tolist(), V.tolist(), W.tolist(), 4
        
        return U.tolist(), V.tolist(), W.tolist(), 7
    
    def strassen_2x2x3():
        """
        Decomposition for 2x2 x 2x3 -> 2x3 matrix multiplication.
        This can be seen as two 2x2 x 2x2 multiplications (for the first two columns)
        plus one 2x2 x 2x1 multiplication (for the third column).
        
        Actually, let's think of it as:
        A is 2x2, B is 2x3, C is 2x3.
        C[:,0:2] = A * B[:,0:2] (2x2 x 2x2 -> 2x2)
        C[:,2] = A * B[:,2] (2x2 x 2x1 -> 2x1)
        
        For the 2x2 x 2x2 part, we can use Strassen (rank 7).
        For the 2x2 x 2x1 part, we need rank 4 (schoolbook) since there's no known better.
        Actually, 2x2 x 2x1 has rank 4 (it's just 4 scalar products).
        
        So total rank would be 7 + 4 = 11.
        But we can do better by sharing some computations.
        
        Alternatively, we can treat the entire 2x2 x 2x3 as a single problem.
        The schoolbook rank is 2*2*3 = 12.
        
        Let me try to construct a rank-10 or rank-11 decomposition.
        
        Actually, for simplicity, let's use the block approach:
        - Use Strassen (rank 7) for the 2x2 x 2x2 part (columns 0,1 of B)
        - Use schoolbook (rank 4) for the 2x2 x 2x1 part (column 2 of B)
        - Total rank: 7 + 4 = 11
        
        But wait, the Strassen decomposition for 2x2 x 2x2 gives us the first two columns of C.
        For the third column, we need 4 more rank-1 terms.
        
        So U will have shape (4, 11), V will have shape (6, 11), W will have shape (6, 11).
        
        Let me construct this.
        """
        # Get Strassen for 2x2 x 2x2
        U_s, V_s, W_s, rank_s = strassen_2x2()
        
        # For the third column (2x2 x 2x1 -> 2x1):
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # Schoolbook for this part (4 terms):
        # Term 0: A[0,0] * B[0,2] -> C[0,2]
        # Term 1: A[0,1] * B[1,2] -> C[0,2]
        # Term 2: A[1,0] * B[0,2] -> C[1,2]
        # Term 3: A[1,1] * B[1,2] -> C[1,2]
        
        total_rank = rank_s + 4
        U = np.zeros((4, total_rank))
        V = np.zeros((6, total_rank))  # B is 2x3, so 6 entries
        W = np.zeros((6, total_rank))  # C is 2x3, so 6 entries
        
        # Copy Strassen part (first 7 columns)
        U[:, :rank_s] = np.array(U_s)
        V[:4, :rank_s] = np.array(V_s)  # First 4 entries of B (2x2 part)
        # V[4:6, :rank_s] = 0 (third column of B not used in Strassen part)
        W[:4, :rank_s] = np.array(W_s)  # First 4 entries of C (2x2 part)
        # W[4:6, :rank_s] = 0 (third column of C not used in Strassen part)
        
        # Add schoolbook for third column
        base = rank_s
        # Term 0: A[0,0] * B[0,2] -> C[0,2]
        U[0, base+0] = 1
        V[4, base+0] = 1  # B[0,2] is at position 0*3+2 = 2... wait
        # B is 2x3, so B[j,k] is at position j*3+k
        # B[0,2] is at position 0*3+2 = 2
        # B[1,2] is at position 1*3+2 = 5
        # C is 2x3, so C[i,k] is at position k*2+i (per the task convention W[k*n+i])
        # C[0,2] is at position 2*2+0 = 4
        # C[1,2] is at position 2*2+1 = 5
        
        # Let me redo with correct indexing:
        # V[j*p+k] where p=3, so V[j*3+k]
        # W[k*n+i] where n=2, so W[k*2+i]
        
        # Term 0: A[0,0] * B[0,2] -> C[0,2]
        U[0*2+0, base+0] = 1  # A[0,0]
        V[0*3+2, base+0] = 1  # B[0,2]
        W[2*2+0, base+0] = 1  # C[0,2]
        
        # Term 1: A[0,1] * B[1,2] -> C[0,2]
        U[0*2+1, base+1] = 1  # A[0,1]
        V[1*3+2, base+1] = 1  # B[1,2]
        W[2*2+0, base+1] = 1  # C[0,2]
        
        # Term 2: A[1,0] * B[0,2] -> C[1,2]
        U[1*2+0, base+2] = 1  # A[1,0]
        V[0*3+2, base+2] = 1  # B[0,2]
        W[2*2+1, base+2] = 1  # C[1,2]
        
        # Term 3: A[1,1] * B[1,2] -> C[1,2]
        U[1*2+1, base+3] = 1  # A[1,1]
        V[1*3+2, base+3] = 1  # B[1,2]
        W[2*2+1, base+3] = 1  # C[1,2]
        
        # Verify
        T = np.zeros((2, 2, 3))
        for c in range(total_rank):
            Uc = U[:, c].reshape(2, 2)
            Vc = V[:, c].reshape(2, 3)
            Wc = W[:, c].reshape(3, 2)
            T += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
        
        if not np.allclose(T, 1.0, atol=1e-10):
            # Fallback to schoolbook for 2x2x3
            U = np.zeros((4, 12))
            V = np.zeros((6, 12))
            W = np.zeros((6, 12))
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U[i*2+j, col] = 1
                        V[j*3+k, col] = 1
                        W[k*2+i, col] = 1
                        col += 1
            return U.tolist(), V.tolist(), W.tolist(), 12
        
        return U.tolist(), V.tolist(), W.tolist(), total_rank
    
    # Start with schoolbook
    school = schoolbook()
    best_result = school
    best_rank = n * m * p
    
    if is_valid(school["U"], school["V"], school["W"], best_rank):
        pass  # Confirmed valid
    else:
        # Should not happen, but handle gracefully
        pass
    
    # Try block decomposition using Strassen
    # Split 4x4 x 4x6 into 2x2 blocks
    # A is 2x2 blocks of 2x2, B is 2x3 blocks of 2x2, C is 2x3 blocks of 2x2
    
    # We can use the 2x2x2x3 decomposition (rank 11) for each of the 4 block positions
    # But we need to be careful about how the blocks interact.
    
    # Actually, the block structure is:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each Bij is 2x2
    # C = [[C11, C12, C13], [C21, C22, C23]] where each Cij is 2x2
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # Each Cij is a sum of two 2x2 x 2x2 products.
    # If we use Strassen for each 2x2 x 2x2 product, we get 7 rank-1 terms per product.
    # So each Cij would need 14 rank-1 terms, giving total rank 6*14 = 84.
    # That's worse than schoolbook (96)... wait, 4*4*6 = 96.
    # 84 < 96, so this is an improvement!
    
    # But actually, we can do better by using the 2x2x2x3 decomposition directly.
    # Let me think of it as: for each row block (2 rows), we compute a 2x2 x 4x6 product.
    # Wait, that's not right either.
    
    # Let me try a different approach: use the known rank-11 decomposition for 2x2 x 2x3
    # and apply it to each of the 4 "subproblems" formed by the block structure.
    
    # Actually, the cleanest approach is:
    # Treat the multiplication as 4 independent 2x2 x 2x3 subproblems? No, that's not correct.
    
    # Let me just try to construct the block decomposition explicitly.
    # For each of the 6 output blocks Cij, we need to compute A_i * B_j where A_i is a 2x4 block
    # and B_j is a 4x2 block. But that's not a standard matrix multiplication.
    
    # I think the safest approach for now is to just return the schoolbook solution,
    # since constructing a valid low-rank decomposition is complex and error-prone.
    
    # But let's at least try the simple block approach with Strassen for each 2x2x2x2 product.
    
    # Construct the block decomposition:
    # Total rank = 6 * 14 = 84 (using Strassen for each of the 12 2x2x2x2 products)
    # Wait, there are 6 output blocks, each is a sum of 2 2x2x2x2 products, so 12 products total.
    # Each product has rank 7, so total rank = 12 * 7 = 84.
    
    U_block = np.zeros((16, 84))  # 4x4 = 16 entries
    V_block = np.zeros((24, 84))  # 4x6 = 24 entries
    W_block = np.zeros((24, 84))  # 6x4 = 24 entries
    
    col = 0
    # For each output block (ci, cj) where ci in {0,1}, cj in {0,1,2}:
    # C[ci*2:ci*2+2, cj*2:cj*2+2] = A[ci*2:ci*2+2, 0:2] * B[0:2, cj*2:cj*2+2]
    #                                 + A[ci*2:ci*2+2, 2:4] * B[2:4, cj*2:cj*2+2]
    
    U_s, V_s, W_s, rank_s = strassen_2x2()
    
    for ci in range(2):
        for cj in range(3):
            # First product: A[ci*2:ci*2+2, 0:2] * B[0:2, cj*2:cj*2+2]
            for c in range(rank_s):
                # A indices: rows ci*2 to ci*2+1, cols 0 to 1
                for i in range(2):
                    for j in range(2):
                        U_block[(ci*2+i)*4+j, col] = U_s[i*2+j, c]
                # B indices: rows 0 to 1, cols cj*2 to cj*2+1
                for j in range(2):
                    for k in range(2):
                        V_block[j*6+(cj*2+k), col] = V_s[j*2+k, c]
                # C indices: rows ci*2 to ci*2+1, cols cj*2 to cj*2+1
                # C[i,k] at position k*4+i
                for k in range(2):
                    for i in range(2):
                        W_block[(cj*2+k)*4+(ci*2+i), col] = W_s[k*2+i, c]
                col += 1
            
            # Second product: A[ci*2:ci*2+2, 2:4] * B[2:4, cj*2:cj*2+2]
            for c in range(rank_s):
                # A indices: rows ci*2 to ci*2+1, cols 2 to 3
                for i in range(2):
                    for j in range(2):
                        U_block[(ci*2+i)*4+(2+j), col] = U_s[i*2+j, c]
                # B indices: rows 2 to 3, cols cj*2 to cj*2+1
                for j in range(2):
                    for k in range(2):
                        V_block[(2+j)*6+(cj*2+k), col] = V_s[j*2+k, c]
                # C indices: rows ci*2 to ci*2+1, cols cj*2 to cj*2+1
                for k in range(2):
                    for i in range(2):
                        W_block[(cj*2+k)*4+(ci*2+i), col] = W_s[k*2+i, c]
                col += 1
    
    # Verify the block decomposition
    if is_valid(U_block.tolist(), V_block.tolist(), W_block.tolist(), 84):
        best_result = {"U": U_block.tolist(), "V": V_block.tolist(), "W": W_block.tolist()}
        best_rank = 84
    
    return best_result
# EVOLVE-BLOCK-END

