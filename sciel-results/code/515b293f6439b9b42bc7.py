"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition: rank = n*m*p = 4*4*7 = 112
    rank_schoolbook = n * m * p
    U = np.zeros((n*m, rank_schoolbook))
    V = np.zeros((m*p, rank_schoolbook))
    W = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using structured approaches
    
    # Approach 1: Winograd minimal filtering for 1x4x7 convolutions
    # Treat each row of the 4x4 matrix as a 1x4 filter convolved with 1x7 signal
    # This gives us a way to potentially reduce rank
    
    # For a 1x4x1 convolution (4-tap filter), Winograd gives rank 3 instead of 4
    # For 1x4x7, we can think of it as 4 parallel 1x4x1 convolutions
    
    # Let's try a tensor product / Kronecker structure approach
    # 4 = 2*2, so we can use Strassen's algorithm for 2x2 blocks
    
    # Strassen 2x2x2 has rank 7
    # For 4x4x4: using tensor product of Strassen with itself gives 7*7=49
    # For 4x4x7: we need to handle the 7 dimension
    
    # Let's try: decompose 4x4x7 as (2x2x7) ⊗ (2x2) in some sense
    # Actually, let's think of it as: C = A*B where A is 4x4, B is 4x7
    
    # We can split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # And B into 2x7 blocks: B = [[B1], [B2]] where B1, B2 are 2x7
    
    # C = A*B = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    # Each 2x2 times 2x7 multiplication can use Strassen-like methods
    # For 2x2 * 2x7: we can use 7 multiplications per 2x2 block pair
    # Actually, 2x2 * 2x7 is like 7 separate 2x2 * 2x1 multiplications
    # Each 2x2 * 2x1 can be done in... let me think
    
    # 2x2 * 2x1: C = A*B where A is 2x2, B is 2x1
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This needs 4 multiplications (schoolbook)
    # Can we do better? The tensor is 2x2x2 = 8 entries, rank at most 8
    # Actually this is just a 2x2 matrix times a vector, which is a bilinear map
    # The rank of this tensor... let me compute it
    
    # For 2x2 * 2x1: the tensor T has T[i,j,k] = 1 if k corresponds to output i
    # Wait, let me be more careful.
    
    # Let me try a different approach: use the fact that we can search for
    # low-rank decompositions with rational/half-integer entries
    
    # Approach: Try to find a decomposition of rank r < 112
    # The tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's try rank 49 (from 7x7 Strassen tensor product for the 4x4 part,
    # then handle the 7 columns)
    
    # Actually, let me try a more systematic search
    # For small ranks, we can try to solve the system of equations
    
    # The condition is: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (a,b,c) not of the form (i*m+j, j*p+k, k*n+i):
    # sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    # This is a system of polynomial equations in the entries of U, V, W
    
    # Let's try a greedy/iterative approach:
    # Start with schoolbook, then try to merge terms
    
    # Actually, let's try a structured approach based on Winograd
    
    # For 1D convolution of length 4 with length 7:
    # Winograd F(4, 4) for 4-point convolution uses 4 multiplications (vs 4 schoolbook)
    # Wait, F(m, n) for m-point filter and n-point signal uses m+n-1 multiplications
    # F(4, 4) uses 4+4-1 = 7? No, that's not right either.
    
    # Let me recall: Winograd's F(m, n) transforms a length-m filter and length-n signal
    # into a pointwise multiplication of length m+n-1.
    # For m=4, n=1 (single output): F(4, 1) uses 4 multiplications (same as schoolbook)
    # For m=4, n=4 (4 outputs): F(4, 4) uses 4+4-1 = 7 multiplications (vs 16 schoolbook)
    
    # Hmm, but our problem is 4x4 * 4x7, which is different.
    
    # Let me try yet another approach: use known bounds
    # The rank of the matrix multiplication tensor <n,m,p> is at most n*m*p
    # For <4,4,4>, the best known upper bound is 49 (from 7^2 using Strassen)
    # For <4,4,7>, we can use <4,4,4> * <4,4,7> structure
    
    # Actually, <4,4,7> can be seen as 7 copies of <4,4,1>
    # <4,4,1> has rank at most 16 (schoolbook)
    # But <4,4,1> is the tensor for multiplying 4x4 by 4x1
    # This is equivalent to 4 separate <1,4,1> multiplications... no
    
    # Let me just try to verify the schoolbook and return it, but also attempt
    # a simple improvement by checking if rank 49 works
    
    # For now, let's try a random search for low-rank decompositions
    # with small integer/half-integer entries
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try to verify schoolbook
    def verify(U_arr, V_arr, W_arr):
        """Verify that the decomposition is correct."""
        r = U_arr.shape[1]
        T = np.zeros((n*m, m*p, p*n))
        for ri in range(r):
            T += np.einsum('a,b,c->abc', U_arr[:, ri], V_arr[:, ri], W_arr[:, ri])
        
        # Check that T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i*m+j, j*p+k, k*n+i] - 1) > 1e-9:
                        return False
        
        # Check that T[a,b,c] = 0 for all other (a,b,c)
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Check if (a,b,c) corresponds to a valid (i,j,k)
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and i == i2 and k == k2:
                        continue  # This is a valid position, already checked
                    if abs(T[a, b, c]) > 1e-9:
                        return False
        return True
    
    # Verify schoolbook
    if verify(U, V, W):
        pass  # Schoolbook is valid
    
    # Try to find lower rank using a simple structured approach
    # Based on the idea that 4x4x7 can be decomposed using tensor products
    
    # Strassen 2x2x2: rank 7
    # U_2x2, V_2x2, W_2x2 for 2x2x2 multiplication
    
    # Let me define Strassen's 7-term decomposition for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me encode this
    # For 2x2x2: n=2, m=2, p=2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    
    # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    # B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
    # C is 2x2: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # U[i*m+j, r] where i,j are A indices
    # V[j*p+k, r] where j,k are B indices  
    # W[k*n+i, r] where k,i are C indices (trace convention)
    
    # For Strassen:
    # Term 1: (A00+A11)(B00+B11) contributes to C00 and C11
    #   U: A00=1, A11=1 -> U[0,0]=1, U[3,0]=1
    #   V: B00=1, B11=1 -> V[0,0]=1, V[3,0]=1
    #   W: C00=1, C11=1 -> W[0,0]=1, W[3,0]=1
    
    # Term 2: (A10+A11)B00 contributes to C10 and C11
    #   U: A10=1, A11=1 -> U[2,0]=1, U[3,0]=1
    #   V: B00=1 -> V[0,0]=1
    #   W: C10=-1, C11=1 -> W[2,0]=-1, W[3,0]=1
    
    # Wait, I need to be more careful. Let me use the standard Strassen formulation.
    
    # Actually, let me just try to construct a 4x4x4 decomposition of rank 49
    # by taking the tensor product of two 2x2x2 Strassen decompositions
    
    # For 4x4x4:
    # A is 4x4, B is 4x4, C is 4x4
    # We can write A = A1 ⊗ A2 where A1, A2 are 2x2
    # Similarly B = B1 ⊗ B2, C = C1 ⊗ C2
    
    # Then C = A*B = (A1*A1')*(B1*B1') where the primes denote the other block
    # Actually, this is more subtle. The tensor product structure gives:
    # T_4x4x4 = T_2x2x2 ⊗ T_2x2x2
    # Rank(T_4x4x4) ≤ Rank(T_2x2x2)^2 = 49
    
    # Let me construct this explicitly
    # For 2x2x2 Strassen, we have U2, V2, W2 of shape (4,7)
    
    # For 4x4x4, the tensor product gives U4, V4, W4 of shape (16, 49)
    # U4[(i1*m1+j1)*m2+j2, r1*7+r2] = U2[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
    # Wait, I need to be careful about the indexing
    
    # For 2x2x2: n=2, m=2, p=2
    # U2 has shape (n*m, 7) = (4, 7)
    # Index: U2[i*2+j, r] where i,j in {0,1}
    
    # For 4x4x4: n=4, m=4, p=4
    # U4 has shape (16, 49)
    # Index: U4[i*4+j, r] where i,j in {0,1,2,3}
    # We can write i = i1*2+i2, j = j1*2+j2 where i1,i2,j1,j2 in {0,1}
    # Then U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    
    # Let me first define the Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matrix multiplication with rank 7."""
        n2, m2, p2 = 2, 2, 2
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # Strassen's 7 multiplications:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00  
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 - M5 + M4 + M7
        # C01 = M5 + M3
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M3
        
        # U2[i*2+j, r]: coefficient for A[i,j] in term r
        # V2[j*2+k, r]: coefficient for B[j,k] in term r
        # W2[k*2+i, r]: coefficient for C[i,k] in term r (trace convention: W[k*n+i, r])
        
        # Term 0 (M1): (A00+A11)(B00+B11)
        U2[0, 0] = 1  # A00
        U2[3, 0] = 1  # A11
        V2[0, 0] = 1  # B00
        V2[3, 0] = 1  # B11
        W2[0, 0] = 1  # C00 (k=0, i=0 -> k*2+i=0)
        W2[3, 0] = 1  # C11 (k=1, i=1 -> k*2+i=3)
        
        # Term 1 (M2): (A10+A11)B00
        U2[2, 1] = 1  # A10
        U2[3, 1] = 1  # A11
        V2[0, 1] = 1  # B00
        W2[2, 1] = 1  # C10 (k=0, i=1 -> k*2+i=2)
        W2[3, 1] = -1  # C11 (k=1, i=1 -> k*2+i=3)
        
        # Term 2 (M3): A00(B10-B11)
        U2[0, 2] = 1  # A00
        V2[2, 2] = 1  # B10
        V2[3, 2] = -1  # B11
        W2[1, 2] = 1  # C01 (k=1, i=0 -> k*2+i=1)
        W2[3, 2] = 1  # C11 (k=1, i=1 -> k*2+i=3)
        
        # Term 3 (M4): A11(B01-B10)
        U2[3, 3] = 1  # A11
        V2[1, 3] = 1  # B01
        V2[2, 3] = -1  # B10
        W2[0, 3] = 1  # C00 (k=0, i=0 -> k*2+i=0)
        W2[2, 3] = 1  # C10 (k=0, i=1 -> k*2+i=2)
        
        # Term 4 (M5): (A00+A01)B11
        U2[0, 4] = 1  # A00
        U2[1, 4] = 1  # A01
        V2[3, 4] = 1  # B11
        W2[0, 4] = -1  # C00 (k=0, i=0 -> k*2+i=0)
        W2[1, 4] = 1  # C01 (k=1, i=0 -> k*2+i=1)
        
        # Term 5 (M6): (A10-A00)(B00+B10)
        U2[0, 5] = -1  # A00
        U2[2, 5] = 1  # A10
        V2[0, 5] = 1  # B00
        V2[2, 5] = 1  # B10
        W2[3, 5] = 1  # C11 (k=1, i=1 -> k*2+i=3)
        
        # Term 6 (M7): (A01-A11)(B01+B11)
        U2[1, 6] = 1  # A01
        U2[3, 6] = -1  # A11
        V2[1, 6] = 1  # B01
        V2[3, 6] = 1  # B11
        W2[0, 6] = 1  # C00 (k=0, i=0 -> k*2+i=0)
        
        return U2, V2, W2
    
    # Verify Strassen 2x2x2
    U2, V2, W2 = strassen_2x2x2()
    
    def verify_2x2x2(U_arr, V_arr, W_arr):
        n2, m2, p2 = 2, 2, 2
        T = np.zeros((4, 4, 4))
        for r in range(7):
            T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
        
        # Check T[i*2+j, j*2+k, k*2+i] = 1 for all i,j,k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if abs(T[i*2+j, j*2+k, k*2+i] - 1) > 1e-9:
                        return False
        
        # Check T[a,b,c] = 0 otherwise
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    i = a // 2
                    j = a % 2
                    j2 = b // 2
                    k = b % 2
                    k2 = c // 2
                    i2 = c % 2
                    if j == j2 and i == i2 and k == k2:
                        continue
                    if abs(T[a, b, c]) > 1e-9:
                        return False
        return True
    
    strassen_valid = verify_2x2x2(U2, V2, W2)
    
    # Now build 4x4x4 from tensor product of two Strassen decompositions
    if strassen_valid:
        U4_4x4 = np.zeros((16, 49))
        V4_4x4 = np.zeros((16, 49))
        W4_4x4 = np.zeros((16, 49))
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                a = (i1*2+i2)*4 + (j1*2+j2)
                                U4_4x4[a, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
                
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                b = (j1*2+j2)*4 + (k1*2+k2)
                                V4_4x4[b, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
                
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                c = (k1*2+k2)*4 + (i1*2+i2)
                                W4_4x4[c, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
        
        # Verify 4x4x4
        def verify_4x4x4(U_arr, V_arr, W_arr):
            n4, m4, p4 = 4, 4, 4
            T = np.zeros((16, 16, 16))
            for r in range(49):
                T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
            
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        if abs(T[i*4+j, j*4+k, k*4+i] - 1) > 1e-9:
                            return False
            
            for a in range(16):
                for b in range(16):
                    for c in range(16):
                        i = a // 4
                        j = a % 4
                        j2 = b // 4
                        k = b % 4
                        k2 = c // 4
                        i2 = c % 4
                        if j == j2 and i == i2 and k == k2:
                            continue
                        if abs(T[a, b, c]) > 1e-9:
                            return False
            return True
        
        if verify_4x4x4(U4_4x4, V4_4x4, W4_4x4):
            # Now extend to 4x4x7
            # We have 4x4x4 with rank 49
            # For 4x4x7, we can think of B as 4x7 = [B_left, B_right] where 
            # B_left is 4x4 and B_right is 4x3
            # C = A*B = A*B_left + A*B_right
            # This doesn't directly help since we need a single decomposition
            
            # Alternative: use 4x4x4 for the first 4 columns, and add 3 more
            # For the last 3 columns, we need additional terms
            # Each additional column of B (4x1) requires at most 16 multiplications
            # So total rank would be 49 + 3*16 = 97, which is worse than 112? No, 97 < 112
            
            # Actually, let's think differently.
            # 4x4x7: A is 4x4, B is 4x7
            # We can split B into B1 (4x4) and B2 (4x3)
            # C = A*B1 + A*B2
            # A*B1 uses rank 49
            # A*B2: A is 4x4, B2 is 4x3
            # For 4x4x3: schoolbook rank is 4*4*3 = 48
            # Total: 49 + 48 = 97
            
            # But we can do better. Let's use the tensor product structure.
            # Actually, 4x4x7 can be viewed as:
            # For each column k of B (k=0..6), we compute A * B[:,k]
            # This is 7 separate 4x4 * 4x1 multiplications
            # Each 4x4 * 4x1 has rank at most 16 (schoolbook)
            # Total: 7*16 = 112 (same as schoolbook)
            
            # But using Strassen for 4x4x4, we get rank 49 for 4 columns
            # For the remaining 3 columns, we need rank at most 3*16 = 48
            # Total: 49 + 48 = 97
            
            # Let's construct this
            # First, the 4x4x4 part (columns 0-3 of B)
            # U4_4x4 has shape (16, 49), V4_4x4 has shape (16, 49), W4_4x4 has shape (16, 49)
            
            # For 4x4x7:
            # U: (16, rank_total), V: (28, rank_total), W: (28, rank_total)
            # Wait, V has shape (m*p, rank) = (4*7, rank) = (28, rank)
            # W has shape (p*n, rank) = (7*4, rank) = (28, rank)
            
            # For the first 4 columns of B (indices 0-3):
            # V4_4x4 has shape (16, 49) corresponding to B indices (j*4+k) for k=0..3
            # We need to embed this into V of shape (28, rank)
            # V[j*7+k, r] for the 4x4x4 part: V[j*7+k, r] = V4_4x4[j*4+k, r] for k=0..3
            
            # For the last 3 columns (indices 4-6):
            # We need to compute A * B[:,4:7] where B[:,4:7] is 4x3
            # Schoolbook for 4x4x3: rank 48
            # U_part2: (16, 48), V_part2: (28, 48), W_part2: (28, 48)
            
            rank_part2 = 4 * 4 * 3  # 48
            U_part2 = np.zeros((16, rank_part2))
            V_part2 = np.zeros((28, rank_part2))
            W_part2 = np.zeros((28, rank_part2))
            
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4, 7):  # Columns 4, 5, 6
                        U_part2[i*4+j, col] = 1
                        V_part2[j*7+k, col] = 1
                        W_part2[k*4+i, col] = 1
                        col += 1
            
            # Combine
            rank_total = 49 + rank_part2  # 97
            U_total = np.zeros((16, rank_total))
            V_total = np.zeros((28, rank_total))
            W_total = np.zeros((28, rank_total))
            
            # First 49 terms: 4x4x4 part
            U_total[:, :49] = U4_4x4
            # V4_4x4 has shape (16, 49) with indices j*4+k for k=0..3
            # We need V_total[j*7+k, r] = V4_4x4[j*4+k, r] for k=0..3
            for j in range(4):
                for k in range(4):
                    V_total[j*7+k, :49] = V4_4x4[j*4+k, :49]
            # W4_4x4 has shape (16, 49) with indices k*4+i for k=0..3
            # We need W_total[k*4+i, r] = W4_4x4[k*4+i, r] for k=0..3
            for k in range(4):
                for i in range(4):
                    W_total[k*4+i, :49] = W4_4x4[k*4+i, :49]
            
            # Last 48 terms: 4x4x3 part
            U_total[:, 49:] = U_part2
            V_total[:, 49:] = V_part2
            W_total[:, 49:] = W_part2
            
            # Verify 4x4x7
            def verify_4x4x7(U_arr, V_arr, W_arr):
                n7, m7, p7 = 4, 4, 7
                T = np.zeros((16, 28, 28))
                r = U_arr.shape[1]
                for ri in range(r):
                    T += np.einsum('a,b,c->abc', U_arr[:, ri], V_arr[:, ri], W_arr[:, ri])
                
                for i in range(n7):
                    for j in range(m7):
                        for k in range(p7):
                            if abs(T[i*4+j, j*7+k, k*4+i] - 1) > 1e-9:
                                return False
                
                for a in range(16):
                    for b in range(28):
                        for c in range(28):
                            i = a // 4
                            j = a % 4
                            j2 = b // 7
                            k = b % 7
                            k2 = c // 4
                            i2 = c % 4
                            if j == j2 and i == i2 and k == k2:
                                continue
                            if abs(T[a, b, c]) > 1e-9:
                                return False
                return True
            
            if verify_4x4x7(U_total, V_total, W_total):
                best_rank = rank_total
                best_U, best_V, best_W = U_total, V_total, W_total
                print(f"Found 4x4x7 decomposition with rank {rank_total}")
    
    # Return the best result
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
