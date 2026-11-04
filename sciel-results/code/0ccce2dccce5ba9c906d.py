import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # safety margin before 150s kill

    def to_list(mat):
        """Convert numpy array to nested list with int/float entries."""
        result = []
        for row in mat:
            r = []
            for x in row:
                if abs(x - round(x)) < 1e-10:
                    r.append(int(round(x)))
                else:
                    r.append(float(x))
            result.append(r)
        return result

    def verify(U, V, W, num_trials=5):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        for trial in range(num_trials):
            A = np.random.randint(-5, 6, size=(n, m))
            B = np.random.randint(-5, 6, size=(m, p))
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for t in range(r):
                            val += A[i, j] * U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                        C_computed[i, k] += val
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    def verify_fast(U, V, W, num_trials=3):
        """Faster verification using tensor contraction."""
        r = U.shape[1]
        for trial in range(num_trials):
            A = np.random.randint(-3, 4, size=(n, m))
            B = np.random.randint(-3, 4, size=(m, p))
            C_expected = A @ B
            # C[i,k] = sum_j sum_t A[i,j] * U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
            C_computed = np.zeros((n, p))
            for t in range(r):
                # For each factor t, compute contribution
                u_t = U[:, t].reshape(n, m)
                v_t = V[:, t].reshape(m, p)
                w_t = W[:, t].reshape(p, n)
                # Contribution: sum_j A[i,j] * u_t[i,j] * v_t[j,k] * w_t[k,i]
                # = sum_j A[i,j] * u_t[i,j] * (v_t @ (w_t * ???))
                # Actually: for fixed t, contribution to C[i,k] = sum_j A[i,j]*u_t[i,j]*v_t[j,k]*w_t[k,i]
                # = (A * u_t) @ v_t * w_t[k,i] ... no, w_t depends on k and i
                # Let's do it more carefully:
                # C_t[i,k] = sum_j A[i,j] * u_t[i,j] * v_t[j,k] * w_t[k,i]
                # = sum_j (A[i,j] * u_t[i,j]) * v_t[j,k] * w_t[k,i]
                # Let D[i,j] = A[i,j] * u_t[i,j]
                # Then C_t[i,k] = sum_j D[i,j] * v_t[j,k] * w_t[k,i]
                # = (D @ v_t)[i,k] * w_t[k,i] ... no, w_t[k,i] is a scalar for each (k,i)
                # C_t[i,k] = (D @ v_t)[i,k] * w_t[k,i]
                D = A * u_t
                temp = D @ v_t  # (n, p)
                C_computed += temp * w_t.T  # w_t is (p,n), w_t.T is (n,p), w_t[k,i] = w_t.T[i,k]
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    # Build the canonical rank n*m*p decomposition (schoolbook)
    rank_schoolbook = n * m * p  # 96
    U0 = np.zeros((n * m, rank_schoolbook))
    V0 = np.zeros((m * p, rank_schoolbook))
    W0 = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Strategy: Use nested Strassen decomposition
    # 4x4x6 = 4x4x4 (first 4 columns of B) + 4x4x2 (last 2 columns of B)
    # 
    # For 4x4x4: Use 2x2 block Strassen with 2x2x2 Strassen inside
    #   Rank = 7 * 7 = 49
    # For 4x4x2: Schoolbook, rank = 4*4*2 = 32... wait, that's too high
    #   Actually for 4x4x2, each output C[i,k] for k=0,1 is sum_j A[i,j]*B[j,k]
    #   That's 8 dot products of length 4, so rank 8 if we can share... no, rank is 4*4*2=32 schoolbook
    #   But we can do better: for fixed k, C[:,k] = A*B[:,k] which is 4x4x1, rank 16 schoolbook
    #   Two such: rank 32. But we can potentially share.
    #   
    # Actually, let me think about this differently.
    # The tensor for 4x4x6 has 4*4*6 = 96 entries.
    # 
    # Better structural approach:
    # Split B into B1 (4x4) and B2 (4x2)
    # C = A*B1 + A*B2
    # 
    # For A*B1 (4x4x4): nested Strassen gives rank 49
    # For A*B2 (4x4x2): 
    #   C2[i,k] = sum_j A[i,j]*B2[j,k] for k=0,1
    #   This is like multiplying A (4x4) by two vectors B2[:,0] and B2[:,1]
    #   Each is a 4x4x1 multiplication (matrix-vector product)
    #   4x4x1: C[i] = sum_j A[i,j]*v[j], rank 16 schoolbook
    #   But we can do better using Strassen-like for matrix-vector:
    #   Actually 4x4x1 has rank at most 16, and likely exactly 16.
    #   Two of them: 32. Total: 49+32 = 81.
    #
    # Even better: can we share between the two matrix-vector products?
    # C2[:,0] = A*v0, C2[:,1] = A*v1
    # C2 = A * [v0, v1] = A * B2
    # This is just 4x4x2 matrix multiplication.
    # 
    # Let me try a different split: 4x4x6 = 2x4x6 + 2x4x6 (split A into top and bottom halves)
    # Each 2x4x6: C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1
    # 2x4x6 schoolbook: 2*4*6 = 48
    # Can we do better? 2x4x6 = 2x4x4 + 2x4x2
    # 2x4x4: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k], 2 outputs per k, 4 k's = 8 dot products of length 4
    #   Rank 32 schoolbook. Can we reduce? 
    #   Actually 2x4x4 can be seen as: for each k, [C[0,k], C[1,k]] = A[:,j] dot products...
    #   This is getting complex. Let me just try to construct the nested Strassen properly.
    
    # Let me construct a proper 4x4x4 nested Strassen decomposition.
    # 
    # 4x4x4 via 2x2 block Strassen:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # 
    # Strassen intermediates (each is a 2x2x2 matrix product):
    # S1 = (A00+A11)*(B00+B11)
    # S2 = (A10+A11)*B00
    # S3 = A00*(B01-B11)
    # S4 = A11*(B10-B00)
    # S5 = (A00+A01)*B11
    # S6 = (A10-A00)*(B00+B01)
    # S7 = (A01-A11)*(B10+B11)
    # 
    # C00 = S1 - S3 - S4 - S7
    # C01 = S5 + S7
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 - S5
    #
    # Each St is a 2x2x2 multiplication with rank 7 (Strassen).
    # Total rank: 7*7 = 49.
    #
    # Now I need to write out the factor matrices.
    # For the 4x4x4 part:
    # U has shape (16, 49): U[i*4+j, t]
    # V has shape (16, 49): V[j*4+k, t]
    # W has shape (16, 49): W[k*4+i, t]
    #
    # For each Strassen factor s (0..6), we have 7 sub-factors (for the 2x2x2 inside).
    # Let's index the 49 columns as t = s*7 + sub, where s in 0..6, sub in 0..6.
    #
    # First, define the 2x2x2 Strassen factors:
    def strassen_2x2x2():
        """
        Return U, V, W for 2x2x2 Strassen decomposition.
        U: (4,7), V: (4,7), W: (4,7)
        U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # Factor 0: (a00+a11)*(b00+b11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        # C00 += +1, C01 += 0, C10 += 0, C11 += +1
        W[0,0]=1; W[1,0]=0; W[2,0]=0; W[3,0]=1
        # Factor 1: (a10+a11)*b00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        # C10 += +1
        W[0,1]=0; W[1,1]=0; W[2,1]=1; W[3,1]=0
        # Factor 2: a00*(b01-b11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        # C00 += -1, C11 += +1
        W[0,2]=-1; W[1,2]=0; W[2,2]=0; W[3,2]=1
        # Factor 3: a11*(b10-b00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        # C00 += -1, C10 += +1
        W[0,3]=-1; W[1,3]=0; W[2,3]=1; W[3,3]=0
        # Factor 4: (a00+a01)*b11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        # C01 += +1, C11 += -1
        W[0,4]=0; W[1,4]=1; W[2,4]=0; W[3,4]=-1
        # Factor 5: (a10-a00)*(b00+b01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        # This factor is not directly in the standard Strassen output formula...
        # Let me re-derive. Standard Strassen:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        # C00 = P1 - P3 - P4 - P7
        # C01 = P5 + P7
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 - P5
        # P6 is not used! That can't be right for rank 7.
        # Actually, I think I'm confusing two different formulations.
        # Let me just use a known correct 2x2x2 rank-7 decomposition.
        # The key is that it exists and has rank 7. Let me use a verified one.
        
        # Actually, the standard Strassen algorithm uses 7 multiplications but
        # the tensor decomposition might be slightly different. Let me just
        # verify that my factors work by checking the tensor identity.
        
        return U, V, W

    # Let me just try a different approach: use the fact that we can verify
    # decompositions, and try to find good ones through structured search.
    
    # For now, let me construct the 4x4x4 part using a direct approach.
    # I'll use the "recursive Strassen" idea but implement it carefully.
    
    def build_4x4x4_nested():
        """
        Build 4x4x4 decomposition via nested Strassen.
        Returns U (16,49), V (16,49), W (16,49)
        """
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # 2x2x2 Strassen factors (verified)
        # Let me use a known correct set:
        # The 2x2x2 multiplication tensor has rank 7.
        # One explicit decomposition:
        sU = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=float)
        sV = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ], dtype=float)
        sW = np.array([
            [1, 0, -1, -1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, -1, 0, 0]
        ], dtype=float)
        
        # Verify 2x2x2 Strassen
        def verify_2x2x2(sU, sV, sW):
            for _ in range(3):
                A = np.random.randint(-3, 4, size=(2,2))
                B = np.random.randint(-3, 4, size=(2,2))
                C_expected = A @ B
                C_computed = np.zeros((2,2))
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            val = 0.0
                            for t in range(7):
                                val += A[i,j] * sU[i*2+j, t] * sV[j*2+k, t] * sW[k*2+i, t]
                            C_computed[i,k] += val
                if not np.allclose(C_computed, C_expected, atol=1e-9):
                    return False
            return True
        
        if not verify_2x2x2(sU, sV, sW):
            # Try another known decomposition
            # Let me just use schoolbook for 2x2x2 (rank 8) as fallback
            sU = np.zeros((4,8)); sV = np.zeros((4,8)); sW = np.zeros((4,8))
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        sU[i*2+j, col] = 1
                        sV[j*2+k, col] = 1
                        sW[k*2+i, col] = 1
                        col += 1
            rank = 8*7  # 56 instead of 49
        
        # Now build 4x4x4 using nested Strassen
        # Block structure:
        # A = [[A00, A01], [A10, A11]], each 2x2
        # B = [[B00, B01], [B10, B11]], each 2x2
        # 
        # Strassen factors:
        # S1 = (A00+A11)*(B00+B11): U gets A00+A11, V gets B00+B11
        # S2 = (A10+A11)*B00: U gets A10+A11, V gets B00
        # S3 = A00*(B01-B11): U gets A00, V gets B01-B11
        # S4 = A11*(B10-B00): U gets A11, V gets B10-B00
        # S5 = (A00+A01)*B11: U gets A00+A01, V gets B11
        # S6 = (A10-A00)*(B00+B01): U gets A10-A00, V gets B00+B01
        # S7 = (A01-A11)*(B10+B11): U gets A01-A11, V gets B10+B11
        #
        # Output:
        # C00 = S1 - S3 - S4 - S7
        # C01 = S5 + S7
        # C10 = S2 + S4
        # C11 = S1 - S2 + S3 - S5
        
        # For each Strassen factor s, the U contribution is a linear combination
        # of the 2x2 blocks of A, and V is a linear combination of 2x2 blocks of B.
        # The W contribution determines how the 2x2 output blocks combine.
        
        # Define the U, V patterns for each Strassen factor:
        # U patterns (which 2x2 blocks of A are used, with coefficients):
        # s=0: A00+ A11 -> blocks (0,0) and (1,1) with coeff 1
        # s=1: A10+ A11 -> blocks (1,0) and (1,1) with coeff 1
        # s=2: A00        -> block (0,0) with coeff 1
        # s=3: A11        -> block (1,1) with coeff 1
        # s=4: A00+ A01 -> blocks (0,0) and (0,1) with coeff 1
        # s=5: A10- A00 -> blocks (1,0) with coeff 1, (0,0) with coeff -1
        # s=6: A01- A11 -> blocks (0,1) with coeff 1, (1,1) with coeff -1
        
        U_patterns = [
            {(0,0):1, (1,1):1},    # S1
            {(1,0):1, (1,1):1},    # S2
            {(0,0):1},              # S3
            {(1,1):1},              # S4
            {(0,0):1, (0,1):1},    # S5
            {(1,0):1, (0,0):-1},   # S6
            {(0,1):1, (1,1):-1},   # S7
        ]
        V_patterns = [
            {(0,0):1, (1,1):1},    # S1: B00+B11
            {(0,0):1},              # S2: B00
            {(0,1):1, (1,1):-1},   # S3: B01-B11
            {(1,0):1, (0,0):-1},   # S4: B10-B00
            {(1,1):1},              # S5: B11
            {(0,0):1, (0,1):1},    # S6: B00+B01
            {(1,0):1, (1,1):1},    # S7: B10+B11
        ]
        # W patterns: how each 2x2 output block (C00,C01,C10,C11) gets contributions
        # C00 = S1 - S3 - S4 - S7
        # C01 = S5 + S7
        # C10 = S2 + S4
        # C11 = S1 - S2 + S3 - S5
        W_patterns = [
            {0:1, 1:0, 2:0, 3:1},   # S1: C00+=1, C11+=1
            {0:0, 1:0, 2:1, 3:0},   # S2: C10+=1
            {0:-1, 1:0, 2:0, 3:1},  # S3: C00+=-1, C11+=1
            {0:-1, 1:0, 2:1, 3:0},  # S4: C00+=-1, C10+=1
            {0:0, 1:1, 2:0, 3:-1},  # S5: C01+=1, C11+=-1
            {0:0, 1:0, 2:0, 3:0},   # S6: not in output (should be 0)
            {0:-1, 1:1, 2:0, 3:0},  # S7: C00+=-1, C01+=1
        ]
        
        sub_rank = sU.shape[1]
        total_rank = 7 * sub_rank
        
        U = np.zeros((16, total_rank))
        V = np.zeros((16, total_rank))
        W = np.zeros((16, total_rank))
        
        for s in range(7):
            for sub in range(sub_rank):
                t = s * sub_rank + sub
                # U: for each 2x2 block (bi, bj) in A_patterns[s]
                for (bi, bj), coeff in U_patterns[s].items():
                    # Block (bi,bj) corresponds to rows bi*2, bi*2+1 and cols bj*2, bj*2+1
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i*4+j, t] += coeff * sU[di*2+dj, sub]
                
                # V: for each 2x2 block (bi, bj) in B_patterns[s]
                for (bi, bj), coeff in V_patterns[s].items():
                    for di in range(2):
                        for dj in range(2):
                            j = bi * 2 + di
                            k = bj * 2 + dj
                            V[j*4+k, t] += coeff * sV[di*2+dj, sub]
                
                # W: for each output block (ci, ck) in W_patterns[s]
                # C block (ci, ck) means output rows ci*2, ci*2+1 and cols ck*2, ck*2+1
                for (out_block, coeff) in W_patterns[s].items():
                    # out_block: 0=C00, 1=C01, 2=C10, 3=C11
                    oi = out_block // 2  # output row block
                    ok = out_block % 2   # output col block
                    for di in range(2):
                        for dk in range(2):
                            i = oi * 2 + di
                            k = ok * 2 + dk
                            W[k*4+i, t] += coeff * sW[dk*2+di, sub]
        
        return U, V, W, total_rank

    # Try to build 4x4x4 nested Strassen
    U_444, V_444, W_444, rank_444 = build_4x4x4_nested()
    
    # Verify 4x4x4
    def verify_4x4x4(U, V, W):
        for _ in range(3):
            A = np.random.randint(-3, 4, size=(4,4))
            B = np.random.randint(-3, 4, size=(4,4))
            C_expected = A @ B
            C_computed = np.zeros((4,4))
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        val = 0.0
                        for t in range(U.shape[1]):
                            val += A[i,j] * U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t]
                        C_computed[i,k] += val
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    if verify_4x4x4(U_444, V_444, W_444):
        # Now build the full 4x4x6 decomposition
        # 4x4x6 = 4x4x4 (first 4 cols of B) + 4x4x2 (last 2 cols of B)
        # 
        # For 4x4x2: schoolbook rank 32
        # But we can do better. Let's use the same nested approach for the 2 columns.
        # Actually, 4x4x2 is just 4x4x1 twice. 4x4x1 rank is 16 schoolbook.
        # Total: 49 + 32 = 81.
        
        rank_442 = 4*4*2  # 32 schoolbook
        U_442 = np.zeros((16, rank_442))
        V_442 = np.zeros((16, rank_442))  # Note: V is (m*p, rank) = (4*6, rank) = (24, rank)
        # Wait, for 4x4x6, V has shape (m*p, rank) = (4*6, rank) = (24, rank)
        # For the 4x4x4 part, V only uses the first 4 columns of B, so rows 0-15 (j*4+k for k=0..3)
        # For the 4x4x2 part, V uses the last 2 columns of B, so rows 16-23 (j*4+k for k=4,5)
        
        # Let me redo this with correct dimensions.
        # U: (n*m, rank) = (16, rank)
        # V: (m*p, rank) = (24, rank)
        # W: (p*n, rank) = (24, rank)
        
        # For 4x4x4 part (k=0,1,2,3):
        # U_444: (16, 49)
        # V_444: (16, 49) -> but we need (24, 49), so pad rows 16-23 with 0
        # W_444: (16, 49) -> but we need (24, 49), so pad rows 16-23 with 0
        
        V_444_full = np.zeros((24, rank_444))
        V_444_full[:16, :] = V_444  # j*4+k for j=0..3, k=0..3 -> rows 0..15
        
        W_444_full = np.zeros((24, rank_444))
        W_444_full[:16, :] = W_444  # k*4+i for k=0..3, i=0..3 -> rows 0..15
        
        # For 4x4x2 part (k=4,5):
        # C[i,k] = sum_j A[i,j]*B[j,k] for k=4,5
        # U_442: (16, 32)
        # V_442: (24, 32) - only rows 16-23 are non-zero (j*4+k for k=4,5)
        # W_442: (24, 32) - only rows 16-23 are non-zero (k*4+i for k=4,5)
        
        U_442 = np.zeros((16, rank_442))
        V_442 = np.zeros((24, rank_442))
        W_442 = np.zeros((24, rank_442))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    U_442[i*4+j, col] = 1
                    V_442[j*4+(k+4), col] = 1  # k=4,5 -> offset by 4
                    W_442[(k+4)*4+i, col] = 1
                    col += 1
        
        # Combine
        total_rank = rank_444 + rank_442
        U_full = np.hstack([U_444, U_442])
        V_full = np.hstack([V_444_full, V_442])
        W_full = np.hstack([W_444_full, W_442])
        
        # Verify full 4x4x6
        if verify_fast(U_full, V_full, W_full, num_trials=3):
            if total_rank < best_rank:
                best_rank = total_rank
                best_U, best_V, best_W = U_full.copy(), V_full.copy(), W_full.copy()
    
    # Try to improve: can we do 4x4x2 with less than rank 32?
    # 4x4x2: C[i,k] = sum_j A[i,j]*B[j,k] for k=0,1 (relabeling)
    # This is two 4x4x1 problems.
    # 4x4x1: C[i] = sum_j A[i,j]*v[j]
    # Can we share between the two? 
    # C[:,0] = A*v0, C[:,1] = A*v1
    # C = A*[v0,v1]
    # This is just 4x4x2 matrix multiplication.
    # The tensor has 4*4*2 = 32 entries.
    # Lower bound: each C[i,k] depends on all A[i,j] and B[j,k].
    # I believe rank 32 is optimal for 4x4x2 (no known improvement).
    
    # Try another structural approach: split differently
    # 4x4x6 = 4x3x6 + 4x1x6
    # 4x3x6: rank 72 schoolbook
    # 4x1x6: C[i,k] = A[i,3]*B[3,k], rank 24 schoolbook
    # Total: 96. No improvement.
    
    # Try: 4x4x6 = 2x4x6 + 2x4x6
    # Each 2x4x6: C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1
    # 2x4x6 schoolbook: 2*4*6 = 48
    # Can we do 2x4x6 in less than 48?
    # 2x4x6 = 2x4x4 + 2x4x2
    # 2x4x4: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k], 2*4 = 8 dot products of length 4
    #   Rank 32 schoolbook. 
    #   Can we reduce? 2x4x4: for each k, [C[0,k], C[1,k]] = [A[0,:], A[1,:]] @ B[:,k]
    #   This is like multiplying a 2x4 matrix by a 4x4 matrix.
    #   I don't see an obvious Strassen-like reduction here.
    # 2x4x2: 2*4*2 = 16 schoolbook.
    # Total per 2x4x6: 32+16 = 48. Two of them: 96. No improvement.
    
    # Let me try to verify and return the best found so far
    if (time.time() - start_time) > deadline - 5:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Try to find improvements through random search
    # Start from the best decomposition and try to reduce rank
    if best_rank < rank_schoolbook:
        # Try to merge columns
        def try_reduce(U, V, W, target_rank, time_limit=10):
            start = time.time()
            U = U.copy()
            V = V.copy()
            W = W.copy()
            r = U.shape[1]
            
            while r > target_rank and (time.time() - start) < time_limit:
                merged = False
                for t1 in range(r):
                    for t2 in range(t1+1, r):
                        # Check if columns can be merged
                        if np.allclose(U[:,t1], U[:,t2]) and np.allclose(V[:,t1], V[:,t2]):
                            W[:,t1] += W[:,t2]
                            U = np.delete(U, t2, axis=1)
                            V = np.delete(V, t2, axis=1)
                            W = np.delete(W, t2, axis=1)
                            r = U.shape[1]
                            merged = True
                            break
                    if merged:
                        break
                if not merged:
                    break
            return U, V, W, r
        
        U_red, V_red, W_red, r_red = try_reduce(best_U, best_V, best_W, best_rank - 1, time_limit=5)
        if r_red < best_rank and verify_fast(U_red, V_red, W_red, num_trials=3):
            best_rank = r_red
            best_U, best_V, best_W = U_red, V_red, W_red
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
