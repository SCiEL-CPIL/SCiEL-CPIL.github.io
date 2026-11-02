import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2():
    """Return Strassen's rank-7 decomposition of 2x2x2 matrix multiplication.
    U shape (4,7), V shape (4,7), W shape (4,7) where
    U[i*2+j, r] corresponds to A[i,j],
    V[j*2+k, r] corresponds to B[j,k],
    W[k*2+i, r] corresponds to C[i,k] (trace convention W[k*n+i]).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices (W[k*2+i]): C00 -> W[0*2+0]=W[0], C01 -> W[1*2+0]=W[2], C10 -> W[0*2+1]=W[1], C11 -> W[1*2+1]=W[3]
    
    # r=0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 and C11
    
    # r=1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C10
    
    # r=2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C01 and C11
    
    # r=3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C00 and C10
    
    # r=4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # -C00, C01
    
    # r=5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11
    
    # r=6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00
    
    return U, V, W

def _verify(U, V, W, rank, n, m, p):
    """Verify the tensor decomposition exactly."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check all constraints: sum_r U[i*m+j, r] V[jp*p+k, r] W[kp*n+ip, r] = delta(i,ip)*delta(k,kp)*delta(j,jp)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            target = 1.0 if (i == ip and k == kp and j == jp) else 0.0
                            val = 0.0
                            for r in range(rank):
                                val += U_np[i*m+j, r] * V_np[jp*p+k, r] * W_np[kp*n+ip, r]
                            if abs(val - target) > 1e-9:
                                return False
    return True

def _verify_fast(U, V, W, rank, n, m, p):
    """Faster verification using tensor operations."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    # The tensor T[i, j, jp, k, ip, kp] = sum_r U[i,j,r] V[jp,k,r] W[kp,ip,r]
    # We need T[i,j,jp,k,ip,kp] = delta(i,ip) delta(k,kp) delta(j,jp)
    
    # Build the full tensor efficiently
    # T[i, j, jp, k, ip, kp] = sum_r U_r[i,j] * V_r[jp,k] * W_r[kp,ip]
    
    # Use einsum: T = np.einsum('ijr,jkr,kir->ijjkkii', ...) -- indices are getting complex
    # Let's use explicit loops over the small dimensions but vectorize over r
    
    # Actually, let's just check the 90 non-zero constraints and sample zero constraints
    # Non-zero: i==ip, k==kp, j==jp -> 90 constraints
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Zero constraints: check all where NOT (i==ip AND k==kp AND j==jp)
    # Total: 8100 - 90 = 8010
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            if i == ip and k == kp and j == jp:
                                continue
                            val = np.sum(U_np[i*m+j, :] * V_np[jp*p+k, :] * W_np[kp*n+ip, :])
                            if abs(val) > 1e-9:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook solution as fallback (rank 90)
    rank_trivial = n * m * p
    U_triv = [[0] * rank_trivial for _ in range(n * m)]
    V_triv = [[0] * rank_trivial for _ in range(m * p)]
    W_triv = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j][col] = 1
                V_triv[j * p + k][col] = 1
                W_triv[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    best_rank = rank_trivial
    
    # Block-diagonal embedding: split 3x5x6 into 3x2x6 and 3x3x6
    # Block 1: 3x2x6 using Strassen for the 2x2 part (rank 7 per 2x2x2, times 3 rows x 3 col-pairs = 7*3*3=63? No)
    # Actually: 3x2x6 = for each of 3 rows of A, 2 cols of A, 6 cols of B
    # We can think of it as: for each pair of columns in B (3 pairs of 2), we do a 3x2x2 multiplication
    # A 3x2x2 multiplication: A is 3x2, B is 2x2, C is 3x2
    # This can be decomposed as 3 independent 1x2x2? No.
    # Better: 3x2x2 = sum of 3 (1x2x2) terms? No, that's not right either.
    # 
    # Let me reconsider. The tensor for 3x2x6:
    # U shape: (3*2, R) = (6, R)
    # V shape: (2*6, R) = (12, R)
    # W shape: (6*3, R) = (18, R)
    # 
    # We can decompose this as: for each of the 3 "row blocks" of A (rows 0,1,2),
    # and for each of the 3 "col pairs" of B (cols (0,1), (2,3), (4,5)),
    # we have a 1x2x2 sub-problem? No, that doesn't work because A has 3 rows.
    #
    # Alternative approach: 3x2x6 = 3 * (1x2x6) where each 1x2x6 is a row of A times B.
    # 1x2x6: A is 1x2, B is 2x6, C is 1x6. Rank = 2*6 = 12 schoolbook.
    # Total: 3 * 12 = 36. Not better than schoolbook 36 for this block.
    #
    # Let me try a different split. Use Strassen on the 2x2x2 blocks.
    # 3x2x6: Think of A as 3 rows, 2 cols. B as 2 rows, 6 cols.
    # Split B into 3 blocks of 2 cols each: B = [B1, B2, B3] where each Bj is 2x2.
    # Then C = A * B = [A*B1, A*B2, A*B3] where each A*Bj is 3x2.
    # Each A*Bj is a 3x2x2 multiplication.
    # 
    # 3x2x2 multiplication: A (3x2), B (2x2), C (3x2).
    # This is equivalent to 3 independent 1x2x2 multiplications (one per row of A).
    # Each 1x2x2 has rank 4 (schoolbook) or can we do better?
    # 1x2x2: A = [a0, a1], B = [[b00,b01],[b10,b11]], C = [a0*b00+a1*b10, a0*b01+a1*b11]
    # This is just 4 scalar multiplications. Rank = 4.
    # So 3x2x2 has rank 3*4 = 12.
    # And 3x2x6 has rank 3 * 12 = 36. Same as schoolbook.
    #
    # Hmm, that's not helping. Let me think differently.
    #
    # What if I use Strassen for the 2x2x2 part properly?
    # 3x2x6: I can view this as 3 copies of a 1x2x6, but that gives 36.
    # OR: I can view the 2x2 part of B and use Strassen.
    #
    # Actually, let me try: 3x5x6 = 3x3x6 + 3x2x6 (splitting the 5 columns of A into 3+2)
    # 3x3x6: schoolbook rank = 3*3*6 = 54
    # 3x2x6: schoolbook rank = 3*2*6 = 36
    # Total schoolbook: 90
    #
    # To improve, I need to use fast matrix multiplication on one of the blocks.
    # 
    # For 3x2x6: Can I use Strassen? Strassen works on 2x2x2.
    # 3x2x6 = for each of 3 rows of A, compute row * B (2x6).
    # Each row (1x2) * B (2x6) = 1x6 result. This is 2*6 = 12 scalar mults.
    # Total: 3*12 = 36. No improvement.
    #
    # Alternative: 3x5x6 = 2x5x6 + 1x5x6
    # 2x5x6: schoolbook = 2*5*6 = 60
    # 1x5x6: schoolbook = 1*5*6 = 30
    # Total: 90
    #
    # For 2x5x6: Can I use Strassen on the 2x2 parts?
    # 2x5x6: A is 2x5, B is 5x6, C is 2x6.
    # Split A into 2x2 + 2x3? Or use Strassen on 2x2x2 blocks of B.
    # B (5x6) can be split into 5x2 + 5x2 + 5x2 (three 5x2 blocks).
    # C = A * B = [A*B1, A*B2, A*B3] where each A*Bj is 2x2.
    # Each A*Bj is a 2x5x2 multiplication.
    # 2x5x2: A (2x5), B (5x2), C (2x2).
    # This is 5 independent 2x1x2 multiplications? No.
    # C[i,k] = sum_j A[i,j] B[j,k]. For fixed i,k, it's a dot product of length 5.
    # There are 2*2 = 4 such dot products, each of length 5. Total 20 scalar mults.
    # So 2x5x2 has rank 20. And 2x5x6 has rank 3*20 = 60. No improvement.
    #
    # Let me try a completely different approach. Use Strassen's algorithm directly.
    # Strassen reduces 2x2x2 from rank 8 to rank 7.
    # For 3x5x6, I can pad to 4x8x8 and use Strassen recursively?
    # 4x8x8: Use Strassen on 4x4x4 blocks?
    # This is getting complicated.
    #
    # Let me just implement the block approach with Strassen where it helps.
    # 
    # Key insight: 3x2x6 can be decomposed using Strassen if I group things right.
    # Actually, 3x2x6 = 3 * (1x2x6). Each 1x2x6 has rank 12 (schoolbook).
    # But if I group the 6 columns into 3 pairs of 2, and use Strassen...
    # No, Strassen needs 2x2x2.
    #
    # Let me try: 3x5x6. Split into 3x3x3 + 3x3x3 + 3x2x3 + 3x2x3? No, that's 3x5x6 = (3x3 + 3x2) x (3+3) ... this is messy.
    #
    # Simplest approach that might work: Use the fact that for small matrices,
    # we can find low-rank decompositions by solving the system of equations.
    #
    # For rank 60, I have 60 variables per factor. The system has 8100 equations.
    # This is underdetermined (60*3 = 180 unknowns vs 8100 equations), so I need to be clever.
    #
    # Actually, the number of unknowns is 15*60 + 30*60 + 18*60 = 900 + 1800 + 1080 = 3780.
    # And 8100 equations. So it's overdetermined. But the equations are not all independent.
    #
    # Let me just try to construct the block-diagonal solution with rank 60.
    # Block 1: 3x2x6 with rank 24 (using Strassen-inspired construction)
    # Block 2: 3x3x6 with rank 36 (schoolbook)
    # Total rank: 60
    
    # For 3x2x6 with rank 24:
    # U1 shape: (6, 24), V1 shape: (12, 24), W1 shape: (18, 24)
    # For 3x3x6 with rank 36:
    # U2 shape: (9, 36), V2 shape: (18, 36), W2 shape: (18, 36)
    
    # Let me construct the 3x2x6 block using a Strassen-like approach.
    # 3x2x6: A is 3x2, B is 2x6, C is 3x6.
    # Split B into 3 blocks of 2 columns: B = [B1, B2, B3], each Bj is 2x2.
    # C = [A*B1, A*B2, A*B3], each A*Bj is 3x2.
    # Each A*Bj: A (3x2) * Bj (2x2) = Cj (3x2).
    # This is 3 independent 1x2x2 multiplications (one per row of A).
    # Each 1x2x2 has rank 4. So each A*Bj has rank 12.
    # Total for 3x2x6: 3 * 12 = 36. Still not 24.
    #
    # To get rank 24 for 3x2x6, I need a different approach.
    # 24 = 3 * 8. So each of the 3 "row groups" should have rank 8.
    # A row of A is 1x2. Multiplied by B (2x6) gives 1x6.
    # 1x2x6 has rank 12 schoolbook. Can I do it in rank 8?
    # 1x2x6: C[k] = a0*B[0,k] + a1*B[1,k] for k=0..5.
    # This is 12 scalar multiplications. To reduce rank, I need to share computations.
    # But since a0 and a1 are fixed for a given row, and B varies, I can't really share.
    # Unless I use the fact that this is a bilinear map and find a low-rank decomposition.
    #
    # Actually, 1x2x6 is just 2 vectors of length 6 being combined. The rank is 2*6=12
    # because there's no structure to exploit. So I can't get below 12 for 1x2x6.
    #
    # Therefore, 3x2x6 has rank at least 3*12 = 36. My assumption of rank 24 was wrong.
    #
    # Let me reconsider. Maybe I should use a different split.
    # 3x5x6 = 3x5x2 + 3x5x2 + 3x5x2 (three 3x5x2 blocks)
    # Each 3x5x2 has schoolbook rank 30. Total 90. No improvement.
    #
    # Or: 3x5x6 = 2x5x6 + 1x5x6
    # 2x5x6: Can I use Strassen? 2x5x6 = for each of 6 columns of B, compute A * col (2x5 * 5x1 = 2x1).
    # Each column: 2*5 = 10 scalar mults. Total 60.
    # 1x5x6: 1*5*6 = 30.
    # Total: 90.
    #
    # It seems hard to beat 90 with simple block decomposition.
    #
    # Let me try a different strategy: use Strassen's algorithm to reduce the rank.
    # Strassen reduces 2x2x2 from 8 to 7.
    # If I can decompose 3x5x6 into blocks where Strassen applies, I can save.
    #
    # Pad 3x5x6 to 4x8x8 (next power of 2).
    # 4x8x8: Use Strassen recursively.
    # 4x8x8 = 2x2x2 blocks of size 4x4x4? No, Strassen works on equal-size squares.
    # 4x8x8 is not square. Let me pad to 8x8x8.
    # 8x8x8: Use Strassen recursively.
    # 8x8x8 = 8 * (1x8x8) = 8 * 64 = 512 schoolbook.
    # With Strassen: 8x8x8 = 7 * (4x4x4) = 7 * 7 * (2x2x2) = 7 * 7 * 7 = 343.
    # But I need to account for the padding. The actual computation is 3x5x6, so I need to
    # extract the relevant parts.
    #
    # This is getting too complex. Let me just return the schoolbook solution for now,
    # but try to implement a simple ALS solver to find a lower rank.
    
    # Try ALS for rank 80, 75, 70, 65, 60
    def als_attempt(target_rank, max_iter=30, seed=42):
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # For fixed i,j, we need to solve for U[i*m+j, :]
                    # The equations are: for all jp, k, ip, kp:
                    # sum_r U[i*m+j, r] V[jp*p+k, r] W[kp*n+ip, r] = delta(i,ip)*delta(k,kp)*delta(j,jp)
                    # Let b_{jp,k,ip,kp} = delta(i,ip)*delta(k,kp)*delta(j,jp)
                    # Let A_{(jp,k,ip,kp), r} = V[jp*p+k, r] * W[kp*n+ip, r]
                    # Then A @ U[i*m+j, :] = b
                    
                    # Build A and b
                    A_mat = np.zeros((m*p*n*p, target_rank))
                    b_vec = np.zeros(m*p*n*p)
                    idx = 0
                    for jp in range(m):
                        for k in range(p):
                            for ip in range(n):
                                for kp in range(p):
                                    target = 1.0 if (i == ip and k == kp and j == jp) else 0.0
                                    b_vec[idx] = target
                                    for r in range(target_rank):
                                        A_mat[idx, r] = V[jp*p+k, r] * W[kp*n+ip, r]
                                    idx += 1
                    
                    # Solve least squares
                    try:
                        U[i*m+j, :], residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        return None
                    
            # Update V
            for jp in range(m):
                for k in range(p):
                    A_mat = np.zeros((n*m*n*p, target_rank))
                    b_vec = np.zeros(n*m*n*p)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for ip in range(n):
                                for kp in range(p):
                                    target = 1.0 if (i == ip and k == kp and j == jp) else 0.0
                                    b_vec[idx] = target
                                    for r in range(target_rank):
                                        A_mat[idx, r] = U[i*m+j, r] * W[kp*n+ip, r]
                                    idx += 1
                    try:
                        V[jp*p+k, :], residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        return None
            
            # Update W
            for kp in range(p):
                for ip in range(n):
                    A_mat = np.zeros((n*m*m*p, target_rank))
                    b_vec = np.zeros(n*m*m*p)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    target = 1.0 if (i == ip and k == kp and j == jp) else 0.0
                                    b_vec[idx] = target
                                    for r in range(target_rank):
                                        A_mat[idx, r] = U[i*m+j, r] * V[jp*p+k, r]
                                    idx += 1
                    try:
                        W[kp*n+ip, :], residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        return None
            
            # Check convergence
            if _verify_fast(U.tolist(), V.tolist(), W.tolist(), target_rank, n, m, p):
                return (U, V, W)
        
        return None
    
    # Try decreasing ranks
    for rank in [80, 75, 70, 65, 60, 55]:
        if time.time() > deadline:
            break
        result = als_attempt(rank)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            break
    
    return best_result
