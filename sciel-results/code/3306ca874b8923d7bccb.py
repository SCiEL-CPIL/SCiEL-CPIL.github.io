import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook decomposition (rank 96)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Strategy: Try Kronecker product of smaller decompositions
    # 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # We need valid decompositions for 2x2x2 (rank 7, Strassen) and 2x2x3 (rank 5)
    
    # ===== Strassen decomposition for 2x2x2 (rank 7) =====
    def strassen_2x2x2():
        """Returns U, V, W for 2x2x2 matrix multiplication, rank 7."""
        nn, mm, pp = 2, 2, 2
        r = 7
        U = np.zeros((nn*mm, r))
        V = np.zeros((mm*pp, r))
        W = np.zeros((pp*nn, r))
        
        # M1 = (a11+a22)(b11+b22)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2 = (a21+a22)b11
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1
        
        # M3 = a11(b12-b22)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[2,2] = 1; W[3,2] = 1
        
        # M4 = a22(b21-b11)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1
        
        # M5 = (a11+a12)b22
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[2,4] = 1
        
        # M6 = (a21-a11)(b11+b12)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # M7 = (a12-a22)(b21+b22)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W
    
    # ===== Rank-5 decomposition for 2x2x3 =====
    # This is the tricky part. Let's derive it carefully.
    # A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We want 5 bilinear terms that reconstruct all 6 entries of C.
    # 
    # Let me try a different approach: use the fact that we can share
    # computations between columns of B.
    
    def rank5_2x2x3():
        """Returns U, V, W for 2x2x3 matrix multiplication, rank 5."""
        nn, mm, pp = 2, 2, 3
        r = 5
        U = np.zeros((nn*mm, r))
        V = np.zeros((mm*pp, r))
        W = np.zeros((pp*nn, r))
        
        # Indexing:
        # U: i*mm + j, i in [0,1], j in [0,1] -> indices 0,1,2,3
        # V: j*pp + k, j in [0,1], k in [0,1,2] -> indices 0,1,2,3,4,5
        # W: k*nn + i, k in [0,1,2], i in [0,1] -> indices 0,1,2,3,4,5
        
        # Term 1: (a11 + a22) * (b11 + b21)
        # Contributes to C[0,0] and C[1,0]
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[2,0] = 1  # C[0,0] and C[1,0]
        
        # Term 2: (a21 + a22) * b11
        # Contributes to C[1,0]
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = -1  # C[1,0] gets -M2
        
        # Term 3: a11 * (b12 - b22)
        # Contributes to C[0,1]
        U[0,2] = 1
        V[1,2] = 1; V[4,2] = -1
        W[1,2] = 1  # C[0,1] gets M3
        
        # Term 4: a22 * (b21 - b11)
        # Contributes to C[1,0]
        U[3,3] = 1
        V[3,3] = 1; V[0,3] = -1
        W[2,3] = 1  # C[1,0] gets M4
        
        # Term 5: (a11 + a12) * b13
        # Contributes to C[0,2]
        U[0,4] = 1; U[1,4] = 1
        V[2,4] = 1
        W[4,4] = 1  # C[0,2] gets M5
        
        # Let's verify:
        # C[0,0] = M1 = (a11+a22)(b11+b21) = a11*b11 + a11*b21 + a22*b11 + a22*b21
        # Should be: a11*b11 + a12*b21
        # This is WRONG. M1 has a11*b21 and a22*b11 and a22*b21 which shouldn't be there.
        
        # The issue is that simple Kronecker-style sharing doesn't work directly.
        # Let me try a different construction.
        
        # Actually, for 2x2x3, let's think of it as:
        # We need to compute:
        # C[0,0] = a11*b11 + a12*b21
        # C[0,1] = a11*b12 + a12*b22
        # C[0,2] = a11*b13 + a12*b23
        # C[1,0] = a21*b11 + a22*b21
        # C[1,1] = a21*b12 + a22*b22
        # C[1,2] = a21*b13 + a22*b23
        
        # With 5 terms, we can at most have 5 independent bilinear products.
        # The 6 outputs must be linear combinations of these 5 products.
        
        # Let's try:
        # M1 = (a11 + a21) * b11  -> helps C[0,0], C[1,0]
        # M2 = (a12 + a22) * b21  -> helps C[0,0], C[1,0]
        # M3 = (a11 + a21) * b12  -> helps C[0,1], C[1,1]
        # M4 = (a12 + a22) * b22  -> helps C[0,1], C[1,1]
        # M5 = (a11 + a21 + a12 + a22) * (b13 + b23)  -> helps C[0,2], C[1,2]
        
        # But this gives:
        # C[0,0] = M1 + M2 = a11*b11 + a21*b11 + a12*b21 + a22*b21
        # Should be: a11*b11 + a12*b21
        # Extra terms: a21*b11 + a22*b21
        
        # This doesn't work either. The fundamental issue is that 2x2x3
        # may not have rank 5 with simple integer coefficients.
        
        # Let me try a different approach: use rank 6 for 2x2x3.
        # Then total rank would be 7 * 6 = 42.
        
        # Actually, let me just use schoolbook for 2x2x3 (rank 12)
        # Total: 7 * 12 = 84, which is better than 96.
        
        return None  # Will use schoolbook for now
    
    # Let's try a direct approach: build 4x4x6 from 2x2x2 and 2x2x3 blocks
    # using a block structure.
    
    # Alternative: Try to find a decomposition using structured patterns.
    # Let's use the approach of fixing U and V to sparse patterns and solving for W.
    
    # For now, let's try a few random structured approaches and verify.
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # For valid decomposition, this should be 1 for all i,j,k
        # And for all other (a,b,c) where the tensor is 0, it should be 0.
        
        # The tensor for matrix multiplication has T[i*m+j, j*p+k, k*n+i] = 1
        # and 0 elsewhere.
        
        # Check all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        
        # Check some random non-tensor entries to ensure they're 0
        # The tensor is zero unless the indices satisfy the matrix multiplication pattern.
        # For a random (a,b,c), it's zero unless a = i*m+j, b = j*p+k, c = k*n+i
        # for some consistent i,j,k.
        
        # Actually, the tensor is defined as:
        # T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
        # and the indices are consistent, else 0.
        
        # For our case, let's just check that the decomposition works for the
        # matrix multiplication identity, which is what matters.
        
        # More thorough check: verify that the bilinear form computes matrix multiplication.
        # For random matrices A (n x m) and B (m x p), check that the bilinear form
        # gives C = A*B.
        
        for trial in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute using the decomposition
            C_computed = np.zeros((n, p))
            for r in range(rank):
                # u_r is a vector of length n*m, reshape to n x m
                u_r = U[:, r].reshape(n, m)
                # v_r is a vector of length m*p, reshape to m x p
                v_r = V[:, r].reshape(m, p)
                # w_r is a vector of length p*n, reshape to p x n
                w_r = W[:, r].reshape(p, n)
                
                # The bilinear form contribution is:
                # sum_{i,j,k} u_r[i,j] * A[i,j] * v_r[j,k] * B[j,k] * w_r[k,i]
                # This should equal sum_{i,k} C[i,k] * w_r[k,i] ... no wait.
                
                # Actually, the decomposition says:
                # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # summed over j and r.
                
                # Let me recompute:
                # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
            for i in range(n):
                for k in range(p):
                    val = 0
                    for j in range(m):
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - C_expected[i, k]) > tol:
                        return False
        
        return True
    
    # Try Kronecker product approach with schoolbook for the 2x2x3 part
    # 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # Rank = rank(2x2x2) * rank(2x2x3)
    
    # For 2x2x2, use Strassen (rank 7)
    # For 2x2x3, use schoolbook (rank 12)
    # Total: 7 * 12 = 84
    
    if time.time() < deadline - 10:
        U1, V1, W1 = strassen_2x2x2()
        
        # Schoolbook for 2x2x3
        nn, mm, pp = 2, 2, 3
        r2 = nn * mm * pp  # 12
        U2 = np.zeros((nn*mm, r2))
        V2 = np.zeros((mm*pp, r2))
        W2 = np.zeros((pp*nn, r2))
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U2[i*mm+j, col] = 1
                    V2[j*pp+k, col] = 1
                    W2[k*nn+i, col] = 1
                    col += 1
        
        # Kronecker product
        # 4x4x6 from 2x2x2 ⊗ 2x2x3
        # The Kronecker product of decompositions:
        # U = U1 ⊗ U2 (with appropriate reshaping)
        # This is complex. Let me just use the schoolbook for now.
        
        # Actually, let me try a simpler structured approach.
        # Use block-diagonal structure.
    
    # Try random search with verification
    random.seed(42)
    
    for target_rank in [40, 45, 50, 55, 60, 65, 70, 75, 80, 84, 88, 92]:
        if time.time() > deadline - 20:
            break
        
        if target_rank >= best_rank:
            continue
        
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            
            # Initialize with sparse random patterns
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Add random sparse entries
            for r in range(target_rank):
                # Each factor has a few non-zero entries
                num_u = random.randint(2, 5)
                num_v = random.randint(2, 5)
                num_w = random.randint(2, 5)
                
                u_idx = random.sample(range(n*m), min(num_u, n*m))
                v_idx = random.sample(range(m*p), min(num_v, m*p))
                w_idx = random.sample(range(p*n), min(num_w, p*n))
                
                for idx in u_idx:
                    U[idx, r] = random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, r] = random.choice([-1, 1])
                for idx in w_idx:
                    W[idx, r] = random.choice([-1, 1])
            
            # Quick check
            if verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
    
    # Return the best result found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
