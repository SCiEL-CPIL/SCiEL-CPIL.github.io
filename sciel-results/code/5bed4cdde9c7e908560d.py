import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We need to decompose the matrix multiplication tensor M_{3,4,8}
    # T[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Key insight: M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # M_{1,1,2} has rank 2: U=[[1,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]
    # 
    # For M_{3,4,4}: 3x4 * 4x4 -> 3x4
    # Pad to 4x4: use Strassen rank-7 for M_{4,4,4}, then restrict
    # Actually, we can use the structure more directly.
    
    # Let's try a direct structured approach:
    # M_{3,4,8}: Split the 4 shared dimension into two 2s, and 8 output into two 4s
    # 
    # Actually, let me use the tensor product structure properly.
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # 
    # If M_{3,4,4} has rank r, then M_{3,4,8} has rank 2r.
    # 
    # For M_{3,4,4}: A(3x4) * B(4x4) -> C(3x4)
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # We can view this as: for each i, compute row_i(A) * B
    # row_i(A) is 1x4, B is 4x4, result is 1x4
    # M_{1,4,4} has schoolbook rank 16
    # So M_{3,4,4} has schoolbook rank 3*16 = 48
    # 
    # Can we do better for M_{1,4,4}?
    # M_{1,4,4} = M_{1,2,2} ⊗ M_{1,2,2} (split 4 into 2+2 for both j and k)
    # M_{1,2,2}: 1x2 * 2x2 -> 1x2, rank 4 (schoolbook: 1*2*2=4)
    # So M_{1,4,4} rank = 4*4 = 16. No improvement.
    
    # Alternative: Use Strassen on the 4x4 part.
    # M_{4,4,4} has rank 7 (Strassen).
    # M_{3,4,4} is a sub-tensor of M_{4,4,4} (just drop the 4th row of output).
    # We can use the same 7 rank-1 tensors from Strassen, ignoring the 4th output row.
    # So M_{3,4,4} has rank <= 7.
    
    # Therefore M_{3,4,8} has rank <= 7 * 2 = 14.
    
    # Let me construct the Strassen rank-7 decomposition for M_{4,4,4}.
    # Strassen for 2x2*2x2:
    # m1 = (a+b)(e+f)
    # m2 = (c+d)(e+f)  
    # m3 = a(h-f)
    # m4 = d(g-e)
    # m5 = (a+e)(h+f)
    # m6 = (c+a)(f-h)
    # m7 = (b+d)(g-c)
    # p11 = m1+m4-m5+m7
    # p12 = m5-m1+m2+m6
    # p21 = m1-m3+m4+m6
    # p22 = -m3+m2+m5+m7
    
    # For 4x4*4x4, apply Strassen recursively:
    # Split 4x4 into four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = [[C11, C12], [C21, C22]] where each block is 2x2
    # 
    # Using Strassen at the block level:
    # S1 = (A11+A22)(B11+B22)
    # S2 = (A21+A22)B11
    # S3 = A11(B21-B22)
    # S4 = A22(B11-B12)
    # S5 = (A11+A12)B22
    # S6 = (A21-A11)(B11+B12)
    # S7 = (A12-A22)(B21+B22)
    # 
    # C11 = S1+S4-S5+S7
    # C12 = S3+S5
    # C21 = S2+S4
    # C22 = S1-S2+S3+S6
    # 
    # Each S_i is a product of two 2x2 matrices, which itself requires 7 multiplications (Strassen).
    # So total rank = 7 * 7 = 49 for M_{4,4,4}.
    # 
    # That's worse than I thought. The recursive Strassen gives 49, not 7.
    # Strassen rank 7 is for M_{2,2,2}, not M_{4,4,4}.
    
    # So M_{3,4,4} has rank at most 49 (from recursive Strassen on M_{4,4,4}).
    # And M_{3,4,8} has rank at most 49 * 2 = 98. That's worse than schoolbook 96!
    
    # Let me reconsider. The schoolbook rank for M_{3,4,8} is 3*4*8 = 96.
    # I need to find a better approach.
    
    # Alternative: Direct decomposition without tensor product.
    # M_{3,4,8}: C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # Split j into j1 (0,1) and j2 (2,3):
    # C[i,k] = sum_{j1=0}^{1} A[i,j1]*B[j1,k] + sum_{j2=0}^{1} A[i,j2+2]*B[j2+2,k]
    # 
    # For the first sum: A[i,0:2] (1x2) * B[0:2,k] (2x8) -> scalar
    # This is M_{1,2,8} for each i, with 3 values of i.
    # M_{1,2,8} has schoolbook rank 16.
    # So total from first part: 3*16 = 48.
    # Similarly second part: 48. Total: 96. No improvement.
    
    # Let me try a randomized search for low-rank decompositions.
    # Start from rank 48 and try to go lower.
    
    def verify_decomposition(U, V, W, rank, tol=1e-8):
        """Verify that the decomposition is exact."""
        # Reshape for easier indexing
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # Check all i, j, k
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    best_rank = n * m * p  # 96
    best_result = None
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = (U_school, V_school, W_school)
    
    # Let me try a proper randomized search
    for target_rank in [48, 40, 36, 32, 28, 24, 20, 16, 14, 12, 10, 8]:
        if time.time() - start_time > deadline:
            break
        
        found = False
        for attempt in range(100):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Random initialization with small integers
            U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
            
            # Compute M[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for c in range(target_rank):
                        M[i, k, c] = np.dot(U[i*m:(i+1)*m, c], V[(0*p+k):(m*p+k):p, c])
            
            # Solve for W: M[i,k,:] dot W[k*n+i,:] = 1
            W = np.zeros((p*n, target_rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    mv = M[i, k, :]
                    norm_sq = np.dot(mv, mv)
                    if norm_sq < 1e-10:
                        valid = False
                        break
                    W[k*n+i, :] = mv / norm_sq
            if not valid:
                continue
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_decomposition(U_r, V_r, W_r, target_rank):
                best_rank = target_rank
                best_result = (U_r, V_r, W_r)
                found = True
                break
        
        if found:
            break
    
    # If no better found, use schoolbook
    if best_result is None or best_rank == n*m*p:
        best_rank = n * m * p
        U = np.zeros((n*m, best_rank))
        V = np.zeros((m*p, best_rank))
        W = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
