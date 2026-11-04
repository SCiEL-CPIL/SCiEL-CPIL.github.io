import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_fb, V_fb, W_fb, best_rank = schoolbook()
    
    # Build 2x3x3 rank-5 tensor from 3x3x3 Strassen
    # 3x3x3 matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T[i,j,k] = delta(i,k) for the multiplication
    # Strassen rank-5 for 3x3x3:
    # We use the known decomposition. Let's construct it directly.
    # 
    # For 3x3x3, the 5 terms (using standard Strassen-like):
    # Term 0: U=diag(1,1,0), V=diag(1,1,0), W=diag(1,1,0) -> handles (0,0),(0,1),(1,0),(1,1)
    # This is getting complex. Let me use a verified rank-5 for 2x3x3 directly.
    #
    # Actually, let's construct 2x3x3 decomposition from first principles using the 
    # fact that rank(2x3x3) <= rank(3x3x3) = 5 by restricting indices.
    #
    # Standard Strassen for 3x3x3 (one common form):
    # The 5 rank-1 tensors have these support patterns:
    # 
    # Let me use a verified construction. For 2x3x3:
    # T[i,j,k] = 1 iff i==k, for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # 
    # I'll use the following rank-5 decomposition (verified):
    # Using the approach: 2x3x3 can be embedded in 3x3x3, so rank <= 5.
    # Let me construct it via the Kronecker idea directly.
    
    # Kronecker approach: 2x3x9 = (2x3x3) ⊗ I_3
    # If 2x3x3 has rank r, then 2x3x9 has rank r*3.
    # 
    # For 2x3x3, let's find a rank-5 decomposition:
    # T[i,j,k] = delta_{i,k} for i in {0,1}
    # 
    # Decomposition using 5 terms:
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,k}
    # 
    # One construction: use the 3x3x3 rank-5 and project.
    # Standard 3x3x3 rank-5 (from literature):
    # Let me use a concrete one.
    
    # Concrete rank-5 for 3x3x3 matrix multiplication tensor:
    # U (3x3 x 5), V (3x3 x 5), W (3x3 x 5)
    # 
    # Using the decomposition from:
    # T = sum_{r=0}^{4} u_r ⊗ v_r ⊗ w_r
    # 
    # A known rank-5 for 3x3x3:
    # (This is the standard Strassen-style for 3x3)
    
    # Let me just use a verified small construction for 2x3x3:
    # 
    # For 2x3x3, the tensor is:
    # T[0,j,0] = 1, T[0,j,1] = 0, T[0,j,2] = 0  for j=0,1,2
    # T[1,j,0] = 0, T[1,j,1] = 1, T[1,j,2] = 0  for j=0,1,2
    # T[i,j,k] = 0 otherwise
    #
    # So T[0,j,0] = 1 for all j, T[1,j,1] = 1 for all j.
    #
    # Simple rank-3: 
    #   Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1
    #   Term 1: U[0,1]=1, V[1,0]=1, W[0,0]=1  -> wait, W[k,i] so W[0,0]
    #   This doesn't work cleanly.
    #
    # Let me think more carefully.
    # T[i,j,k] = 1 iff i=k
    # For i=0: T[0,j,0]=1 for j=0,1,2
    # For i=1: T[1,j,1]=1 for j=0,1,2
    #
    # Rank-3 decomposition:
    # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> contributes to T[0,0,0]
    # Term 1: U[0,1]=1, V[1,0]=1, W[0,0]=1  -> contributes to T[0,1,0]  
    # Term 2: U[0,2]=1, V[2,0]=1, W[0,0]=1  -> contributes to T[0,2,0]
    # Term 3: U[1,0]=1, V[0,1]=1, W[1,1]=1  -> contributes to T[1,0,1]
    # Term 4: U[1,1]=1, V[1,1]=1, W[1,1]=1  -> contributes to T[1,1,1]
    # Term 5: U[1,2]=1, V[2,1]=1, W[1,1]=1  -> contributes to T[1,2,1]
    # That's rank 6, not 5.
    #
    # For rank 5, we need to combine some terms.
    # 
    # Key insight: W[0,0] appears in terms 0,1,2 and W[1,1] in terms 3,4,5.
    # We can factor: 
    #   For i=0 part: U[0,:] = [1,1,1,0,0,0], V[:,0] = [1,1,1,0,0,0]^T, W[0,0]=1
    #   This is rank 1 for the i=0 slice if we could, but U and V have different structures.
    #
    # Actually, the i=0 slice is: U_0[j] * V_0[j,k] * W_0[k,i]
    # For i=0, k must be 0: U_0[j] * V_0[j,0] * W_0[0,0] = 1 for all j
    # So U_0[j] * V_0[j,0] = 1/W_0[0,0]. If W_0[0,0]=1, then U_0[j]*V_0[j,0]=1.
    # This is rank 1 in the (j) dimension!
    #
    # Similarly for i=1, k=1: U_1[j] * V_1[j,1] * W_1[1,1] = 1 for all j
    #
    # So we can do:
    # Term 0: U[0,j]=1 for j=0,1,2; V[j,0]=1 for j=0,1,2; W[0,0]=1
    #          -> T[0,j,0] += 1 for all j. 
    #          But also T[0,j,k] for k!=0 gets 0 since W[0,k]=0 for k!=0. Good.
    #          Also T[i,j,k] for i!=0 gets 0 since U[i,j]=0 for i!=0. Good.
    #
    # Term 1: U[1,j]=1 for j=0,1,2; V[j,1]=1 for j=0,1,2; W[1,1]=1
    #          -> T[1,j,1] += 1 for all j.
    #
    # That's rank 2! Let me verify:
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # For i=0, k=0: U[0,j,0]*V[j,0,0]*W[0,0,0] + U[0,j,1]*V[j,0,1]*W[0,0,1]
    #   = 1*1*1 + 0*0*0 = 1. Correct!
    # For i=0, k=1: U[0,j,0]*V[j,1,0]*W[1,0,0] + U[0,j,1]*V[j,1,1]*W[1,0,1]
    #   = 1*1*0 + 0*1*0 = 0. Correct!
    # For i=1, k=1: U[1,j,0]*V[j,1,0]*W[1,1,0] + U[1,j,1]*V[j,1,1]*W[1,1,1]
    #   = 0*1*0 + 1*1*1 = 1. Correct!
    # For i=1, k=0: U[1,j,0]*V[j,0,0]*W[0,1,0] + U[1,j,1]*V[j,0,1]*W[0,1,1]
    #   = 0*1*0 + 1*0*0 = 0. Correct!
    #
    # So 2x3x3 has rank 2! Then 2x3x9 = (2x3x3) ⊗ I_3 has rank 2*3 = 6.
    
    # Let me verify this rank-2 for 2x3x3:
    n2, m2, p2 = 2, 3, 3
    U233 = np.zeros((n2*m2, 2))  # 6x2
    V233 = np.zeros((m2*p2, 2))  # 9x2
    W233 = np.zeros((p2*n2, 2))  # 6x2
    
    # Term 0: i=0, k=0
    for j in range(3):
        U233[0*3+j, 0] = 1  # U[0,j,0] = 1
        V233[j*3+0, 0] = 1  # V[j,0,0] = 1
    W233[0*2+0, 0] = 1  # W[0,0,0] = 1
    
    # Term 1: i=1, k=1
    for j in range(3):
        U233[1*3+j, 1] = 1  # U[1,j,1] = 1
        V233[j*3+1, 1] = 1  # V[j,1,1] = 1
    W233[1*2+1, 1] = 1  # W[1,1,1] = 1
    
    # Verify 2x3x3
    T233 = np.einsum('ur,vr,wr->uvw', U233, V233, W233)
    T233_target = np.zeros((6, 9, 6))
    for i in range(2):
        for j in range(3):
            for k in range(3):
                if i == k:
                    T233_target[i*3+j, j*3+k, k*2+i] = 1.0
    assert np.allclose(T233, T233_target, atol=1e-10), "2x3x3 rank-2 verification failed"
    
    # Now Kronecker with I_3 to get 2x3x9
    # 2x3x9: multiply 2x3 by 3x9
    # T[i,j,k] = delta_{i, k mod 3} for the block structure
    # Actually, 3x9 = 3x(3*3), so k = k_block * 3 + k_inner
    # The multiplication is: C[i, k_block*3+k_inner] = sum_j A[i,j] * B[j, k_block*3+k_inner]
    # This is (2x3 x 3x3) ⊗ I_3 where the 3x3 is the inner block and I_3 is the block index
    #
    # More precisely: B is 3x9, so B[j, K] where K = 3*kb + ki for kb in {0,1,2}, ki in {0,1,2}
    # C[i, K] = sum_j A[i,j] * B[j, K]
    # This is exactly: for each block kb, C[:, kb*3:kb*3+3] = A @ B[:, kb*3:kb*3+3]
    # So it's 3 independent 2x3x3 multiplications, one for each block.
    #
    # Kronecker: rank = rank(2x3x3) * 3 = 2 * 3 = 6
    
    rank_kron = 6
    U_kron = np.zeros((n*m, rank_kron))  # 6x6
    V_kron = np.zeros((m*p, rank_kron))  # 27x6
    W_kron = np.zeros((p*n, rank_kron))  # 18x6
    
    # For each block b in {0,1,2} and each term r in {0,1}:
    # column = b*2 + r
    for b in range(3):
        for r in range(2):
            col = b * 2 + r
            # U: same as U233 but only for the relevant i
            for i in range(2):
                for j in range(3):
                    U_kron[i*3+j, col] = U233[i*3+j, r]
            
            # V: V233[j, ki] for the inner index, but now k = b*3 + ki
            # V[j*p + k, col] where k = b*3 + ki
            for j in range(3):
                for ki in range(3):
                    k = b * 3 + ki
                    V_kron[j*9 + k, col] = V233[j*3 + ki, r]
            
            # W: W233[ki*n + i, r] -> W[k*n + i, col] where k = b*3 + ki
            for ki in range(3):
                for i in range(2):
                    k = b * 3 + ki
                    W_kron[k*2 + i, col] = W233[ki*2 + i, r]
    
    # Verify 2x3x9
    T_calc = np.einsum('ur,vr,wr->uvw', U_kron, V_kron, W_kron)
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # k = b*3 + ki, need i == ki for the block b
                ki = k % 3
                if i == ki:
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    if np.allclose(T_calc, T_target, atol=1e-10):
        return {"U": U_kron.tolist(), "V": V_kron.tolist(), "W": W_kron.tolist()}
    
    # Fallback: schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
