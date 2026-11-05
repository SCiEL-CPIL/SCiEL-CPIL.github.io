import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # We seek: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # U: (9, rank), V: (21, rank), W: (21, rank)
    
    # Try rank 21: one term per (i,k) output position, with j-summed structure
    # For term r corresponding to (i_r, k_r):
    #   U[i*m+j, r] = 1 if i == i_r else 0  (for all j)
    #   V[j*p+k, r] = 1 if k == k_r else 0  (for all j)  
    #   W[k'*n+i', r] = delta[j... ] -- this doesn't work directly
    
    # Better: use rank = m * p = 21, with terms indexed by (j, k):
    # Term (j,k): 
    #   U[i*m+j', r] = delta[j,j'] * (some function of i)
    #   V[j'*p+k', r] = delta[j,j'] * delta[k,k']
    #   W[k'*n+i', r] = (some function of i')
    # This needs: sum_{j'} delta[j,j'] * f(i,j') * delta[k,k'] * g(i',k') = 1 for all i,j,k
    # => f(i,j) * g(i,k) = 1 for all i,j,k -- impossible unless f,g are constants
    
    # Correct approach: schoolbook is rank 63. Let me try to find a lower rank.
    # Use alternating least squares or direct construction.
    
    # For now, return a valid rank-21 decomposition using the following trick:
    # The 3x7 product can be seen as 3 independent 1x7 row products.
    # Row i of C = Row i of A times B.
    # Row i of A has 3 elements. B has 3x7.
    # So C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:]
    # This is 3 additions of 1x7 vectors per row, 3 rows = 9 vector additions of length 7
    # Each vector addition of length 7 is a bilinear form of rank 7 (schoolbook for 1x3 * 3x7)
    # Total: 3*7 = 21 terms
    
    # Term (i, j, k) for i in 0..2, j in 0..2, k in 0..6:
    # U[i*m+j, r] = 1 if (i,j) matches, else 0
    # V[j'*p+k', r] = 1 if (j',k') = (j,k), else 0
    # W[k'*n+i', r] = 1 if (k',i') = (k,i), else 0
    # This gives rank 3*3*7 = 63 again.
    
    # To get rank 21, we need to share. The key: for fixed i, the 3 terms (i,0,k), (i,1,k), (i,2,k) 
    # for a fixed k can be combined if we allow U to depend on i and j simultaneously.
    
    # Actually, rank 21 is achievable with this construction:
    # For each (i, k) pair (21 pairs), create one term:
    #   U[i*m+j, r_{i,k}] = 1 for all j  (U row for (i,j) is 1 for this term)
    #   V[j*p+k, r_{i,k}] = 1 for all j  (V row for (j,k) is 1 for this term)
    #   W[k'*n+i', r_{i,k}] = 1 if k'=k and i'=i, else 0
    # Check: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    #        = sum_{(i',k')} [1 if i==i'] * [1 if k==k'] * [1 if i==i' and k==k']
    #        = 1 (only the term (i,k) contributes)
    # YES! This works! Rank = 21.
    
    rank = n * p  # 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for k in range(p):
            # U: for all j, U[i*m+j, col] = 1
            for j in range(m):
                U[i * m + j, col] = 1
            # V: for all j, V[j*p+k, col] = 1
            for j in range(m):
                V[j * p + k, col] = 1
            # W: W[k*n+i, col] = 1, all others 0
            W[k * n + i, col] = 1
            col += 1
    
    # Verify
    if time.time() < deadline - 10:
        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        if not np.allclose(T_recon, 1.0, atol=1e-9):
            # Fallback to schoolbook rank 63
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, c] = 1
                        V_fb[j*p+k, c] = 1
                        W_fb[k*n+i, c] = 1
                        c += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
