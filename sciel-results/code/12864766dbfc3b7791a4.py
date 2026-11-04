import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = 110  # 110 seconds to leave buffer
    
    n, m, p = DIMENSIONS
    # Tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Try to find a rank 68 decomposition
    # Structure: Rank 23 for 3x3x3 part, Rank 45 for 3x3x5 part
    # This is a hypothesis. We will try to optimize a rank 68 decomposition.
    
    rank = 68
    
    # Initialize with schoolbook for the first 3 cols and last 5 cols?
    # Schoolbook rank is 72.
    # We try to compress 4 ranks.
    
    # Initialize U, V, W
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We use a random integer initialization for the search
    np.random.seed(42)
    
    # Initialize with schoolbook values for the first 72, then truncate?
    # Better: Initialize with zeros and set some to 1.
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Fill with schoolbook pattern for the first 68 columns?
    # Schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # We map the 72 schoolbook terms to 68 slots.
    # This is not straightforward.
    
    # Alternative: Use the block decomposition.
    # Part 1: 3x3x3 (Rank 23). Part 2: 3x3x5 (Rank 45).
    # We need the Strassen 3x3x3 coefficients.
    # Since I don't have them hardcoded, I will try to find them numerically for the 3x3x3 sub-problem.
    
    # Sub-problem 1: 3x3x3
    T1 = np.ones((3, 3, 3))
    rank1 = 23
    U1 = np.zeros((9, rank1))
    V1 = np.zeros((9, rank1))
    W1 = np.zeros((9, rank1))
    
    # Initialize U1, V1, W1 with random values
    U1[:] = np.random.randn(9, rank1)
    V1[:] = np.random.randn(9, rank1)
    W1[:] = np.random.randn(9, rank1)
    
    def error1(params):
        u, v, w = np.split(params, [9*rank1, 18*rank1])
        u = u.reshape(9, rank1)
        v = v.reshape(9, rank1)
        w = w.reshape(9, rank1)
        # Compute tensor
        # T[i,j,k] = sum_r u[i,r] v[j,r] w[k,r]
        # Use einsum
        T_est = np.einsum('ir,jr,kr->ijk', u, v, w)
        return np.sum((T1 - T_est)**2)
    
    x0 = np.concatenate([U1.flatten(), V1.flatten(), W1.flatten()])
    
    if time.time() - start_time < deadline - 50:
        res1 = minimize(error1, x0, method='L-BFGS-B', options={'maxiter': 100})
        U1, V1, W1 = np.split(res1.x, [9*rank1, 18*rank1])
        U1 = U1.reshape(9, rank1)
        V1 = V1.reshape(9, rank1)
        W1 = W1.reshape(9, rank1)
        
        # Round to nearest integer
        U1 = np.round(U1)
        V1 = np.round(V1)
        W1 = np.round(W1)
        
        # Verify
        T1_est = np.einsum('ir,jr,kr->ijk', U1, V1, W1)
        if not np.allclose(T1_est, T1, atol=1e-6):
            # Fallback to schoolbook for this part
            U1 = np.zeros((9, 27))
            V1 = np.zeros((9, 27))
            W1 = np.zeros((9, 27))
            col = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U1[i*3+j, col] = 1
                        V1[j*3+k, col] = 1
                        W1[k*3+i, col] = 1
                        col += 1
            rank1 = 27

    # Sub-problem 2: 3x3x5
    T2 = np.ones((3, 3, 5))
    rank2 = 45
    # Use schoolbook for this part
    U2 = np.zeros((9, rank2))
    V2 = np.zeros((15, rank2))
    W2 = np.zeros((15, rank2))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(5):
                U2[i*3+j, col] = 1
                V2[j*5+k, col] = 1
                W2[k*3+i, col] = 1
                col += 1
                
    total_rank = rank1 + rank2
    
    # Construct full U, V, W
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Fill Part 1 (cols 0-2 of B, so k in 0..2)
    # U1 is 9xR1. U is 9xTotalRank.
    U[:, :rank1] = U1
    
    # V1 is 9xR1 (j in 0..2, k in 0..2). V is 24xTotalRank (j in 0..2, k in 0..7).
    # V1 corresponds to k=0,1,2.
    # V[j*p + k, r]
    # V1[j*3 + k, r]
    for r in range(rank1):
        for j in range(3):
            for k in range(3):
                V[j*8 + k, r] = V1[j*3 + k, r]
                
    # W1 is 9xR1 (k in 0..2, i in 0..2). W is 24xTotalRank (k in 0..7, i in 0..2).
    # W1[k*3 + i, r]
    # W[k*3 + i, r]
    for r in range(rank1):
        for k in range(3):
            for i in range(3):
                W[k*3 + i, r] = W1[k*3 + i, r]
                
    # Fill Part 2 (cols 3-7 of B, so k in 3..7)
    offset = rank1
    U[:, offset:offset+rank2] = U2
    
    # V2 is 15xR2 (j in 0..2, k in 0..4). V is 24xTotalRank.
    # V2[j*5 + k, r]
    # V[j*8 + (3+k), r]
    for r in range(rank2):
        for j in range(3):
            for k in range(5):
                V[j*8 + 3 + k, offset + r] = V2[j*5 + k, r]
                
    # W2 is 15xR2 (k in 0..4, i in 0..2). W is 24xTotalRank.
    # W2[k*3 + i, r]
    # W[(3+k)*3 + i, r]
    for r in range(rank2):
        for k in range(5):
            for i in range(3):
                W[(3+k)*3 + i, offset + r] = W2[k*3 + i, r]

    # Final Verification
    T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_est, T, atol=1e-6):
        # Fallback to full schoolbook
        rank = 72
        U = np.zeros((9, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(8):
                    U[i*3+j, col] = 1
                    V[j*8+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
