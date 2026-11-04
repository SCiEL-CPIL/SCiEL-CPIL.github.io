import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j  # Note: In task, A is n x m, so index is i*m + j?
                # Task says: "multiplying a 2 by 4 matrix by a 4 by 5 matrix"
                # A is 2x4 (n=2, m=4). B is 4x5 (m=4, p=5).
                # Standard convention: A[i,j], B[j,k].
                # Index a for A: i * m + j.
                # Index b for B: j * p + k.
                # Index c for C: k * n + i? 
                # Task says: "output factor follows the official trace convention W[k*n+i,r]"
                # This implies C index is k*n + i.
                # So C[k,i] = sum_j A[i,j] B[j,k].
                # This matches standard matrix multiplication C = A @ B.
                # Let's stick to the task's index definitions.
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Helper to check error
    def check_error(U, V, W):
        max_err = 0.0
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    max_err = max(max_err, abs(val - 1.0))
        # Check some zero entries
        # We need to ensure that for (a,b,c) not in the support, sum is 0.
        # Support size is n*m*p = 40.
        # Total entries in tensor: (2*4)*(4*5)*(5*2) = 8*20*10 = 1600.
        # We can check all 1600 entries if it's fast enough.
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if T[a,b,c] == 0:
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        max_err = max(max_err, abs(val))
        return max_err

    # Try ranks
    for rank in range(20, 36):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations with structure
        for trial in range(200):
            if time.time() > deadline - 2:
                break
                
            # Initialize with small random values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 2:
                    break
                
                # Update U
                VW = V @ W.T
                for a in range(n*m):
                    try:
                        U[a,:] = np.linalg.lstsq(VW, T[a,:,:].reshape(-1), rcond=None)[0]
                    except:
                        pass
                
                # Update V
                # T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r])
                # Let P_r = U[:,r] * W[:,r]
                # T[:,b,:] flattened = [P_0, ..., P_{rank-1}] @ V[b,:]^T
                P_flat = np.array([np.outer(U[:,r], W[:,r]).flatten() for r in range(rank)])
                for b in range(m*p):
                    target = T[:,b,:].flatten()
                    try:
                        V[b,:] = np.linalg.lstsq(P_flat.T, target, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                Q_flat = np.array([np.outer(U[:,r], V[:,r]).flatten() for r in range(rank)])
                for c in range(p*n):
                    target = T[:,:,c].flatten()
                    try:
                        W[c,:] = np.linalg.lstsq(Q_flat.T, target, rcond=None)[0]
                    except:
                        pass
            
            err = check_error(U, V, W)
            
            if err < 1e-6:
                # Found a low-rank decomposition! Try to round to rationals
                U_frac = [[Fraction(U[a,r]).limit_denominator(1000) for r in range(rank)] for a in range(n*m)]
                V_frac = [[Fraction(V[b,r]).limit_denominator(1000) for r in range(rank)] for b in range(m*p)]
                W_frac = [[Fraction(W[c,r]).limit_denominator(1000) for r in range(rank)] for c in range(p*n)]
                
                # Verify the rounded version
                max_err_rounded = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = sum(float(U_frac[a][r]) * float(V_frac[b][r]) * float(W_frac[c][r]) for r in range(rank))
                            max_err_rounded = max(max_err_rounded, abs(val - 1.0))
                
                if max_err_rounded < 1e-9:
                    U_out = [[float(x) for x in row] for row in U_frac]
                    V_out = [[float(x) for x in row] for row in V_frac]
                    W_out = [[float(x) for x in row] for row in W_frac]
                    return {"U": U_out, "V": V_out, "W": W_out}

    # If no better found, return schoolbook
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
