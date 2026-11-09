import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # The 2x3 * 3x9 multiplication can be viewed as:
    # For each output row i (0,1) and column k (0..8):
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor has structure: T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    # 
    # Key insight: We can decompose this using the fact that the 3x9 matrix
    # can be seen as three 3x3 blocks. For each block s (0,1,2), we have
    # B[j, s*3+k] for k=0,1,2.
    #
    # A rank-27 solution exists by treating each (i,k) pair independently
    # but sharing the B factor structure.
    #
    # Actually, let's think more carefully. The standard schoolbook is rank 54.
    # But we can do better by exploiting the structure.
    #
    # For 2x3 * 3x9, the rank is at most 2*9*3 = 54 (schoolbook)
    # but potentially lower.
    #
    # Let's try a direct construction. The tensor T has entries:
    # T[i*m+j, j*p+k, k*n+i] = 1
    #
    # We need rank-r factors U (6 x r), V (27 x r), W (18 x r) such that:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    #
    # A known approach: use the fact that matrix multiplication rank for
    # (n x m) * (m x p) is at most n*m*p, but can be lower.
    #
    # For 2x3 * 3x9, let's try rank 27 (which is 3 * 9 = m * p).
    #
    # Actually, a simpler approach: use the identity that if we can find
    # rank-r decomposition for smaller cases, we can combine them.
    #
    # Let's just try to find a rank-27 solution directly.
    # The key is that for each (j,k) pair, we need the sum over i to work.
    
    # Try rank 27 first
    r = 27
    
    # Construction: 
    # U[i*m+j, t] where t = (j, k) for k in 0..2, but we need 27 columns
    # Let's use t = (j, k, s) where j in 0..2, k in 0..8, s in 0..2
    # That's 3*9*1 = 27... no, 3*9 = 27
    
    # Let me think differently. 
    # For rank m*p = 27, let t index (j, k) for j in 0..2, k in 0..8
    # U[i*m+j, (j',k')] = delta_{j,j'} * something
    # V[j*p+k, (j',k')] = delta_{j,j'} * delta_{k,k'} * something
    # W[k*n+i, (j',k')] = something
    
    # This is getting complex. Let's use a numerical approach but with
    # a smarter initialization based on the structure.
    
    # Actually, let's just return the schoolbook rank-54 solution which is guaranteed correct.
    # The task says score is the rank, and we want minimal rank.
    # But if we can't find a better one quickly, rank 54 is safe.
    
    # Let's try a hybrid: attempt rank 27 with a structured initialization,
    # and if it fails, fall back to rank 54.
    
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    target_indices = np.array(target_indices)
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find rank-27 solution
    # Use a structured initialization
    np.random.seed(42)
    
    for r in [27, 28, 29, 30]:
        if time.time() > 140:
            break
        
        found = False
        for restart in range(5):
            if time.time() > 145:
                break
            
            # Structured initialization
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            if r == 27:
                # t = (j, k) where j in 0..2, k in 0..8
                for j in range(m):
                    for k in range(p):
                        t = j * p + k
                        # U[i*m+j, t] should be 1 for all i
                        for i in range(n):
                            U[i*m + j, t] = 1.0
                        # V[j*p+k, t] = 1
                        V[j*p + k, t] = 1.0
                        # W[k*n+i, t] should be 1/n for all i to get sum = 1
                        for i in range(n):
                            W[k*n + i, t] = 1.0 / n
            
            # Add small random perturbation
            U += np.random.randn(n*m, r) * 0.01
            V += np.random.randn(m*p, r) * 0.01
            W += np.random.randn(p*n, r) * 0.01
            
            # Verify if this is already a solution
            err = 0.0
            for (a, b, c) in target_indices:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err += (val - 1.0) ** 2
            
            if err < 1e-9:
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break
    
    if best_U is None:
        # Fall back to schoolbook rank 54
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
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
