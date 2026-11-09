"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = 3, 3, 7
    
    # Rank-23 decomposition for 3x3x3 matrix multiplication
    terms_3x3 = [
        ([1,0,0],[1,0,0],[1,0,0]),
        ([0,1,0],[0,1,0],[0,1,0]),
        ([0,0,1],[0,0,1],[0,0,1]),
        ([1,1,0],[1,0,0],[0,1,0]),
        ([1,1,0],[0,1,0],[1,0,0]),
        ([1,1,0],[0,0,1],[0,0,1]),
        ([1,0,1],[1,0,0],[0,0,1]),
        ([1,0,1],[0,1,0],[0,1,0]),
        ([1,0,1],[0,0,1],[1,0,0]),
        ([0,1,1],[1,0,0],[0,0,1]),
        ([0,1,1],[0,1,0],[1,0,0]),
        ([0,1,1],[0,0,1],[0,1,0]),
        ([1,-1,0],[1,0,0],[0,1,0]),
        ([1,-1,0],[0,1,0],[1,0,0]),
        ([1,-1,0],[0,0,1],[0,0,1]),
        ([0,1,-1],[1,0,0],[0,0,1]),
        ([0,1,-1],[0,1,0],[0,1,0]),
        ([0,1,-1],[0,0,1],[1,0,0]),
        ([1,0,-1],[1,0,0],[1,0,0]),
        ([1,0,-1],[0,1,0],[0,0,1]),
        ([1,0,-1],[0,0,1],[0,1,0]),
        ([0,-1,1],[1,0,0],[0,1,0]),
        ([0,-1,1],[0,1,0],[0,0,1]),
    ]
    
    # Verify 3x3 decomposition
    T = np.zeros((3,3,3))
    for u,v,w in terms_3x3:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T[i,j,k] += u[i]*v[j]*w[k]
    
    valid_3x3 = np.allclose(T, np.ones((3,3,3)), atol=1e-10)
    
    if valid_3x3:
        # Block decomposition: 3x7 = 3x3 + 3x3 + 3x1
        rank_total = 23 + 23 + 9
        U = np.zeros((n*m, rank_total))
        V = np.zeros((m*p, rank_total))
        W = np.zeros((p*n, rank_total))
        
        # First 3x3 block (k=0,1,2)
        offset1 = 0
        for r, (u,v,w) in enumerate(terms_3x3):
            col = offset1 + r
            for i in range(n):
                for j in range(m):
                    U[i*m + j, col] = u[i]
            for j in range(m):
                for k in range(3):
                    V[j*p + k, col] = v[j]
            for k in range(3):
                for i in range(n):
                    W[k*n + i, col] = w[k]
        
        # Second 3x3 block (k=3,4,5)
        offset2 = 23
        for r, (u,v,w) in enumerate(terms_3x3):
            col = offset2 + r
            for i in range(n):
                for j in range(m):
                    U[i*m + j, col] = u[i]
            for j in range(m):
                for k_loc in range(3):
                    k = 3 + k_loc
                    V[j*p + k, col] = v[j]
            for k_loc in range(3):
                k = 3 + k_loc
                for i in range(n):
                    W[k*n + i, col] = w[k_loc]
        
        # Third block: 3x1 (k=6), schoolbook rank 9
        offset3 = 46
        for i in range(n):
            for j in range(m):
                col = offset3 + i*m + j
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
        
        # Verify full decomposition
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 63
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
