import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 130

    def make_rank28():
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 2, r] = 1
                V[2 * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
        
        # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
        for i in range(n):
            for j in range(2):
                U[i * m + j, r] = 1
                V[j * p + 4, r] = 1
                W[4 * n + i, r] = 1
                r += 1
        
        # Part 3: Strassen for A1*B11 - 7 terms
        U[0, r] = 1; U[4, r] = 1; V[0, r] = 1; V[6, r] = 1; W[0, r] = 1; W[3, r] = 1; r += 1
        U[3, r] = 1; U[4, r] = 1; V[0, r] = 1; W[1, r] = 1; W[3, r] = -1; r += 1
        U[0, r] = 1; V[1, r] = 1; V[7, r] = -1; W[2, r] = 1; W[3, r] = 1; r += 1
        U[4, r] = 1; V[5, r] = 1; V[0, r] = -1; W[0, r] = 1; W[1, r] = 1; r += 1
        U[0, r] = 1; U[1, r] = 1; V[6, r] = 1; W[0, r] = -1; W[2, r] = 1; r += 1
        U[3, r] = 1; U[0, r] = -1; V[0, r] = 1; V[1, r] = 1; W[3, r] = 1; r += 1
        U[1, r] = 1; U[4, r] = -1; V[5, r] = 1; V[6, r] = 1; W[0, r] = 1; r += 1
        
        # Part 4: Strassen for A1*B12 - 7 terms
        U[0, r] = 1; U[4, r] = 1; V[2, r] = 1; V[8, r] = 1; W[4, r] = 1; W[7, r] = 1; r += 1
        U[3, r] = 1; U[4, r] = 1; V[2, r] = 1; W[5, r] = 1; W[7, r] = -1; r += 1
        U[0, r] = 1; V[3, r] = 1; V[9, r] = -1; W[6, r] = 1; W[7, r] = 1; r += 1
        U[4, r] = 1; V[7, r] = 1; V[2, r] = -1; W[4, r] = 1; W[5, r] = 1; r += 1
        U[0, r] = 1; U[1, r] = 1; V[8, r] = 1; W[4, r] = -1; W[6, r] = 1; r += 1
        U[3, r] = 1; U[0, r] = -1; V[2, r] = 1; V[3, r] = 1; W[7, r] = 1; r += 1
        U[1, r] = 1; U[4, r] = -1; V[7, r] = 1; V[8, r] = 1; W[4, r] = 1; r += 1
        
        return U, V, W, rank
    
    def validate(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(rank):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    U, V, W, best_rank = make_rank28()
    
    # Try to find rank 27 by checking if any two terms can be merged
    # This is a heuristic: try all pairs and see if their sum can be expressed as one term
    # For now, just return the rank-28 solution
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
