"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product, combinations
from fractions import Fraction
import random

def construct():
    n, m, p = DIMENSIONS
    
    # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Strategy: Fix W to have simple structure, then solve for U and V
    
    # Lower bound on rank
    lower_bound = max(n*m, m*p, p*n)  # 24
    
    for R in range(lower_bound, 73):
        # Try to find a decomposition of rank R
        # Approach: Fix W entries to be in {-1, 0, 1, 0.5, -0.5}
        # Then for each (i,j,k), the equation becomes linear in U and V
        
        # Better approach: Use the fact that we can set W[k*n+i, r] = 1 for specific r
        # and 0 otherwise, which makes the problem separable
        
        # Try: for each r, assign W[k*n+i, r] = 1 for a subset of (k,i) pairs
        # This means each rank component contributes to a subset of output entries
        
        # Alternative: Use random search with verification
        random.seed(R)
        
        # Try multiple random initializations
        for attempt in range(200):
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U = np.array([[random.choice(vals) for _ in range(R)] for _ in range(n*m)])
            V = np.array([[random.choice(vals) for _ in range(R)] for _ in range(m*p)])
            W = np.array([[random.choice(vals) for _ in range(R)] for _ in range(p*n)])
            
            # Check all tensor equations
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = U[i*m + j]
                        vj = V[j*p + k]
                        wk = W[k*n + i]
                        val = np.dot(ui * vj, wk)
                        if abs(val - 1.0) > 1e-9:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if success:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If random search fails, try a structured approach
        # Set W[k*n+i, r] = 1 if (k,i) is in some pattern, else 0
        # This reduces to finding U and V such that for each (i,j,k):
        # sum_{r: W[k*n+i,r]=1} U[i*m+j,r] * V[j*p+k,r] = 1
        
        # For simplicity, try W[k*n+i, r] = 1 if r == k*n+i (but this gives rank p*n=18)
        # Then U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all i,j,k
        # This requires U[i*m+j, k*n+i] = V[j*p+k, k*n+i] = 1, which works!
        # But rank would be 18, which is below lower bound 24 - impossible
        
        # The issue: for fixed (i,j), we need sum over k of A[i,j]*B[j,k]
        # Each (i,j,k) triple must be covered
        
        pass
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
