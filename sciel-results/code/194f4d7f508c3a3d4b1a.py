import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper function to verify a solution
    def verify(U, V, W, r):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r_idx in range(r):
                        val += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if val != 1:
                        return False
        return True

    # Try to find a decomposition of rank r
    # Lower bound from flattening is max(n*m, m*p, p*n) = 12
    # Known bounds suggest rank is around 15-18 for 3x3x4
    
    max_rank = n * m * p
    
    # Try different ranks starting from a reasonable lower bound
    for r in range(15, min(max_rank + 1, 25)):
        # Try to find a solution using structured search
        # We use a deterministic approach with small integer entries
        
        # Initialize with a structured pattern
        found = False
        
        # Try various patterns for small r
        # Pattern 1: Diagonal-like with some off-diagonal terms
        for attempt in range(50):
            # Create random sparse matrices with small integer entries
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            # Fill with random small integers
            for i in range(n * m):
                for j in range(r):
                    if random.random() < 0.3:
                        U[i][j] = random.choice([-2, -1, 1, 2])
            
            for j in range(m * p):
                for k in range(r):
                    if random.random() < 0.3:
                        V[j][k] = random.choice([-2, -1, 1, 2])
            
            for k in range(p * n):
                for i in range(r):
                    if random.random() < 0.3:
                        W[k][i] = random.choice([-2, -1, 1, 2])
            
            # Verify
            if verify(U, V, W, r):
                return {"U": U, "V": V, "W": W}
        
        # If random search fails, try a more systematic approach
        # Use the fact that we need sum_r U[i,j,r] V[j,k,r] W[i,k,r] = 1
        
        # Try a construction based on polynomial interpolation
        # For 3x3x4, we can try rank 16 or 17
        
        if r >= 16:
            # Try a structured construction
            # Use the identity: sum over r of products = 1
            # This is equivalent to finding a tensor decomposition
            
            # Try a specific pattern for r=16
            if r == 16:
                # Construct using a known pattern
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                # Try a cyclic pattern
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = (i * m + j) * p + k
                            if idx < r:
                                U[i * m + j][idx] = 1
                                V[j * p + k][idx] = 1
                                W[k * n + i][idx] = 1
                
                if verify(U, V, W, r):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: Schoolbook algorithm
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
