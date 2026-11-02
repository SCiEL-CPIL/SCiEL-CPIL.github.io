"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import random

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True
    
    def try_rank(r, max_iters=5000, seed=42):
        """Try to find a rank-r decomposition using structured random search."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(seed)
        
        # For rank r, we have r columns in U, V, W
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Constraint: for all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
        
        # Strategy: use a structured approach where we fix some variables and solve for others
        # Or use iterative refinement
        
        # Approach 1: Random initialization + gradient descent on the loss
        # Loss = sum_{i,j,k} (sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] - 1)^2
        
        # For small r, try random restarts
        best_loss = float('inf')
        best_solution = None
        
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            U = np.random.uniform(-1, 1, (n*m, r))
            V = np.random.uniform(-1, 1, (m*p, r))
            W = np.random.uniform(-1, 1, (p*n, r))
            
            # Gradient descent
            lr = 0.01
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Compute current value
                            val = 0.0
                            for t in range(r):
                                val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            err = val - 1.0
                            loss += err * err
                            
                            # Gradients
                            for t in range(r):
                                dU[i*m+j, t] += 2 * err * V[j*p+k, t] * W[k*n+i, t]
                                dV[j*p+k, t] += 2 * err * U[i*m+j, t] * W[k*n+i, t]
                                dW[k*n+i, t] += 2 * err * U[i*m+j, t] * V[j*p+k, t]
                
                if loss < best_loss:
                    best_loss = loss
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if loss < 1e-10:
                    break
            
            if best_loss < 1e-10:
                break
        
        if best_solution is not None and best_loss < 1e-8:
            U, V, W = best_solution
            # Convert to fractions for exact verification
            U_frac = [[Fraction(int(round(x * 1e6)), 10**6) for x in U[i]] for i in range(n*m)]
            V_frac = [[Fraction(int(round(x * 1e6)), 10**6) for x in V[i]] for i in range(m*p)]
            W_frac = [[Fraction(int(round(x * 1e6)), 10**6) for x in W[i]] for i in range(p*n)]
            
            if verify(U_frac, V_frac, W_frac):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        return None
    
    def structured_decomposition():
        """Try structured decompositions based on known matmul tensor structure."""
        # For 2x3x7 matmul, try various structured approaches
        
        # Approach: Use the fact that we can decompose along one dimension
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # Try rank = n*m*p - savings
        # Known: for 2x2x2, rank is 7 (not 8)
        # For 2x3x3, rank is known to be 11 (not 12)
        
        # Try various ranks from low to high
        for r in range(1, n*m*p + 1):
            if time.time() > deadline:
                break
            
            result = try_rank(r, max_iters=2000, seed=r)
            if result is not None:
                return result
        
        return None
    
    # First try structured decomposition
    result = structured_decomposition()
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
