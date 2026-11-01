import numpy as np
from fractions import Fraction
import itertools

def construct():
    n, m, p = 2, 3, 7
    
    # We want to find a rank-r decomposition of the matrix multiplication tensor
    # T[i,j,k] = 1 for the multiplication (i in [0,n), j in [0,m), k in [0,p))
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We'll try to find a decomposition with rank < n*m*p = 42
    # Using a greedy approach with rational numbers
    
    # First, let's try the schoolbook decomposition as a baseline
    rank_schoolbook = n * m * p  # 42
    
    # Try to find a lower rank decomposition
    # We'll use a systematic search for small ranks
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # For each candidate rank, try to find a decomposition
    # We'll use a simple iterative approach
    
    def verify_decomposition(U, V, W, r):
        """Verify that U, V, W form a valid decomposition"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    expected = 1
                    actual = sum(U[a][rr] * V[b][rr] * W[c][rr] for rr in range(r))
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try ranks from 20 up to schoolbook
    # For each rank, we'll try to construct a decomposition
    
    # Strategy: Use the fact that for small m, we can use block structure
    # m=3, so we have 3 "intermediate" indices
    
    # Let's try a different approach: enumerate possible rank-r decompositions
    # using a constraint-based method
    
    # For rank r, we need to find U (6 x r), V (21 x r), W (14 x r)
    # such that for all (i,j,k): sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    
    # This is a system of 42 equations in 6r + 21r + 14r = 41r unknowns
    
    # Let's try to solve this using a simple approach:
    # Fix U and V, then solve for W (linear system)
    
    # Actually, let's try a more direct approach:
    # For each rank r, try random initializations and use iterative refinement
    
    import random
    random.seed(42)
    
    # Try to find a decomposition with rank < 42
    # We'll use a simple approach: try to find a rank-r decomposition
    # by solving the tensor equation
    
    # For a given rank r, the tensor equation is:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a 3-linear equation. We can try to solve it iteratively.
    
    # Let's try ranks from 30 to 41 (below schoolbook)
    for target_rank in range(25, rank_schoolbook):
        # Try to find a decomposition with this rank
        # Use random initialization and iterative refinement
        
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Iteratively refine
        converged = False
        for iteration in range(100):
            # Update W: for each c, W[c, r] = sum over (i,j) where k*n+i=c
            # Actually, let's do a simple gradient step
            
            # Compute current tensor
            current = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        current[i, j, k] = sum(U[a, r] * V[b, r] * W[c, r] for r in range(target_rank))
            
            # Check convergence
            error = np.max(np.abs(current - 1.0))
            if error < 1e-8:
                converged = True
                break
            
            # Gradient descent step
            lr = 0.01
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        diff = current[i, j, k] - 1.0
                        for r in range(target_rank):
                            U[a, r] -= lr * diff * V[b, r] * W[c, r]
                            V[b, r] -= lr * diff * U[a, r] * W[c, r]
                            W[c, r] -= lr * diff * U[a, r] * V[b, r]
        
        if converged:
            # Check if this is a valid decomposition
            U_list = [[float(U[i, j]) for j in range(target_rank)] for i in range(n * m)]
            V_list = [[float(V[i, j]) for j in range(target_rank)] for i in range(m * p)]
            W_list = [[float(W[i, j]) for j in range(target_rank)] for i in range(p * n)]
            
            if verify_decomposition(U_list, V_list, W_list, target_rank):
                best_rank = target_rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_U is not None:
        # Convert to integer/half-integer if possible
        # For now, return as floats (they should be close to rational)
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Return schoolbook decomposition
        U = np.zeros((n * m, rank_schoolbook), dtype=float)
        V = np.zeros((m * p, rank_schoolbook), dtype=float)
        W = np.zeros((p * n, rank_schoolbook), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        
        U_list = [list(map(int, row)) for row in U]
        V_list = [list(map(int, row)) for row in V]
        W_list = [list(map(int, row)) for row in W]
        
        return {"U": U_list, "V": V_list, "W": W_list}
