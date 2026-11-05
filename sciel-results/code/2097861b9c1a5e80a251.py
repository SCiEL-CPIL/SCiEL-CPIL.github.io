"""Program to find rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import numpy as np
import random
import itertools
from collections import defaultdict

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # The tensor T is of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if the indices correspond to a valid multiplication term
    # a = i*m + j, b = j*p + k, c = k*n + i
    # where i in [0, n), j in [0, m), k in [0, p)
    
    # Build the set of "1" entries in the tensor
    tensor_points = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_points.add((a, b, c))
    
    # Schoolbook decomposition has rank n*m*p = 60
    # Each term corresponds to exactly one point in tensor_points
    # We want to find a decomposition with rank < 60
    
    # Strategy: Local search / SAT-like approach
    # 1. Start with a random rank-r decomposition (r < 60)
    # 2. For each term t, we have U[:, t], V[:, t], W[:, t]
    # 3. The condition is: for all (a, b, c), sum_t U[a,t]*V[b,t]*W[c,t] = 1 if (a,b,c) in tensor_points else 0
    
    # This is a system of non-linear equations.
    # We can use a heuristic:
    # - Fix a support pattern (which U[a,t], V[b,t], W[c,t] are non-zero)
    # - Solve for the values using least squares or algebraic methods
    
    # A simpler heuristic:
    # - Start with the schoolbook decomposition (rank 60)
    # - Try to merge pairs of terms that are "similar"
    # - Or, try to find a rank-r decomposition directly by random search
    
    # Given the 120s limit, let's try a few random searches for rank r < 60
    
    def solve_for_rank(r, max_iters=100):
        """Try to find a rank-r decomposition using random initialization and gradient descent."""
        # Initialize U, V, W with small random values
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Target tensor (sparse)
        target = np.zeros((n*m, m*p, p*n))
        for (a, b, c) in tensor_points:
            target[a, b, c] = 1
        
        # Gradient descent
        lr = 0.01
        for it in range(max_iters):
            # Compute current tensor
            current = np.einsum('at,bt,ct->abc', U, V, W)
            
            # Compute error
            error = current - target
            loss = np.sum(error**2)
            
            if loss < 1e-10:
                return U, V, W
            
            # Compute gradients
            # d_loss/dU = 2 * error contracted with V and W
            grad_U = 2 * np.einsum('abc,bt,ct->at', error, V, W)
            grad_V = 2 * np.einsum('abc,at,ct->bt', error, U, W)
            grad_W = 2 * np.einsum('abc,at,bt->ct', error, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Normalize to prevent divergence
            for t in range(r):
                norm = np.sqrt(np.sum(U[:, t]**2) + np.sum(V[:, t]**2) + np.sum(W[:, t]**2))
                if norm > 1e10:
                    U[:, t] /= norm
                    V[:, t] /= norm
                    W[:, t] /= norm
        
        # Check if we found a solution
        current = np.einsum('at,bt,ct->abc', U, V, W)
        error = current - target
        if np.sum(error**2) < 1e-8:
            return U, V, W
        
        return None
    
    # Try ranks from 55 down to 50 (a reasonable range)
    for r in range(55, 49, -1):
        result = solve_for_rank(r, max_iters=50)
        if result is not None:
            U, V, W = result
            # Convert to lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no better decomposition found, return schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
