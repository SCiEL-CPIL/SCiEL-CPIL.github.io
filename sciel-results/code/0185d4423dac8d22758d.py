"""Find a rank-minimal exact tensor decomposition for multiplying a 3x5 by 5x7 matrix."""
import time
import random
import numpy as np
from numpy.linalg import svd, norm

DIMENSIONS = [3, 5, 7]

# Precompute the matrix multiplication tensor as a 3D array
# M[i, j, k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k]
# In our convention, the tensor T is indexed by (i, j, k) where:
# T[i, j, k] is the coefficient of A[i,j] * B[j,k] in C[i,k].
# For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
# Wait, the decomposition is:
# C[i,k] = sum_{j} A[i,j] B[j,k]
# So the tensor is T[i,j,k] = 1.
#
# The factorization is:
# U has shape (n*m, rank). Index u = i*m + j.
# V has shape (m*p, rank). Index v = j*p + k.
# W has shape (p*n, rank). Index w = k*n + i.
#
# The condition is:
# For all i, k:
# C[i,k] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] ... wait, no.
#
# The standard rank-r decomposition of the tensor T_{i,j,k} is:
# T_{i,j,k} = sum_{c=1}^r U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
#
# Wait, this is not quite right. The tensor T is a 3-index tensor.
# The decomposition is T_{i,j,k} = sum_c U_{i,m+j,c} V_{j,p+k,c} W_{k,n+i,c}.
# This is the tensor rank decomposition.
#
# Let's verify the schoolbook:
# T_{i,j,k} = 1.
# U_{i*m+j, c_{i,j,k}} = 1, V_{j*p+k, c_{i,j,k}} = 1, W_{k*n+i, c_{i,j,k}} = 1.
# Then sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = sum_{i',j',k'} U_{i*m+j, c_{i',j',k'}} V_{j*p+k, c_{i',j',k'}} W_{k*n+i, c_{i',j',k'}}
# = sum_{i',j',k'} [i*m+j == i'*m+j'] * [j*p+k == j'*p+k'] * [k*n+i == k'*n+i']
# = 1.
# This works.

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Tensor T is all ones of shape (n, m, p)
    # T[i, j, k] = 1
    
    # We want to find U (nm x r), V (mp x r), W (pn x r) such that
    # for all i, j, k: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    
    # This is a system of n*m*p equations with 3*r variables.
    # Actually, it's a set of constraints.
    
    # Let's use a heuristic search.
    # Start with a random rank-r decomposition and try to reduce r.
    
    # We know that the rank must be at least max(nm, mp, pn) = 15.
    # But it's likely higher.
    
    # Let's try to find a decomposition with rank r = 15, 16, ...
    
    # For a given r, we have a system of equations.
    # We can use a randomized approach:
    # 1. Initialize U, V, W with random values.
    # 2. Define the error E = T - (U, V, W) decomposition.
    # 3. Try to reduce the error using gradient descent or a similar method.
    
    # However, the objective is non-convex.
    
    # Let's try a different approach.
    # We can use the fact that the tensor is all ones.
    # This is a special case.
    
    # Let's try to find a decomposition with small rank.
    # We can use a grid search over small ranks.
    
    # For each rank r, we try to find U, V, W.
    # We can use a randomized initialization and then refine.
    
    # Let's try a simple heuristic:
    # 1. Start with r = max(nm, mp, pn) = 15.
    # 2. Try to find a solution.
    # 3. If not found, increase r.
    
    # To find a solution for a given r, we can use a local search method.
    
    # Let's implement a simple local search.
    
    def try_rank(r, time_budget=5.0):
        """Try to find a decomposition of rank r."""
        deadline = time.time() + time_budget
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Randomized restarts
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random values
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Normalize to avoid overflow
            U /= norm(U)
            V /= norm(V)
            W /= norm(W)
            
            # Local search using gradient descent
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Compute the error
                # E[i, j, k] = 1 - sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                E = np.ones((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            E[i, j, k] = 1.0 - val
                
                error = np.sum(E**2)
                if error < best_error:
                    best_error = error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if error < 1e-12:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                dU[i*m+j, c] += -2 * E[i, j, k] * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += -2 * E[i, j, k] * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += -2 * E[i, j, k] * U[i*m+j, c] * V[j*p+k, c]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Normalize
                U /= norm(U)
                V /= norm(V)
                W /= norm(W)
        
        if best_error < 1e-10:
            return best_U, best_V, best_W, best_error
        return None, None, None, best_error
    
    # Try to find a decomposition with small rank
    for r in range(15, 30):
        if time.time() > start_time + 100:
            break
        
        U, V, W, error = try_rank(r, time_budget=10.0)
        if U is not None:
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
