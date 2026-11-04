import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication target tensor."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    T = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    return T

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook (trivial) decomposition."""
    rank = n * m * p
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    U = np.zeros((size_u, rank))
    V = np.zeros((size_v, rank))
    W = np.zeros((size_w, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that the decomposition is correct."""
    X = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
    return np.allclose(X, T, atol=tol)

def _try_find_decomposition(n, m, p, target_rank, time_budget, seed=0):
    """
    Try to find a decomposition of the given rank using alternating least squares.
    Returns (U, V, W) if found, None otherwise.
    """
    size_u = n * m
    size_v = m * p
    size_w = p * n
    T = _build_target_tensor(n, m, p)
    
    rng = np.random.default_rng(seed)
    
    # Try multiple random restarts
    for restart in range(20):
        # Initialize with small random values
        U = rng.normal(0, 0.1, (size_u, target_rank))
        V = rng.normal(0, 0.1, (size_v, target_rank))
        W = rng.normal(0, 0.1, (size_w, target_rank))
        
        # Alternating least squares iterations
        for step in range(100):
            # Solve for U with V and W fixed
            # For each r: U[:, r] minimizes ||T - sum_r U_r ⊗ V_r ⊗ W_r||
            # This is a linear system in U[:, r]
            
            # Efficient approach: use the fact that for fixed r,
            # the contribution of U[:, r] is U[:, r] ⊗ (V[:, r] ⊗ W[:, r])
            
            # We can solve this by building a linear system
            # For each r, we want:
            # sum_r U[:, r] * (V[:, r] ⊗ W[:, r]) = T
            
            # Let's do gradient descent with adaptive learning rate
            
            X = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
            diff = T - X
            loss = np.sum(diff**2)
            
            if loss < 1e-10:
                break
            
            # Compute gradients
            dU = np.einsum('ijk,jr,kr->ir', diff, V, W, optimize=True)
            dV = np.einsum('ijk,ir,kr->jr', diff, U, W, optimize=True)
            dW = np.einsum('ijk,ir,jr->kr', diff, U, V, optimize=True)
            
            # Adaptive learning rate based on loss
            lr = 0.01 / (1 + step * 0.01)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if loss < 1e-12:
                break
        
        # Try to round to half-integers
        for mult in [1, 2, 4]:
            U_rounded = np.rint(U * mult) / mult
            V_rounded = np.rint(V * mult) / mult
            W_rounded = np.rint(W * mult) / mult
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, T):
                return U_rounded, V_rounded, W_rounded
        
        # Also try integer rounding
        U_int = np.rint(U)
        V_int = np.rint(V)
        W_int = np.rint(W)
        if _verify_decomposition(U_int, V_int, W_int, T):
            return U_int, V_int, W_int
    
    return None

def construct():
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    T = _build_target_tensor(n, m, p)
    
    # Try to find lower rank decompositions
    # Theoretical lower bounds and known results suggest we might find
    # something better than schoolbook (90)
    
    # Try ranks in increasing order of difficulty
    candidate_ranks = [50, 55, 60, 65, 70, 75, 80, 85]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 30.0:
            break
        
        result = _try_find_decomposition(n, m, p, rank, deadline - time.time(), seed=rank)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, T):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
