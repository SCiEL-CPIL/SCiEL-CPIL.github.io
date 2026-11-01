"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Tensor T[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
    # u = i*m + j, v = j*p + k, w = k*n + i
    # We need T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k and 0 otherwise.
    
    # Build the full tensor T
    # T shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for c in range(rank):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    
    # Check the non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > tol:
                    return False
    
    # Check that entries that should be zero are actually zero
    # The tensor has 1's at positions (i*m+j, j*p+k, k*n+i)
    # All other positions should be 0
    # Let's check a few random positions to be safe, or check all
    # Actually, let's construct the expected tensor and compare
    T_expected = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_expected[u, v, w] = 1.0
    
    if not np.allclose(T, T_expected, atol=tol):
        return False
    
    return True

def _try_rank(r, n, m, p, deadline, max_iter=2000):
    """Try to find a decomposition of rank r using random initialization + gradient descent."""
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    # Initialize with small random values
    U *= 0.1
    V *= 0.1
    W *= 0.1
    
    # Learning rate schedule
    lr = 0.01
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Compute gradients
        # T = sum_c outer(U[:,c], V[:,c], W[:,c])
        # Loss = 0.5 * ||T - T_target||^2
        # dT/dc = outer(U[:,c], V[:,c], W[:,c])
        # dL/dU[:,c] = sum_{v,w} (T[u,v,w] - T_target[u,v,w]) * V[v,c] * W[w,c]
        # Similar for V and W
        
        # Compute T
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for c in range(r):
            T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        # T_target
        T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T_target[u, v, w] = 1.0
        
        # Gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(r):
            dU[:, c] = np.einsum('ijk,jk->i', T - T_target, V[:, c] * W[:, c].reshape(1, 1, -1) if False else np.einsum('j,k->jk', V[:, c], W[:, c]))
            dV[:, c] = np.einsum('ijk,ik->j', T - T_target, np.einsum('i,k->ik', U[:, c], W[:, c]))
            dW[:, c] = np.einsum('ijk,ij->k', T - T_target, np.einsum('i,j->ij', U[:, c], V[:, c]))
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        if it % 100 == 0:
            lr *= 0.9
        
        # Check convergence
        if np.max(np.abs(T - T_target)) < 1e-6:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Try ranks from lower bound up to n*m*p
    # Lower bound for matrix multiplication rank is max(n*m, m*p, p*n)
    lower_bound = max(n*m, m*p, p*n)
    
    # First, return the schoolbook decomposition as a fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_result = (U_school.tolist(), V_school.tolist(), W_school.tolist(), rank_school)
    
    # Try to find lower rank decompositions
    for r in range(lower_bound, rank_school):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_result = (U.tolist(), V.tolist(), W.tolist(), r)
                break  # Found the lowest rank possible with this approach
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
