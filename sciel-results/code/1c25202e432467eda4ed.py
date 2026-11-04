"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the factor matrices exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = 1 if the term contributes to C[i,k] from A[i,j]*B[j,k]
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # And for terms that don't contribute (i.e., where the product should be 0), it must also be 0.
    # Actually, the matrix multiplication tensor is:
    # T_{i,j,k} = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # The decomposition is: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # This must equal 1 for all valid (i,j,k) triples since each term A[i,j]*B[j,k] contributes exactly once.
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor value should be 1 (each product term contributes exactly 1)
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    # For each r, we have outer products
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can check this by building the tensor and comparing
    
    # Build expected tensor: all ones for valid (i,j,k)
    # Check via random sampling for speed, then full check if needed
    # Full check:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_random_rank(n, m, p, rank, max_iter=500, seed=None):
    """Try to find a rank-`rank` decomposition via random initialization and gradient descent."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    best_err = float('inf')
    lr = 0.01
    
    for it in range(max_iter):
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # dT/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                    # dT/dV[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
                    # dT/dW[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]
                    diff = 1.0 - np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += diff * diff
                    dU[i*m+j, :] += diff * V[j*p+k, :] * W[k*n+i, :]
                    dV[j*p+k, :] += diff * U[i*m+j, :] * W[k*n+i, :]
                    dW[k*n+i, :] += diff * U[i*m+j, :] * V[j*p+k, :]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
    
    if best_err < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None

def _structured_rank(n, m, p, rank):
    """Try structured decompositions based on known results."""
    # For small matrices, try specific known decompositions
    # This is a placeholder for structured approaches
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook (rank = 90)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 3x5 x 5x6, the rank is at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Upper bound is 90 (schoolbook)
    # Try ranks from lower bound up to best_rank - 1
    
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try random search for this rank
        result = _try_random_rank(n, m, p, rank, max_iter=200, seed=rank)
        if result is not None:
            # Verify
            if _verify_decomposition_fast(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = rank
                break  # Found minimal rank, stop
    
    # Convert to lists
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
# EVOLVE-BLOCK-END
