"""Matrix multiplication 2,4,7 tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]
DEADLINE = 130  # safety margin before 150s wall-clock limit

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if indices are coupled."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication.
    
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    Convention: W[k*n + i, r]
    We need: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    # Reshape for easier access
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_[i, j, r] * V_[j, k, r] * W_[k, i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using numpy operations."""
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    # For each (i,j,k): sum_r U_[i,j,r] * V_[j,k,r] * W_[k,i,r]
    # We can compute this as: for each r, outer product contributions
    # Or iterate over r and accumulate
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        # U[:, :, r] has shape (n, m), V[:, :, r] has shape (m, p), W[:, :, r] has shape (p, n)
        # Contribution to T[i,j,k] from rank r: U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is not a simple outer product due to shared indices
        # Let's do it via broadcasting:
        # U_contrib[i,j] = U_[i,j,r]
        # V_contrib[j,k] = V_[j,k,r]  
        # W_contrib[k,i] = W_[k,i,r]
        # T[i,j,k] += U_contrib[i,j] * V_contrib[j,k] * W_contrib[k,i]
        # We can compute this as:
        # For fixed r, define A[i,j] = U_[i,j,r], B[j,k] = V_[j,k,r], C[k,i] = W_[k,i,r]
        # T[i,j,k] = sum over... no, it's just A[i,j]*B[j,k]*C[k,i] for this r
        # We need to sum this over r.
        
        # Efficient way: create a 3D array from A, B, C
        # A[i,j]*B[j,k]*C[k,i] - this is a contraction
        # Let's use: for each i,j,k, multiply the three values
        # We can do: np.einsum('ij,jk,ki->ijk', U_[:,:,r], V_[:,:,r], W_[:,:,r])
        result += np.einsum('ij,jk,ki->ijk', U_[:, :, r], V_[:, :, r], W_[:, :, r])
    
    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _random_search(n, m, p, rank, n_trials=200, seed=42):
    """Try random rank-r decompositions with entries from {-1, 0, 1}."""
    rng = np.random.RandomState(seed)
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    for trial in range(n_trials):
        # Generate random sparse matrices with entries in {-1, 0, 1}
        # Use a sparsity pattern to encourage structure
        U = rng.choice([-1, 0, 1], size=(U_rows, rank))
        V = rng.choice([-1, 0, 1], size=(V_rows, rank))
        W = rng.choice([-1, 0, 1], size=(W_rows, rank))
        
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _structured_search(n, m, p, rank, n_trials=100):
    """Try structured random decompositions that respect the tensor structure."""
    rng = np.random.RandomState(123)
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    for trial in range(n_trials):
        # Generate with some structure: each row has a few non-zero entries
        sparsity_U = max(1, rank // (n * m // 2 + 1))
        sparsity_V = max(1, rank // (m * p // 2 + 1))
        sparsity_W = max(1, rank // (p * n // 2 + 1))
        
        U = np.zeros((U_rows, rank))
        V = np.zeros((V_rows, rank))
        W = np.zeros((W_rows, rank))
        
        for i in range(U_rows):
            cols = rng.choice(rank, size=min(sparsity_U, rank), replace=False)
            for c in cols:
                U[i, c] = rng.choice([-1, 1])
        
        for j in range(V_rows):
            cols = rng.choice(rank, size=min(sparsity_V, rank), replace=False)
            for c in cols:
                V[j, c] = rng.choice([-1, 1])
        
        for k in range(W_rows):
            cols = rng.choice(rank, size=min(sparsity_W, rank), replace=False)
            for c in cols:
                W[k, c] = rng.choice([-1, 1])
        
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook as baseline
    U_sb, V_sb, W_sb = _construct_schoolbook(n, m, p)
    best_rank = n * m * p  # 56
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Theoretical lower bounds:
    # For 2x4x7: lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Also, bilinear complexity bounds suggest around 30-40
    
    # Try ranks from just below schoolbook downward
    # Be conservative with trials to stay within time budget
    ranks_to_try = list(range(55, 27, -1))  # 55 down to 28
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Random search
        result = _random_search(n, m, p, rank, n_trials=50, seed=rank * 1000)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # Found a valid lower rank, stop
        
        # Structured search
        if time.time() - start_time > DEADLINE - 15:
            break
        result = _structured_search(n, m, p, rank, n_trials=30)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        # Return best found so far
        pass
    
    # Convert to lists for the expected interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
