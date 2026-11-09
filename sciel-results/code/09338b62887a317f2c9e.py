"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U @ V @ W^T (with proper indexing) equals the multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    rank = U.shape[1]
    # Build tensor efficiently using einsum
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Let's reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = np.array(U).reshape(n, m, rank)
    V_r = np.array(V).reshape(m, p, rank)
    W_r = np.array(W).reshape(p, n, rank)
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    # Expected tensor: T[i,j,k] = 1 if we're computing (A*B)[i,k] where A is n x m, B is m x p
    # Actually the multiplication tensor for A (n x m) * B (m x p) = C (n x p) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k} ... no.
    # The tensor is T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    # The multiplication tensor has entries T[i,j,k] = 1 if (i,j) is a valid entry of A,
    # (j,k) is a valid entry of B, and the product contributes to (i,k) of C.
    # Since every A[i,j]*B[j,k] contributes to C[i,k], T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=tol):
        return False
    return True

def _make_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_random_rank(n, m, p, rank, time_limit, seed=0):
    """Try to find a rank-rank decomposition using random initialization and gradient descent."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Reshape for efficient computation
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Target tensor
    target = np.ones((n, m, p))
    
    # Learning rate
    lr = 0.01
    lr_decay = 0.999
    
    for step in range(100000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current tensor
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        diff = T - target
        
        # Compute gradients
        # dL/dU_r[i,j,c] = sum_{k} diff[i,j,k] * V_r[j,k,c] * W_r[k,i,c]
        # dL/dV_r[j,k,c] = sum_{i} diff[i,j,k] * U_r[i,j,c] * W_r[k,i,c]
        # dL/dW_r[k,i,c] = sum_{j} diff[i,j,k] * U_r[i,j,c] * V_r[j,k,c]
        
        grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        
        # Update
        U_r -= lr * grad_U
        V_r -= lr * grad_V
        W_r -= lr * grad_W
        
        lr *= lr_decay
        
        # Check convergence
        if np.max(np.abs(diff)) < 1e-9:
            break
    
    # Final check
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    if np.max(np.abs(T - target)) < 1e-9:
        return U_r.reshape(n*m, rank), V_r.reshape(m*p, rank), W_r.reshape(p*n, rank)
    
    return None

def _try_local_search(n, m, p, rank, time_limit, seed=0):
    """Local search starting from a random initialization."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Start with a structured initialization
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Initialize each column with a random pattern
    for c in range(rank):
        # Randomly assign non-zero entries
        for i in range(n):
            for j in range(m):
                if rng.random() < 0.3:
                    U[i*m+j, c] = rng.choice([-1, 1])
        for j in range(m):
            for k in range(p):
                if rng.random() < 0.3:
                    V[j*p+k, c] = rng.choice([-1, 1])
        for k in range(p):
            for i in range(n):
                if rng.random() < 0.3:
                    W[k*n+i, c] = rng.choice([-1, 1])
    
    # Try to refine using local search
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    target = np.ones((n, m, p))
    
    # Compute initial error
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    error = np.max(np.abs(T - target))
    
    # Local search: try to flip entries to reduce error
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        if error < 1e-9:
            break
        
        # Try flipping a random entry
        best_improvement = 0
        best_change = None
        
        for _ in range(100):
            # Choose which matrix to modify
            mat_idx = rng.randint(0, 3)
            if mat_idx == 0:
                i, j, c = rng.randint(0, n-1), rng.randint(0, m-1), rng.randint(0, rank-1)
                old_val = U_r[i, j, c]
                U_r[i, j, c] = -old_val
                T_new = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                new_error = np.max(np.abs(T_new - target))
                U_r[i, j, c] = old_val
            elif mat_idx == 1:
                j, k, c = rng.randint(0, m-1), rng.randint(0, p-1), rng.randint(0, rank-1)
                old_val = V_r[j, k, c]
                V_r[j, k, c] = -old_val
                T_new = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                new_error = np.max(np.abs(T_new - target))
                V_r[j, k, c] = old_val
            else:
                k, i, c = rng.randint(0, p-1), rng.randint(0, n-1), rng.randint(0, rank-1)
                old_val = W_r[k, i, c]
                W_r[k, i, c] = -old_val
                T_new = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                new_error = np.max(np.abs(T_new - target))
                W_r[k, i, c] = old_val
            
            improvement = error - new_error
            if improvement > best_improvement:
                best_improvement = improvement
                best_change = (mat_idx, (i, j, k, c) if mat_idx == 0 else (j, k, c) if mat_idx == 1 else (k, i, c))
        
        if best_change is not None and best_improvement > 1e-12:
            mat_idx, idx = best_change
            if mat_idx == 0:
                i, j, c = idx
                U_r[i, j, c] = -U_r[i, j, c]
            elif mat_idx == 1:
                j, k, c = idx
                V_r[j, k, c] = -V_r[j, k, c]
            else:
                k, i, c = idx
                W_r[k, i, c] = -W_r[k, i, c]
            
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            error = np.max(np.abs(T - target))
    
    if error < 1e-9:
        return U_r.reshape(n*m, rank), V_r.reshape(m*p, rank), W_r.reshape(p*n, rank)
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)
    
    # Try to find lower rank decompositions
    # Known lower bounds: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Upper bound: min(n*m*p, ...) 
    # Let's try ranks from a reasonable lower bound upward
    
    # Try some candidate ranks
    candidate_ranks = []
    for r in range(35, best_rank + 1):
        if time.time() > deadline - 10:
            break
        candidate_ranks.append(r)
    
    # Try random gradient descent for each candidate rank
    for rank in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try a few random seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            result = _try_random_rank(n, m, p, rank, time_limit=10, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_rank = rank
                    best_result = (U, V, W)
                    break
        
        # If we found a valid decomposition at this rank, try to go lower
        if best_rank == rank:
            continue
    
    # Try local search for the best rank found
    if time.time() < deadline - 10:
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = _try_local_search(n, m, p, best_rank, time_limit=5, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_result = (U, V, W)
                    break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
