"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification using einsum."""
    rank = len(U[0])
    U = np.array(U, dtype=float).reshape(n, m, rank)
    V = np.array(V, dtype=float).reshape(m, p, rank)
    W = np.array(W, dtype=float).reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    return np.allclose(result, np.ones((n, m, p)), atol=tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_alternative_structure(n, m, p, rank, max_iter=2000, seed=42):
    """
    Try to find a rank-r decomposition using an alternative parameterization.
    
    Key insight: For matrix multiplication, we can exploit the structure by
    fixing one of the factor matrices to have a specific form.
    
    Strategy: Fix W to be the identity-like structure (p*n x rank) where each
    row has a single 1, and optimize U and V.
    
    Actually, let's try a different approach: use the fact that the tensor
    T[i,j,k] = 1 can be decomposed as a sum of rank-1 tensors. We'll use
    an iterative optimization approach with multiple restarts.
    """
    rng = np.random.RandomState(seed)
    
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(3):
        # Initialize with random matrices
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        # Try a structured initialization: make some entries zero
        # to encourage sparse solutions
        for c in range(rank):
            # Randomly zero out some entries
            mask_u = rng.rand(n * m) > 0.5
            mask_v = rng.rand(m * p) > 0.5
            mask_w = rng.rand(p * n) > 0.5
            U[mask_u, c] = 0
            V[mask_v, c] = 0
            W[mask_w, c] = 0
        
        current_err = np.inf
        for it in range(max_iter):
            # Compute error
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            err = np.mean((result - 1.0) ** 2)
            
            if err < current_err:
                current_err = err
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-12:
                    break
            
            # Gradient-like update: perturb in direction that reduces error
            # Use a simple coordinate descent approach
            # Pick a random coordinate and adjust
            for _ in range(3):
                mat = rng.choice([0, 1, 2])
                row = rng.randint(0, [n*m, m*p, p*n][mat])
                col = rng.randint(0, rank)
                
                if mat == 0:
                    old_val = U[row, col]
                    U[row, col] = old_val + 0.01 * rng.randn()
                    U3 = U.reshape(n, m, rank)
                    V3 = V.reshape(m, p, rank)
                    W3 = W.reshape(p, n, rank)
                    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                    new_err = np.mean((result - 1.0) ** 2)
                    if new_err > err:
                        U[row, col] = old_val
                elif mat == 1:
                    old_val = V[row, col]
                    V[row, col] = old_val + 0.01 * rng.randn()
                    U3 = U.reshape(n, m, rank)
                    V3 = V.reshape(m, p, rank)
                    W3 = W.reshape(p, n, rank)
                    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                    new_err = np.mean((result - 1.0) ** 2)
                    if new_err > err:
                        V[row, col] = old_val
                else:
                    old_val = W[row, col]
                    W[row, col] = old_val + 0.01 * rng.randn()
                    U3 = U.reshape(n, m, rank)
                    V3 = V.reshape(m, p, rank)
                    W3 = W.reshape(p, n, rank)
                    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                    new_err = np.mean((result - 1.0) ** 2)
                    if new_err > err:
                        W[row, col] = old_val
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _try_block_structure(n, m, p, rank, seed=42):
    """
    Try a block-structured decomposition.
    
    Idea: Group the (i,j,k) indices into blocks and try to find a decomposition
    where each block is handled by a subset of the rank dimensions.
    """
    rng = np.random.RandomState(seed)
    
    # For 3x4x6, we have 3*4*6 = 72 entries
    # Try to find a decomposition with rank < 72
    
    # One approach: use the fact that T[i,j,k] = 1 can be written as
    # sum_c f_c(i) * g_c(j) * h_c(k)
    # where f_c, g_c, h_c are functions of i, j, k respectively.
    
    # This is equivalent to finding a low-rank decomposition of the tensor.
    
    # Let's try a specific construction:
    # For each (i, k) pair, we need sum_j,c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all j.
    
    # If we fix W[k*n+i, c] = 1 for all k, i, c (i.e., W is all ones),
    # then we need sum_j,c U[i*m+j, c] * V[j*p+k, c] = 1 for all i, j, k.
    
    # This is a linear system in U and V. Let's try this.
    
    # Actually, let's try a different approach: use known results from
    # matrix multiplication theory. For 3x4x6, the rank is not known exactly,
    # but we can try to find a decomposition with rank around 30-40.
    
    # Let's use a random search with a better initialization strategy.
    
    best_err = np.inf
    best_result = None
    
    for attempt in range(5):
        # Initialize with a structured pattern
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Try to make each column of U, V, W correspond to a "pattern"
        for c in range(rank):
            # Randomly assign values
            U[:, c] = rng.choice([-1, 0, 1], size=n*m)
            V[:, c] = rng.choice([-1, 0, 1], size=m*p)
            W[:, c] = rng.choice([-1, 0, 1], size=p*n)
        
        # Optimize using simple iterations
        for it in range(500):
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            err = np.mean((result - 1.0) ** 2)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-12:
                break
            
            # Perturb randomly
            mat = rng.choice([0, 1, 2])
            if mat == 0:
                idx = rng.randint(0, n*m*rank)
                r, c = divmod(idx, rank)
                U[r, c] += 0.1 * rng.randn()
            elif mat == 1:
                idx = rng.randint(0, m*p*rank)
                r, c = divmod(idx, rank)
                V[r, c] += 0.1 * rng.randn()
            else:
                idx = rng.randint(0, p*n*rank)
                r, c = divmod(idx, rank)
                W[r, c] += 0.1 * rng.randn()
    
    if best_err < 1e-8:
        U, V, W = best_result
        return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, leave 5s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    
    # Try various ranks from low to high
    ranks_to_try = list(range(24, 72))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # Try multiple methods for this rank
        methods = [
            ("alternative", lambda: _try_alternative_structure(n, m, p, rank, max_iter=1000, seed=rank)),
            ("block", lambda: _try_block_structure(n, m, p, rank, seed=rank)),
        ]
        
        for method_name, method_func in methods:
            if time.time() > deadline - 15:
                break
            
            try:
                result = method_func()
                if result is not None:
                    U_new, V_new, W_new = result
                    
                    # Verify
                    if _verify_fast(U_new, V_new, W_new, n, m, p, tol=1e-9):
                        best_result = {"U": U_new, "V": V_new, "W": W_new}
                        best_rank = rank
                        return best_result
            except Exception:
                pass
    
    # If no better decomposition found, return schoolbook
    return best_result

# EVOLVE-BLOCK-END
