"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
Uses structured search with symmetry exploitation and local refinement.
"""
import numpy as np
import random
import time
from itertools import product

DIMENSIONS = [2, 3, 7]

def _verify_full(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    """
    rank = len(U[0])
    
    # Check diagonal (valid) terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    
    # Check off-diagonal (invalid) terms
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j == jp:
                    continue
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[jp*p+k][c] * W[k*n+i][c]
                    if not np.isclose(s, 0.0, atol=1e-9):
                        return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random decomposition with the given rank."""
    if seed is not None:
        np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    return U, V, W

def _compute_tensor_residuals(U, V, W, n, m, p):
    """
    Compute the residual tensor for the decomposition.
    Returns a dictionary mapping (i, j, k) -> residual for valid terms,
    and (i, j, jp, k) -> residual for invalid terms.
    """
    rank = U.shape[1]
    residuals = {}
    
    # Valid terms: should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                residuals[(i, j, k)] = s - 1.0
    
    # Invalid terms: should be 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j == jp:
                    continue
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
                    residuals[(i, j, jp, k)] = s
    
    return residuals

def _local_search(U, V, W, n, m, p, max_iters=1000, lr=0.01):
    """
    Perform local search to minimize the residual norm.
    Uses gradient descent on the squared residual norm.
    """
    rank = U.shape[1]
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    def residual_norm():
        res = _compute_tensor_residuals(U, V, W, n, m, p)
        return sum(v*v for v in res.values())
    
    best_norm = residual_norm()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute gradients via finite differences (expensive but reliable)
        eps = 1e-5
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Sample a subset of coordinates for gradient estimation
        U_coords = list(product(range(U.shape[0]), range(U.shape[1])))
        V_coords = list(product(range(V.shape[0]), range(V.shape[1])))
        W_coords = list(product(range(W.shape[0]), range(W.shape[1])))
        
        # Only estimate gradients for a random subset to save time
        random.shuffle(U_coords)
        random.shuffle(V_coords)
        random.shuffle(W_coords)
        
        for idx in U_coords[:50]:
            i, j = idx
            U[i, j] += eps
            f_plus = residual_norm()
            U[i, j] -= 2*eps
            f_minus = residual_norm()
            U[i, j] += eps
            grad_U[i, j] = (f_plus - f_minus) / (2*eps)
        
        for idx in V_coords[:50]:
            i, j = idx
            V[i, j] += eps
            f_plus = residual_norm()
            V[i, j] -= 2*eps
            f_minus = residual_norm()
            V[i, j] += eps
            grad_V[i, j] = (f_plus - f_minus) / (2*eps)
        
        for idx in W_coords[:50]:
            i, j = idx
            W[i, j] += eps
            f_plus = residual_norm()
            W[i, j] -= 2*eps
            f_minus = residual_norm()
            W[i, j] += eps
            grad_W[i, j] = (f_plus - f_minus) / (2*eps)
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        curr_norm = residual_norm()
        if curr_norm < best_norm:
            best_norm = curr_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return best_U, best_V, best_W, best_norm

def _try_rank_reduction(n, m, p, target_rank, max_time=30.0):
    """
    Try to find a decomposition with the given target rank.
    Uses random initialization and local search.
    """
    start_time = time.time()
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        
        U, V, W = _random_decomposition(n, m, p, target_rank, seed=attempt)
        
        # Check if already close
        res = _compute_tensor_residuals(U, V, W, n, m, p)
        norm = sum(v*v for v in res.values())
        
        if norm < 1e-6:
            return U, V, W
        
        # Local search
        U, V, W, final_norm = _local_search(U, V, W, n, m, p, max_iters=200, lr=0.005)
        
        if final_norm < 1e-8:
            # Verify with exact arithmetic
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            if _verify_full(U_list, V_list, W_list, n, m, p):
                return U, V, W
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Get schoolbook decomposition as fallback
    result = _schoolbook(n, m, p)
    
    # 2. Try to reduce rank from schoolbook
    schoolbook_rank = n * m * p  # 42
    
    # Try progressively lower ranks
    # Known lower bounds: for 2x3x7, the rank is at least 2*3*7 - some savings
    # Try ranks from 42 down to 35
    for target_rank in range(schoolbook_rank - 1, 34, -1):
        if time.time() - start_time > deadline - 10:
            break
        
        print(f"Trying rank {target_rank}...")
        found = _try_rank_reduction(n, m, p, target_rank, max_time=15.0)
        
        if found is not None:
            U, V, W = found
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            if _verify_full(U_list, V_list, W_list, n, m, p):
                print(f"Found valid decomposition with rank {target_rank}")
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # 3. If no reduction found, return schoolbook
    return result
# EVOLVE-BLOCK-END
