"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the trivial schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is exact within tolerance.
    The matrix multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i,j,k (since it represents the identity in the bilinear map).
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _compute_error(U, V, W, n, m, p, target_rank):
    """Compute the sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(target_rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err += (val - 1.0) ** 2
    return err


def _try_rank(n, m, p, target_rank, max_iterations=5000, deadline=None):
    """
    Attempt to find a decomposition of exactly target_rank using iterative refinement.
    Uses a randomized initialization and gradient-free local search with adaptive step sizes.
    """
    if deadline is not None:
        if time.time() > deadline:
            return None
    
    np.random.seed(42 + target_rank)
    
    # Initialize with random values scaled appropriately
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # Adaptive step size
    step_size = 0.1
    best_err = _compute_error(U, V, W, n, m, p, target_rank)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Multiple restarts within the iteration budget
    num_restarts = 3
    iters_per_restart = max_iterations // num_restarts
    
    for restart in range(num_restarts):
        if restart > 0:
            # Re-initialize with different seed
            np.random.seed(42 + target_rank + restart * 100)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            step_size = 0.1
            best_err = _compute_error(U, V, W, n, m, p, target_rank)
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(iters_per_restart):
            if deadline is not None and time.time() > deadline:
                return None
            
            err = _compute_error(U, V, W, n, m, p, target_rank)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # Decrease step size on improvement
                step_size *= 0.95
            else:
                # Revert and increase step size
                U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
                step_size *= 1.1
            
            if best_err < 1e-12:
                return best_U, best_V, best_W
            
            # Perturb one factor randomly
            factor = random.choice(['U', 'V', 'W'])
            if factor == 'U':
                row = random.randint(0, n * m - 1)
                col = random.randint(0, target_rank - 1)
                U[row, col] += random.gauss(0, step_size)
            elif factor == 'V':
                row = random.randint(0, m * p - 1)
                col = random.randint(0, target_rank - 1)
                V[row, col] += random.gauss(0, step_size)
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, target_rank - 1)
                W[row, col] += random.gauss(0, step_size)
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None


def _try_rank_with_structure(n, m, p, target_rank, deadline=None):
    """
    Try to find a decomposition using structural insights.
    For 3x3 * 3x7, we can exploit the fact that the output has 7 columns.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    # Strategy: Use the fact that C[i,k] = sum_j A[i,j] * B[j,k]
    # We can group terms by output column k.
    # For each k, we need to compute a 3-vector (column k of C).
    # This is like 7 separate 3x3 * 3x1 multiplications.
    
    # Schoolbook for 3x3 * 3x1 has rank 9.
    # Total schoolbook: 7 * 9 = 63.
    
    # Can we do better? The 7 multiplications share the input A.
    # This suggests potential for rank reduction.
    
    # Let's try a structured approach:
    # Represent the decomposition as:
    # U has rows indexed by (i,j), V by (j,k), W by (k,i)
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to: for each (i,j,k), the inner product of
    # U_row[i*m+j], V_row[j*p+k], W_row[k*n+i] over r equals 1.
    
    # Let's try to construct a solution with rank < 63 using a more informed initialization.
    
    np.random.seed(123 + target_rank)
    
    # Initialize with a pattern that respects the structure
    U = np.zeros((n * m, target_rank), dtype=float)
    V = np.zeros((m * p, target_rank), dtype=float)
    W = np.zeros((p * n, target_rank), dtype=float)
    
    # Fill with small random values
    U[:] = np.random.randn(n * m, target_rank) * 0.3
    V[:] = np.random.randn(m * p, target_rank) * 0.3
    W[:] = np.random.randn(p * n, target_rank) * 0.3
    
    # Local search with structure-aware perturbations
    step_size = 0.05
    best_err = _compute_error(U, V, W, n, m, p, target_rank)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    max_iters = 3000
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
        
        err = _compute_error(U, V, W, n, m, p, target_rank)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            step_size = max(0.001, step_size * 0.97)
        else:
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
            step_size = min(1.0, step_size * 1.05)
        
        if best_err < 1e-12:
            return best_U, best_V, best_W
        
        # Perturb with structure awareness
        choice = random.random()
        if choice < 0.33:
            row = random.randint(0, n * m - 1)
            col = random.randint(0, target_rank - 1)
            U[row, col] += random.gauss(0, step_size)
        elif choice < 0.66:
            row = random.randint(0, m * p - 1)
            col = random.randint(0, target_rank - 1)
            V[row, col] += random.gauss(0, step_size)
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, target_rank - 1)
            W[row, col] += random.gauss(0, step_size)
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But we can do better than this lower bound in practice.
    
    # Try ranks from a reasonable lower bound up to the schoolbook rank
    # We'll try a range of ranks and use multiple strategies
    
    for target_rank in range(21, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple strategies
        result = _try_rank(n, m, p, target_rank, max_iterations=2000, deadline=deadline)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
        
        # Try structure-aware approach
        result = _try_rank_with_structure(n, m, p, target_rank, deadline=deadline)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    return best_result

# EVOLVE-BLOCK-END
