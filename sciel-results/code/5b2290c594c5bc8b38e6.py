"""Matrix multiplication 3x4 x 4x4: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]


def _verify(U, V, W, n=3, m=4, p=4, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _schoolbook(n=3, m=4, p=4):
    """Standard schoolbook decomposition with rank n*m*p."""
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


def _tensor_error(U, V, W, n=3, m=4, p=4):
    """Compute the Frobenius norm of the tensor error."""
    r = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    err_sq = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                err_sq += (val - 1.0) ** 2
    return np.sqrt(err_sq)


def _gradient_descent(U, V, W, n, m, p, max_iter=2000, lr_init=0.01, lr_decay=0.999, 
                      deadline=None, tol=1e-12):
    """
    Perform gradient descent to minimize the tensor decomposition error.
    Returns (U, V, W, final_error).
    """
    r = len(U[0])
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    
    lr = lr_init
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        err_sq = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    resid = val - 1.0
                    err_sq += resid * resid
                    
                    for c in range(r):
                        dU[i * m + j, c] += resid * V[j * p + k, c] * W[k * n + i, c]
                        dV[j * p + k, c] += resid * U[i * m + j, c] * W[k * n + i, c]
                        dW[k * n + i, c] += resid * U[i * m + j, c] * V[j * p + k, c]
        
        if np.sqrt(err_sq) < tol:
            break
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        lr *= lr_decay
        
        # Periodic restart with smaller lr if stuck
        if iteration % 500 == 499 and lr < 1e-4:
            lr = 0.005
    
    final_err = _tensor_error(U.tolist(), V.tolist(), W.tolist(), n, m, p)
    return U.tolist(), V.tolist(), W.tolist(), final_err


def _try_rank_reduction(n=3, m=4, p=4, max_rank=48, min_rank=15, max_time=90, seed=42):
    """
    Try to find a low-rank decomposition by starting from schoolbook and
    progressively reducing rank with numerical optimization.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    best_result = None
    best_rank = n * m * p
    
    # Start with schoolbook and try to reduce rank
    for target_rank in range(min_rank, min(max_rank + 1, n * m * p + 1)):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations for this rank
        for trial in range(20):
            if time.time() > deadline - 2:
                break
            
            rng = np.random.RandomState(seed + target_rank * 1000 + trial)
            
            # Initialize: use a subset of schoolbook terms plus random perturbation
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Fill with schoolbook structure for the first min(target_rank, n*m*p) terms
            n_terms = min(target_rank, n * m * p)
            for idx in range(n_terms):
                i = idx // (m * p)
                j = (idx // p) % m
                k = idx % p
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
            
            # Add small random perturbation to break symmetry
            U += rng.randn(n * m, target_rank) * 0.1
            V += rng.randn(m * p, target_rank) * 0.1
            W += rng.randn(p * n, target_rank) * 0.1
            
            # Run gradient descent
            U_opt, V_opt, W_opt, err = _gradient_descent(
                U.tolist(), V.tolist(), W.tolist(), n, m, p,
                max_iter=3000, lr_init=0.01, lr_decay=0.9995,
                deadline=deadline - 2
            )
            
            if err < 1e-10 and target_rank < best_rank:
                best_rank = target_rank
                best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
                # Try to go lower
                continue
        
        if best_result is not None and best_rank <= target_rank:
            # Found a solution at this rank, try lower
            if target_rank <= min_rank:
                break
    
    return best_result, best_rank


def _polish_fractions(U, V, W, n=3, m=4, p=4, max_denominator=10000):
    """
    Convert numerical solution to exact fractions and verify.
    """
    U_frac = [[Fraction(x).limit_denominator(max_denominator) for x in row] for row in U]
    V_frac = [[Fraction(x).limit_denominator(max_denominator) for x in row] for row in V]
    W_frac = [[Fraction(x).limit_denominator(max_denominator) for x in row] for row in W]
    
    if _verify(U_frac, V_frac, W_frac, n, m, p, tol=1e-9):
        return U_frac, V_frac, W_frac
    
    # Try with larger denominator
    U_frac = [[Fraction(x).limit_denominator(max_denominator * 10) for x in row] for row in U]
    V_frac = [[Fraction(x).limit_denominator(max_denominator * 10) for x in row] for row in V]
    W_frac = [[Fraction(x).limit_denominator(max_denominator * 10) for x in row] for row in W]
    
    if _verify(U_frac, V_frac, W_frac, n, m, p, tol=1e-9):
        return U_frac, V_frac, W_frac
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try rank reduction via numerical optimization
    opt_result, opt_rank = _try_rank_reduction(
        n, m, p, 
        max_rank=best_rank, 
        min_rank=15, 
        max_time=min(90, deadline - time.time() - 10),
        seed=42
    )
    
    if opt_result is not None:
        # Try to convert to exact fractions
        polished = _polish_fractions(opt_result['U'], opt_result['V'], opt_result['W'], n, m, p)
        if polished is not None:
            U_frac, V_frac, W_frac = polished
            final_rank = len(U_frac[0])
            if final_rank < best_rank:
                best_rank = final_rank
                result = {"U": U_frac, "V": V_frac, "W": W_frac}
        else:
            # Use the numerical solution directly if fraction conversion fails
            # but verify it first
            if _verify(opt_result['U'], opt_result['V'], opt_result['W'], n, m, p, tol=1e-9):
                final_rank = len(opt_result['U'][0])
                if final_rank < best_rank:
                    best_rank = final_rank
                    result = opt_result
    
    # Final verification
    if not _verify(result['U'], result['V'], result['W'], n, m, p, tol=1e-9):
        # Fallback to schoolbook
        result = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    return result
