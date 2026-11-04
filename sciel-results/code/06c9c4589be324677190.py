"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition of the all-ones tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
    return np.allclose(T, 1.0, atol=tol)


def _verify_fast(U, V, W, n, m, p, rank, tol=1e-7):
    """Faster verification using batched operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _rank_of_matrix(M):
    """Compute numerical rank of a matrix."""
    return np.linalg.matrix_rank(M, tol=1e-9)


def _try_rank_grad(n, m, p, target_rank, max_time=8.0, seed=42):
    """
    Try to find a decomposition of the given rank using gradient descent
    on the loss ||T(U,V,W) - 1||^2.
    
    The tensor entry is T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    We want T[i,j,k] = 1 for all i,j,k.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Lower bound on rank
    lb = max(n * m, m * p, p * n)
    if target_rank < lb:
        return None
    
    # Initialize
    U = rng.randn(n * m, target_rank) * 0.5
    V = rng.randn(m * p, target_rank) * 0.5
    W = rng.randn(p * n, target_rank) * 0.5
    
    # Precompute index mappings
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
    
    n_entries = len(indices)
    
    def compute_loss_and_grads(U, V, W):
        loss = 0.0
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for (ui, vi, wi) in indices:
            # T[i,j,k] = sum_c U[ui,c] * V[vi,c] * W[wi,c]
            t_val = np.dot(U[ui, :], V[vi, :] * W[wi, :])
            err = t_val - 1.0
            loss += err * err
            
            # Gradient w.r.t. U[ui,c] = err * V[vi,c] * W[wi,c]
            gU[ui, :] += 2.0 * err * V[vi, :] * W[wi, :]
            # Gradient w.r.t. V[vi,c] = err * U[ui,c] * W[wi,c]
            gV[vi, :] += 2.0 * err * U[ui, :] * W[wi, :]
            # Gradient w.r.t. W[wi,c] = err * U[ui,c] * V[vi,c]
            gW[wi, :] += 2.0 * err * U[ui, :] * V[vi, :]
        
        return loss, gU, gV, gW
    
    # Adam optimizer
    lr = 0.05
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    t = 0
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() - start_time > max_time:
            break
        
        loss, gU, gV, gW = compute_loss_and_grads(U, V, W)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            return best_U, best_V, best_W
        
        t += 1
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        
        mU = beta1 * mU + (1 - beta1) * gU
        vU = beta2 * vU + (1 - beta2) * gU ** 2
        mV = beta1 * mV + (1 - beta1) * gV
        vV = beta2 * vV + (1 - beta2) * gV ** 2
        mW = beta1 * mW + (1 - beta1) * gW
        vW = beta2 * vW + (1 - beta2) * gW ** 2
        
        mU_hat = mU / bc1; vU_hat = vU / bc2
        mV_hat = mV / bc1; vV_hat = vV / bc2
        mW_hat = mW / bc1; vW_hat = vW / bc2
        
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        # Periodic re-initialization if stuck
        if it > 0 and it % 500 == 0 and loss > 0.1:
            # Try a different initialization
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            mU = np.zeros_like(U); vU = np.zeros_like(U)
            mV = np.zeros_like(V); vV = np.zeros_like(V)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            t = 0
            lr = 0.05
    
    # Final check
    if best_loss < 1e-8:
        return best_U, best_V, best_W
    return None


def _try_rank_random(n, m, p, target_rank, max_time=5.0, n_trials=200):
    """Try random initializations with a few gradient steps."""
    rng = np.random.RandomState(123)
    start_time = time.time()
    
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
    
    for trial in range(n_trials):
        if time.time() - start_time > max_time:
            break
        
        U = rng.randn(n * m, target_rank) * 0.3
        V = rng.randn(m * p, target_rank) * 0.3
        W = rng.randn(p * n, target_rank) * 0.3
        
        # A few gradient steps
        lr = 0.1
        for step in range(20):
            if time.time() - start_time > max_time:
                break
            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)
            loss = 0.0
            for (ui, vi, wi) in indices:
                t_val = np.dot(U[ui, :], V[vi, :] * W[wi, :])
                err = t_val - 1.0
                loss += err * err
                gU[ui, :] += 2.0 * err * V[vi, :] * W[wi, :]
                gV[vi, :] += 2.0 * err * U[ui, :] * W[wi, :]
                gW[wi, :] += 2.0 * err * U[ui, :] * V[vi, :]
            
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            
            if loss < 1e-10:
                return U, V, W
        
        # Check final
        loss = 0.0
        for (ui, vi, wi) in indices:
            t_val = np.dot(U[ui, :], V[vi, :] * W[wi, :])
            err = t_val - 1.0
            loss += err * err
        if loss < 1e-10:
            return U, V, W
    
    return None


def _to_rational_list(arr, tol=1e-6):
    """Convert array to list of lists, rounding to nearest integer or half-integer if close."""
    result = []
    for row in arr:
        new_row = []
        for val in row:
            # Try to round to nearest multiple of 0.5
            nearest_half = round(val * 2) / 2.0
            if abs(val - nearest_half) < tol:
                new_row.append(nearest_half)
            else:
                # Try integer
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    new_row.append(float(nearest_int))
                else:
                    new_row.append(float(val))
        result.append(new_row)
    return result


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline guard: 140s to be safe (evaluator kills at 150s)
    deadline = time.time() + 140
    
    # Build schoolbook as fallback
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    
    # Lower bound: rank >= max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound upward
    best_result = None
    best_rank = schoolbook_rank
    
    for r in range(lb, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 10
        if remaining < 2:
            break
        
        # Try gradient-based approach
        max_time = min(remaining * 0.5, 8.0)
        result = _try_rank_grad(n, m, p, r, max_time=max_time, seed=r)
        
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, n, m, p, r, tol=1e-7):
                U_list = _to_rational_list(U)
                V_list = _to_rational_list(V)
                W_list = _to_rational_list(W)
                if _verify(U_list, V_list, W_list, n, m, p, r, tol=1e-5):
                    return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try random approach with shorter time
        if time.time() < deadline - 10:
            max_time_r = min(remaining * 0.2, 3.0)
            result = _try_rank_random(n, m, p, r, max_time=max_time_r, n_trials=50)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p, r, tol=1e-7):
                    U_list = _to_rational_list(U)
                    V_list = _to_rational_list(V)
                    W_list = _to_rational_list(W)
                    if _verify(U_list, V_list, W_list, n, m, p, r, tol=1e-5):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If we found something at this rank, stop searching lower ranks
        # (but we already would have returned)
    
    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
