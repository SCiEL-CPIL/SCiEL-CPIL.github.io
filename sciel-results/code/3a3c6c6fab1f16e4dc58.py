"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _dft_initialize(n, m, p, rank, seed=42):
    """Initialize U, V, W using DFT structure on the middle dimension."""
    np.random.seed(seed)
    
    # 4-point DFT matrix
    omega = np.exp(2j * np.pi / m)
    F = np.array([[omega**(j*l) for l in range(m)] for j in range(m)])
    F_inv = np.conj(F).T / m
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Distribute rank across frequency components
    # Each frequency l gets rank/m components (if divisible)
    rank_per_freq = rank // m
    remainder = rank % m
    
    col = 0
    for l in range(m):
        r_l = rank_per_freq + (1 if l < remainder else 0)
        for c in range(r_l):
            # For frequency component l, use DFT weights
            for i in range(n):
                for j in range(m):
                    U[i*m+j, col] = F[j, l].real  # Use real part for real-valued decomposition
                for k in range(p):
                    V[j*p+k, col] = F_inv[l, j].real  # Inverse DFT weight
                W[k*n+i, col] = 1.0 / m  # Normalization
            col += 1
    
    # Fill remaining columns with random values
    if col < rank:
        U[:, col:] = np.random.randn(n*m, rank - col)
        V[:, col:] = np.random.randn(m*p, rank - col)
        W[:, col:] = np.random.randn(p*n, rank - col)
    
    return U, V, W

def _try_rank_structured(n, m, p, target_rank, max_time=15, use_dft=True):
    """Try to find a decomposition of the given rank using structured random search."""
    start_time = time.time()
    best = None
    
    target = np.ones((n, m, p))
    
    # Try DFT initialization first
    if use_dft:
        U, V, W = _dft_initialize(n, m, p, target_rank, seed=0)
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
        if np.allclose(T, target, atol=1e-6):
            return U, V, W
    
    # Random search with various seeds
    for attempt in range(5000):
        if time.time() - start_time > max_time:
            break
        
        # Random initialization with different strategies
        if attempt % 3 == 0:
            # Pure random
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
        elif attempt % 3 == 1:
            # DFT-inspired
            U, V, W = _dft_initialize(n, m, p, target_rank, seed=attempt)
        else:
            # Sparse random
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            # Make sparse
            mask = np.random.rand(n*m, target_rank) < 0.5
            U[mask] = 0
            mask = np.random.rand(m*p, target_rank) < 0.5
            V[mask] = 0
            mask = np.random.rand(p*n, target_rank) < 0.5
            W[mask] = 0
        
        # Check if this gives the right tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
        
        if np.allclose(T, target, atol=1e-6):
            return U, V, W
    
    return None

def _gradient_descent_polish(U, V, W, n, m, p, max_iter=100, lr=0.01, tol=1e-8):
    """Polish the decomposition using gradient descent."""
    target = np.ones((n, m, p))
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for iteration in range(max_iter):
        # Compute current tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
        
        # Check convergence
        error = np.linalg.norm(T - target)
        if error < tol:
            break
        
        # Compute gradients (simplified)
        # dT/dc = U[i*m+j] * V[j*p+k] * W[k*n+i]
        # Gradient of loss = ||T - target||^2 w.r.t. U, V, W
        
        # For simplicity, use a simple update rule
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = T[i,j,k] - target[i,j,k]
                    # Gradient w.r.t. U[i*m+j, c] = diff * V[j*p+k, c] * W[k*n+i, c]
                    dU[i*m+j,:] += diff * V[j*p+k,:] * W[k*n+i,:]
                    dV[j*p+k,:] += diff * U[i*m+j,:] * W[k*n+i,:]
                    dW[k*n+i,:] += diff * U[i*m+j,:] * V[j*p+k,:]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Adaptive learning rate
        if error > 1.0:
            lr *= 0.5
        elif error < 0.1:
            lr *= 1.2
    
    return U, V, W, np.linalg.norm(T - target)

def _to_rational_list(arr, tol=1e-6):
    """Convert array to list with integer/half-integer entries if close."""
    result = []
    for row in arr:
        new_row = []
        for val in row:
            # Try to find nearest rational with small denominator
            best_val = val
            best_err = abs(val)
            for denom in [1, 2, 4, 8]:
                candidate = round(val * denom) / denom
                err = abs(val - candidate)
                if err < best_err:
                    best_err = err
                    best_val = candidate
            if best_err < tol:
                new_row.append(best_val)
            else:
                new_row.append(float(val))
        result.append(new_row)
    return result

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    deadline = time.time() + 110  # 10s safety margin before 120s limit
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound upward
    best_result = None
    best_rank = schoolbook_rank
    
    for r in range(lower_bound, min(schoolbook_rank + 1, 50)):
        if time.time() > deadline - 10:
            break
        
        remaining_time = deadline - time.time() - 10
        if remaining_time < 2:
            break
        
        result = _try_rank_structured(n, m, p, r, max_time=min(15, remaining_time), use_dft=True)
        if result is not None:
            U, V, W = result
            
            # Polish with gradient descent
            U, V, W, final_error = _gradient_descent_polish(U, V, W, n, m, p, max_iter=50, lr=0.005)
            
            if final_error < 1e-5:
                # Convert to rational list
                U_list = _to_rational_list(U)
                V_list = _to_rational_list(V)
                W_list = _to_rational_list(W)
                
                # Verify
                if _verify(U_list, V_list, W_list, n, m, p, r, tol=1e-4):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    # Found a valid decomposition, return immediately
                    return best_result
    
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
