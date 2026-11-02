"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = U.shape[1]
    # Build the tensor T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.allclose(T, 1.0, atol=tol)


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _try_fixed_U(n, m, p, target_rank, time_limit, seed=42):
    """
    Try to find a rank-`target_rank` decomposition by fixing U and V to sparse patterns
    and solving for W.
    
    The tensor equation is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    For fixed U and V, this is linear in W.
    For each pair (i, k), we have m equations (one for each j) involving the same W[k*n+i, :].
    Let w_ik = W[k*n+i, :] (a vector of length R).
    The equations are:
    sum_r (U[i*m+j, r] * V[j*p+k, r]) * w_ik[r] = 1 for j=0..m-1.
    
    This can be written as M_ik @ w_ik = 1_m, where
    M_ik[j, r] = U[i*m+j, r] * V[j*p+k, r].
    M_ik has shape (m, R).
    
    We need to find w_ik that satisfies this system.
    If R >= m, a solution generally exists (least squares).
    We check if the residual is close to 0.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # We iterate over ranks from target_rank up to a reasonable limit, or just try target_rank.
    # The caller passes target_rank. We will try to find a solution for that specific rank.
    R = target_rank
    
    # Try multiple random attempts for this rank
    num_attempts = 200
    for attempt in range(num_attempts):
        if time.time() > start + time_limit - 1:
            break
        
        # Generate sparse U: each column has 2-3 non-zero entries
        U = np.zeros((n * m, R))
        for r in range(R):
            num_nonzero = rng.choice([2, 3])
            positions = rng.choice(n * m, size=num_nonzero, replace=False)
            for pos in positions:
                U[pos, r] = rng.choice([-1, 1])
        
        # Generate sparse V: each column has 2-3 non-zero entries
        V = np.zeros((m * p, R))
        for r in range(R):
            num_nonzero = rng.choice([2, 3])
            positions = rng.choice(m * p, size=num_nonzero, replace=False)
            for pos in positions:
                V[pos, r] = rng.choice([-1, 1])
        
        # Solve for W
        W = np.zeros((p * n, R))
        success = True
        for i in range(n):
            for k in range(p):
                # Build M_ik of shape (m, R)
                M_ik = np.zeros((m, R))
                for j in range(m):
                    M_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Solve M_ik @ w = ones(m)
                # Use least squares to handle underdetermined or overdetermined cases
                try:
                    # If R >= m, we can find a minimum norm solution or any solution.
                    # np.linalg.lstsq gives the minimum norm solution for underdetermined systems.
                    w_sol, residuals, rank, sv = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                    
                    # Check if the solution is good
                    # Residuals are sum of squared residuals.
                    if len(residuals) > 0 and residuals[0] < 1e-9:
                        W[k * n + i, :] = w_sol
                    else:
                        # Fallback: try pseudo-inverse explicitly if lstsq fails or is not exact
                        # For integer/half-integer targets, we might need exact solutions.
                        # But lstsq should be fine for verification.
                        # If not exact, this attempt fails.
                        success = False
                        break
                except np.linalg.LinAlgError:
                    success = False
                    break
            
            if not success:
                break
        
        if success:
            # Verify the full tensor equation
            if _verify(U, V, W, n, m, p):
                return (U, V, W)
    
    return None


def _try_gradient_descent(n, m, p, target_rank, time_limit, seed=42):
    """
    Use gradient descent to find a rank-`target_rank` decomposition.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # Initialize with random values
    U = rng.randn(n * m, target_rank) * 0.1
    V = rng.randn(m * p, target_rank) * 0.1
    W = rng.randn(p * n, target_rank) * 0.1
    
    learning_rate = 0.01
    best_loss = float('inf')
    best_params = (U.copy(), V.copy(), W.copy())
    
    for step in range(10000):
        if time.time() > start + time_limit - 2:
            break
        
        # Compute tensor T
        T = np.zeros((n, m, p))
        for r in range(target_rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        # Compute loss
        loss = np.sum((T - 1.0) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_params = (U.copy(), V.copy(), W.copy())
        
        if loss < 1e-12:
            U, V, W = best_params
            if _verify(U, V, W, n, m, p):
                return (U, V, W)
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(target_rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = T[i, j, k] - 1.0
                        dU[i * m + j, r] += 2 * err * v[j, k] * w[k, i]
                        dV[j * p + k, r] += 2 * err * u[i, j] * w[k, i]
                        dW[k * n + i, r] += 2 * err * u[i, j] * v[j, k]
        
        # Update
        U -= learning_rate * dU
        V -= learning_rate * dV
        W -= learning_rate * dW
    
    U, V, W = best_params
    if _verify(U, V, W, n, m, p):
        return (U, V, W)
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = n * m * p
    
    # Try to find a better decomposition using fixed-U approach
    # Start from lower ranks and work up
    # The theoretical lower bound is max(n*m, m*p, p*n) = 12.
    # Known lower bound for 3x3x4 is 19? Or lower?
    # Let's try from 12 upwards.
    for target_rank in range(12, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        remaining_time = deadline - time.time() - 5
        if remaining_time < 2:
            break
        
        # Use different seeds for diversity
        seed = 42 + target_rank
        result = _try_fixed_U(n, m, p, target_rank, remaining_time, seed=seed)
        if result is not None:
            U, V, W = result
            # Double check verification
            if _verify(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                break
    
    return best_result

# EVOLVE-BLOCK-END
