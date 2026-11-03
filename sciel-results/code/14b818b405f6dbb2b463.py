"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
import random
import itertools

DEADLINE = time.time() + 140  # 140s safety margin before 150s kill

def _schoolbook_decomposition(n, m, p):
    """Construct the standard rank n*m*p tensor decomposition for matrix multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the given decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U_np = np.array(U, dtype=object)
    V_np = np.array(V, dtype=object)
    W_np = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_np[i * m + j]
                v_row = V_np[j * p + k]
                w_row = W_np[k * n + i]
                s = sum(u_row[r] * v_row[r] * w_row[r] for r in range(rank))
                if s != 1:
                    return False
    return True

def _verify_numeric(U, V, W, n, m, p, tol=1e-9):
    """Fast numeric verification."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_random_search(n, m, p, target_rank, max_iters=500, seed=42):
    """Try random initialization + local search for low-rank decomposition."""
    if time.time() > DEADLINE - 10:
        return None
    
    rng = random.Random(seed)
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    best_residual = float('inf')
    best_result = None
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 15:
            break
        
        # Random initialization with small values
        U = [[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(U_dim)]
        V = [[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(V_dim)]
        W = [[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(W_dim)]
        
        # Local search: try to reduce residual
        residual = _compute_residual(U, V, W, n, m, p)
        
        if residual < best_residual:
            best_residual = residual
            best_result = (U[:], V[:], W[:])
        
        if residual < 1e-10:
            # Try to round to rationals
            rounded = _try_round_to_rationals(U, V, W, n, m, p)
            if rounded is not None:
                return rounded
        
        # Simple local search: perturb one random entry
        for _ in range(10):
            if time.time() > DEADLINE - 15:
                break
            
            # Choose which matrix and entry to perturb
            mat_idx = rng.choice([0, 1, 2])
            if mat_idx == 0:
                row = rng.randint(0, U_dim - 1)
                col = rng.randint(0, target_rank - 1)
                old_val = U[row][col]
                U[row][col] += rng.uniform(-0.1, 0.1)
            elif mat_idx == 1:
                row = rng.randint(0, V_dim - 1)
                col = rng.randint(0, target_rank - 1)
                old_val = V[row][col]
                V[row][col] += rng.uniform(-0.1, 0.1)
            else:
                row = rng.randint(0, W_dim - 1)
                col = rng.randint(0, target_rank - 1)
                old_val = W[row][col]
                W[row][col] += rng.uniform(-0.1, 0.1)
            
            new_residual = _compute_residual(U, V, W, n, m, p)
            
            if new_residual >= residual:
                # Revert
                if mat_idx == 0:
                    U[row][col] = old_val
                elif mat_idx == 1:
                    V[row][col] = old_val
                else:
                    W[row][col] = old_val
            else:
                residual = new_residual
        
        if residual < best_residual:
            best_residual = residual
            best_result = (U[:], V[:], W[:])
    
    return best_result

def _compute_residual(U, V, W, n, m, p):
    """Compute the sum of squared residuals."""
    rank = len(U[0])
    residual = 0.0
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                residual += (s - 1.0) ** 2
    return residual

def _try_round_to_rationals(U, V, W, n, m, p):
    """Try to round to simple rationals and verify."""
    rank = len(U[0])
    
    # Try rounding to nearest 1/2
    U_r = [[round(U[i][r] * 2) / 2 for r in range(rank)] for i in range(len(U))]
    V_r = [[round(V[i][r] * 2) / 2 for r in range(rank)] for i in range(len(V))]
    W_r = [[round(W[i][r] * 2) / 2 for r in range(rank)] for i in range(len(W))]
    
    if _verify_decomposition(U_r, V_r, W_r, n, m, p):
        return {"U": U_r, "V": V_r, "W": W_r}
    
    # Try rounding to nearest integer
    U_r = [[round(U[i][r]) for r in range(rank)] for i in range(len(U))]
    V_r = [[round(V[i][r]) for r in range(rank)] for i in range(len(V))]
    W_r = [[round(W[i][r]) for r in range(rank)] for i in range(len(W))]
    
    if _verify_decomposition(U_r, V_r, W_r, n, m, p):
        return {"U": U_r, "V": V_r, "W": W_r}
    
    return None

def _try_structured_search(n, m, p, target_rank):
    """Try structured patterns for low-rank decomposition."""
    if time.time() > DEADLINE - 10:
        return None
    
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Try patterns where each rank-1 term has a specific structure
    # For matrix multiplication, we can try to group terms
    
    best_residual = float('inf')
    best_result = None
    
    # Try random sparse patterns
    for trial in range(100):
        if time.time() > DEADLINE - 15:
            break
        
        rng = random.Random(1000 + trial)
        
        U = [[0.0] * target_rank for _ in range(U_dim)]
        V = [[0.0] * target_rank for _ in range(V_dim)]
        W = [[0.0] * target_rank for _ in range(W_dim)]
        
        # Randomly assign nonzero entries
        for r in range(target_rank):
            # Choose which (i,j) pairs get nonzero U values
            num_u = rng.randint(1, min(3, U_dim))
            u_indices = rng.sample(range(U_dim), num_u)
            for idx in u_indices:
                U[idx][r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            
            num_v = rng.randint(1, min(3, V_dim))
            v_indices = rng.sample(range(V_dim), num_v)
            for idx in v_indices:
                V[idx][r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            
            num_w = rng.randint(1, min(3, W_dim))
            w_indices = rng.sample(range(W_dim), num_w)
            for idx in w_indices:
                W[idx][r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        
        residual = _compute_residual(U, V, W, n, m, p)
        if residual < best_residual:
            best_residual = residual
            best_result = (U[:], V[:], W[:])
        
        if residual < 1e-10:
            rounded = _try_round_to_rationals(U, V, W, n, m, p)
            if rounded is not None:
                return rounded
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Get schoolbook as fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    
    # Try to find lower rank decompositions
    # Start from rank 30 and go up
    for target_rank in range(30, schoolbook_rank):
        if time.time() > DEADLINE - 20:
            break
        
        # Try random search
        result = _try_random_search(n, m, p, target_rank, max_iters=100)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # Try structured search
        result = _try_structured_search(n, m, p, target_rank)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Return schoolbook decomposition
    return schoolbook
# EVOLVE-BLOCK-END
