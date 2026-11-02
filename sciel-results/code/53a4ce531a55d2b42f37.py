"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 16
    dim_W = p * n  # 12
    
    # Precompute indices for fast error calculation
    # T[i, j, k] = 1
    # Error = sum_{i,j,k} (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1)^2
    
    def calculate_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(len(U[0])):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    diff = val - 1.0
                    err += diff * diff
        return err

    def local_search(rank, seed, max_iters=2000):
        rng = np.random.default_rng(seed)
        # Initialize with small integers
        U = [[0]*rank for _ in range(dim_U)]
        V = [[0]*rank for _ in range(dim_V)]
        W = [[0]*rank for _ in range(dim_W)]
        
        # Random initialization from {-1, 0, 1}
        for idx in range(dim_U):
            for c in range(rank):
                U[idx][c] = int(rng.integers(-1, 2))
        for idx in range(dim_V):
            for c in range(rank):
                V[idx][c] = int(rng.integers(-1, 2))
        for idx in range(dim_W):
            for c in range(rank):
                W[idx][c] = int(rng.integers(-1, 2))
                
        current_error = calculate_error(U, V, W)
        
        for _ in range(max_iters):
            if current_error < 1e-9:
                return U, V, W, current_error
            
            # Pick random slice and coordinate
            c = int(rng.integers(0, rank))
            tensor_choice = int(rng.integers(0, 3))
            
            if tensor_choice == 0:
                idx = int(rng.integers(0, dim_U))
                old_val = U[idx][c]
                # Try perturbations
                for delta in [-1, 1, -0.5, 0.5]:
                    new_val = old_val + delta
                    U[idx][c] = new_val
                    new_error = calculate_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        break
                    else:
                        U[idx][c] = old_val
            elif tensor_choice == 1:
                idx = int(rng.integers(0, dim_V))
                old_val = V[idx][c]
                for delta in [-1, 1, -0.5, 0.5]:
                    new_val = old_val + delta
                    V[idx][c] = new_val
                    new_error = calculate_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        break
                    else:
                        V[idx][c] = old_val
            else:
                idx = int(rng.integers(0, dim_W))
                old_val = W[idx][c]
                for delta in [-1, 1, -0.5, 0.5]:
                    new_val = old_val + delta
                    W[idx][c] = new_val
                    new_error = calculate_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        break
                    else:
                        W[idx][c] = old_val
                        
        return U, V, W, current_error

    # Try ranks in increasing order
    # Schoolbook rank is 48. We try to beat it.
    target_ranks = [20, 24, 28, 32, 36, 40, 44, 48]
    
    for rank in target_ranks:
        # Multiple random restarts
        for trial in range(10):
            U, V, W, err = local_search(rank, seed=42 + rank*100 + trial, max_iters=500)
            if err < 1e-9:
                # Convert to Fractions
                def to_frac(val):
                    if abs(val) < 1e-9:
                        return Fraction(0, 1)
                    # Check if it's a half-integer or integer
                    if abs(val - round(val)) < 1e-9:
                        return Fraction(int(round(val)), 1)
                    if abs(val * 2 - round(val * 2)) < 1e-9:
                        return Fraction(int(round(val * 2)), 2)
                    # Fallback
                    return Fraction(val).limit_denominator(100)
                
                U_frac = [[to_frac(val) for val in row] for row in U]
                V_frac = [[to_frac(val) for val in row] for row in V]
                W_frac = [[to_frac(val) for val in row] for row in W]
                
                return {"U": U_frac, "V": V_frac, "W": W_frac}
                
    # Fallback to schoolbook if no better solution found
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
# EVOLVE-BLOCK-END
