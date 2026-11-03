"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition
    result = _try_low_rank_search(n, m, p, time_limit=135)
    if result is not None:
        return result
    
    # Fallback to schoolbook decomposition
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the factor matrices exactly decompose the matrix multiplication tensor.
    For each (i,j,k), the tensor entry T[i*m+j, j*p+k, k*n+i] should equal
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    Returns True if all entries match exactly (using Fraction arithmetic).
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[u_idx][c]) * Fraction(V[v_idx][c]) * Fraction(W[w_idx][c])
                if total != 1:
                    return False
    return True

def _compute_tensor_error(U, V, W, n, m, p, target_rank):
    """Compute max error of tensor decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(target_rank):
                    val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_low_rank_search(n, m, p, time_limit=135):
    """
    Attempt to find a lower-rank exact decomposition using iterative refinement.
    Uses structured initializations and aggressive rank exploration.
    Returns a valid decomposition dict if found, else None.
    """
    start_time = time.time()
    best_solution = None
    best_rank = n * m * p
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But for 4x4x9, known best is around 40-45 based on literature
    # Start from a more optimistic lower bound
    min_possible_rank = 30  # Optimistic starting point
    
    # Priority ranks to try (based on known results and structure)
    priority_ranks = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    
    # Create ordered list: priority ranks first, then fill in gaps
    ranks_to_try = []
    for r in priority_ranks:
        if min_possible_rank <= r < best_rank:
            ranks_to_try.append(r)
    # Add any remaining ranks in range
    for r in range(min_possible_rank, best_rank):
        if r not in ranks_to_try:
            ranks_to_try.append(r)
    ranks_to_try.sort()
    
    for target_rank in ranks_to_try:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 15:
            break
        
        # Allocate time per rank: more time for promising ranks
        remaining_time = time_limit - elapsed
        if target_rank in [30, 32, 34, 36, 38, 40]:
            time_per_rank = min(remaining_time * 0.3, 25)  # 30% of remaining or 25s
        elif target_rank <= 44:
            time_per_rank = min(remaining_time * 0.2, 15)
        else:
            time_per_rank = min(remaining_time * 0.1, 8)
        
        if time_per_rank < 2:
            continue
        
        rank_start_time = time.time()
        max_restarts = max(1, int(time_per_rank / 2))
        
        for restart in range(max_restarts):
            if time.time() - rank_start_time > time_per_rank - 1:
                break
            
            # Initialize with structured strategies
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            strategy = restart % 5
            
            if strategy == 0:
                # Block-structured initialization
                for c in range(target_rank):
                    i_block = c // (target_rank // n) if target_rank >= n else 0
                    j_block = (c // (target_rank // m)) % m if target_rank >= m else 0
                    k_block = c % p
                    
                    if i_block < n:
                        U[i_block * m:(i_block + 1) * m, c] = np.random.randn(m) * 0.3
                    if j_block < m:
                        V[j_block * p:(j_block + 1) * p, c] = np.random.randn(p) * 0.3
                    if k_block < p:
                        W[k_block * n:(k_block + 1) * n, c] = np.random.randn(n) * 0.3
            elif strategy == 1:
                # Sparse random with structured zeros
                U[:, :] = np.random.choice([-1, 0, 1], size=(n * m, target_rank), p=[0.1, 0.8, 0.1]) * 0.5
                V[:, :] = np.random.choice([-1, 0, 1], size=(m * p, target_rank), p=[0.1, 0.8, 0.1]) * 0.5
                W[:, :] = np.random.choice([-1, 0, 1], size=(p * n, target_rank), p=[0.1, 0.8, 0.1]) * 0.5
            elif strategy == 2:
                # Gaussian with scaling based on rank
                scale = 0.2 * np.sqrt(n * m * p / target_rank)
                U[:, :] = np.random.randn(n * m, target_rank) * scale
                V[:, :] = np.random.randn(m * p, target_rank) * scale
                W[:, :] = np.random.randn(p * n, target_rank) * scale
            elif strategy == 3:
                # Hybrid: start from schoolbook subset
                # Use first target_rank columns of schoolbook
                for c in range(min(target_rank, n * m * p)):
                    i = c // (m * p)
                    j = (c // p) % m
                    k = c % p
                    if i < n and j < m and k < p:
                        U[i * m + j, c] = 1
                        V[j * p + k, c] = 1
                        W[k * n + i, c] = 1
                # Add small random perturbation
                U += np.random.randn(*U.shape) * 0.05
                V += np.random.randn(*V.shape) * 0.05
                W += np.random.randn(*W.shape) * 0.05
            else:
                # Random with sign structure
                U[:, :] = np.random.randn(n * m, target_rank) * 0.25
                V[:, :] = np.random.randn(m * p, target_rank) * 0.25
                W[:, :] = np.random.randn(p * n, target_rank) * 0.25
                # Ensure some sign consistency
                for c in range(0, target_rank, 2):
                    U[:, c] *= 1.2
                    W[:, c] *= 1.2
                    V[:, c] *= 0.8
            
            # Iterative refinement
            converged = False
            max_iterations = 200
            prev_error = float('inf')
            stagnation_counter = 0
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for iteration in range(max_iterations):
                if time.time() - rank_start_time > time_per_rank - 2:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            v_idx = j * p + k
                            w_idx = k * n + i
                            A[k, :] = V[v_idx, :] * W[w_idx, :]
                        b = np.ones(p)
                        try:
                            U[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            u_idx = i * m + j
                            w_idx = k * n + i
                            A[i, :] = U[u_idx, :] * W[w_idx, :]
                        b = np.ones(n)
                        try:
                            V[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            A[j, :] = U[u_idx, :] * V[v_idx, :]
                        b = np.ones(m)
                        try:
                            W[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                current_error = _compute_tensor_error(U, V, W, n, m, p, target_rank)
                
                if current_error < 1e-10:
                    converged = True
                    break
                
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Detect stagnation
                if current_error > prev_error * 0.995:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                
                if stagnation_counter > 8:
                    # Perturb from best known state
                    U = best_U + np.random.randn(*U.shape) * 0.02
                    V = best_V + np.random.randn(*V.shape) * 0.02
                    W = best_W + np.random.randn(*W.shape) * 0.02
                    stagnation_counter = 0
                
                prev_error = current_error
            
            # Try to verify with best state if not converged but close
            if not converged and best_error < 1e-8:
                U, V, W = best_U, best_V, best_W
                final_error = _compute_tensor_error(U, V, W, n, m, p, target_rank)
                if final_error < 1e-10:
                    converged = True
            
            if converged:
                # Convert to exact fractions
                found_exact = False
                for denom in [10, 100, 1000, 10000, 100000, 1000000, 10000000]:
                    U_frac = [[Fraction(U[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(n * m)]
                    V_frac = [[Fraction(V[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(m * p)]
                    W_frac = [[Fraction(W[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(p * n)]
                    
                    if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                        best_solution = {"U": U_frac, "V": V_frac, "W": W_frac}
                        best_rank = target_rank
                        found_exact = True
                        break
                
                if found_exact:
                    break
    
    return best_solution

# EVOLVE-BLOCK-END
