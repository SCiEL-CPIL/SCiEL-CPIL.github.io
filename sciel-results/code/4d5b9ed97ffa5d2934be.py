<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]
import time
import numpy as np
from fractions import Fraction

def _schoolbook_3x3x8():
    """Returns schoolbook decomposition for 3x3x8 with rank 72."""
    n, m, p = 3, 3, 8
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tolerance=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The matrix multiplication tensor is: T[i,j,k] = 1 if this is a valid term, else 0
    # Actually, the tensor for matrix multiplication C = A*B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 (it's a rank-1 tensor for each (i,j,k) triple)
    # Wait, no. The tensor is: sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # For the multiplication to be correct, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # But that's not right either. Let me think again.
    
    # The matrix multiplication tensor M satisfies:
    # M[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: M = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > tolerance:
                    return False
    return True

def _try_rank_r_decomposition(n, m, p, r, time_budget=10.0):
    """Try to find a rank-r decomposition using random search."""
    start_time = time.time()
    np.random.seed(42)
    
    # Coefficients to try: -2, -1, -0.5, 0, 0.5, 1, 2
    coeffs = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    # Initialize with random sparse patterns
    for attempt in range(1000):
        if time.time() - start_time > time_budget:
            break
            
        # Reset
        U[:] = 0
        V[:] = 0
        W[:] = 0
        
        # Each column gets a random sparse pattern
        for col in range(r):
            # Choose random non-zero positions
            n_nonzero_U = np.random.randint(1, 4)
            n_nonzero_V = np.random.randint(1, 4)
            n_nonzero_W = np.random.randint(1, 4)
            
            idx_U = np.random.choice(n * m, size=n_nonzero_U, replace=False)
            idx_V = np.random.choice(m * p, size=n_nonzero_V, replace=False)
            idx_W = np.random.choice(p * n, size=n_nonzero_W, replace=False)
            
            for idx in idx_U:
                U[idx, col] = np.random.choice(coeffs)
            for idx in idx_V:
                V[idx, col] = np.random.choice(coeffs)
            for idx in idx_W:
                W[idx, col] = np.random.choice(coeffs)
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    """
    Construct the decomposition for 3x3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook (rank 72) as fallback
    best_U, best_V, best_W = _schoolbook_3x3x8()
    best_rank = 72
    
    # Try to find better decompositions
    # Strategy: Try ranks from 50 to 71, but this is computationally expensive
    # Instead, let's try a smarter approach: block decomposition
    
    # Block approach: split p=8 into blocks
    # Try 3+3+2: need rank for 3x3x3 and 3x3x2
    # 3x3x3 schoolbook: 27, 3x3x2 schoolbook: 18, total: 72 (no gain)
    # If we can find rank 23 for 3x3x3: 23+23+18 = 64
    
    # Let's try to construct a rank-23 decomposition for 3x3x3
    # Using a known structure from Winograd-like methods
    
    # Actually, let me try a different approach: use the fact that we can
    # combine blocks more cleverly.
    
    # For now, return schoolbook if we can't find better quickly
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try random search for rank 64
    result = _try_rank_r_decomposition(n, m, p, 64, time_budget=10.0)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try rank 65
    result = _try_rank_r_decomposition(n, m, p, 65, time_budget=10.0)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try rank 66
    result = _try_rank_r_decomposition(n, m, p, 66, time_budget=10.0)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]
import time
import numpy as np
from fractions import Fraction
import itertools

def _schoolbook_3x3x8():
    """Returns schoolbook decomposition for 3x3x8 with rank 72."""
    n, m, p = 3, 3, 8
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tolerance=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > tolerance:
                    return False
    return True

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual tensor (what we need to fix)."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    residual = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                residual[i, j, k] = 1.0 - val
    return residual

def _local_polish(U, V, W, n, m, p, max_iterations=200, tol=1e-12):
    """
    Local polish: try to fix small violations by adjusting entries.
    Uses a coordinate descent approach targeting the worst violations.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for iteration in range(max_iterations):
        residual = _compute_residual(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p)
        max_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        max_val = residual[max_idx]
        
        if abs(max_val) < tol:
            break
        
        i, j, k = max_idx
        found_improvement = False
        
        # Try adjusting entries that directly affect this (i,j,k) term
        for col in range(rank):
            u_old = U_arr[i * m + j, col]
            v_old = V_arr[j * p + k, col]
            w_old = W_arr[k * n + i, col]
            
            # Try combinations of small adjustments
            for du in [-1, -0.5, 0.5, 1]:
                for dv in [-1, -0.5, 0.5, 1]:
                    for dw in [-1, -0.5, 0.5, 1]:
                        U_arr[i * m + j, col] = u_old + du
                        V_arr[j * p + k, col] = v_old + dv
                        W_arr[k * n + i, col] = w_old + dw
                        
                        new_residual = _compute_residual(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p)
                        new_max = np.max(np.abs(new_residual))
                        old_max = np.max(np.abs(residual))
                        
                        if new_max < old_max - 1e-15:
                            found_improvement = True
                            break
                        else:
                            U_arr[i * m + j, col] = u_old
                            V_arr[j * p + k, col] = v_old
                            W_arr[k * n + i, col] = w_old
                    if found_improvement:
                        break
                if found_improvement:
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            # Try random perturbations as a fallback
            for _ in range(20):
                mat_choice = np.random.choice(3)
                if mat_choice == 0:
                    idx = np.random.randint(n * m)
                    col = np.random.randint(rank)
                    old = U_arr[idx, col]
                    delta = np.random.choice([-1, 1]) * np.random.choice([0.5, 1])
                    U_arr[idx, col] = old + delta
                elif mat_choice == 1:
                    idx = np.random.randint(m * p)
                    col = np.random.randint(rank)
                    old = V_arr[idx, col]
                    delta = np.random.choice([-1, 1]) * np.random.choice([0.5, 1])
                    V_arr[idx, col] = old + delta
                else:
                    idx = np.random.randint(p * n)
                    col = np.random.randint(rank)
                    old = W_arr[idx, col]
                    delta = np.random.choice([-1, 1]) * np.random.choice([0.5, 1])
                    W_arr[idx, col] = old + delta
                
                new_residual = _compute_residual(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p)
                new_max = np.max(np.abs(new_residual))
                old_max = np.max(np.abs(residual))
                
                if new_max < old_max - 1e-15:
                    found_improvement = True
                    break
                else:
                    if mat_choice == 0:
                        U_arr[idx, col] = old
                    elif mat_choice == 1:
                        V_arr[idx, col] = old
                    else:
                        W_arr[idx, col] = old
            
            if not found_improvement:
                break
    
    # Round to nearest half-integer
    U_arr = np.round(U_arr * 2) / 2
    V_arr = np.round(V_arr * 2) / 2
    W_arr = np.round(W_arr * 2) / 2
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

def _structured_initialization(n, m, p, rank, seed=None):
    """
    Create a structured initialization based on the tensor structure.
    """
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for col in range(rank):
        n_i = np.random.randint(1, 3)
        n_j = np.random.randint(1, 3)
        n_k = np.random.randint(1, 3)
        
        sel_i = np.random.choice(n, size=n_i, replace=False)
        sel_j = np.random.choice(m, size=n_j, replace=False)
        sel_k = np.random.choice(p, size=n_k, replace=False)
        
        for i in sel_i:
            for j in sel_j:
                U[i * m + j, col] = np.random.choice([1, -1])
        
        for j in sel_j:
            for k in sel_k:
                V[j * p + k, col] = np.random.choice([1, -1])
        
        for k in sel_k:
            for i in sel_i:
                W[k * n + i, col] = np.random.choice([1, -1])
    
    return U, V, W

def _try_rank_r_decomposition(n, m, p, r, time_budget=10.0):
    """Try to find a rank-r decomposition using structured search + local polish."""
    start_time = time.time()
    
    best_solution = None
    best_residual = float('inf')
    
    n_attempts = 300
    for attempt in range(n_attempts):
        if time.time() - start_time > time_budget:
            break
        
        seed = 42 + attempt
        U, V, W = _structured_initialization(n, m, p, r, seed=seed)
        
        residual = _compute_residual(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        max_res = np.max(np.abs(residual))
        
        if max_res < best_residual:
            best_residual = max_res
            best_solution = (U.copy(), V.copy(), W.copy())
        
        if max_res < 5.0:
            polished_U, polished_V, polished_W = _local_polish(
                U.tolist(), V.tolist(), W.tolist(), n, m, p, max_iterations=100
            )
            residual = _compute_residual(polished_U, polished_V, polished_W, n, m, p)
            max_res = np.max(np.abs(residual))
            
            if max_res < best_residual:
                best_residual = max_res
                best_solution = (
                    np.array(polished_U), 
                    np.array(polished_V), 
                    np.array(polished_W)
                )
            
            if max_res < 1e-9:
                return polished_U, polished_V, polished_W
    
    if best_solution is not None and best_residual < 10.0:
        U, V, W = best_solution
        polished_U, polished_V, polished_W = _local_polish(
            U.tolist(), V.tolist(), W.tolist(), n, m, p, max_iterations=500
        )
        residual = _compute_residual(polished_U, polished_V, polished_W, n, m, p)
        max_res = np.max(np.abs(residual))
        
        if max_res < 1e-9:
            return polished_U, polished_V, polished_W
    
    return None

def construct():
    """
    Construct the decomposition for 3x3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    best_U, best_V, best_W = _schoolbook_3x3x8()
    best_rank = 72
    
    for target_rank in range(55, 72):
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        
        time_budget = min(remaining_time * 0.25, 15.0)
        
        result = _try_rank_r_decomposition(n, m, p, target_rank, time_budget=time_budget)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
