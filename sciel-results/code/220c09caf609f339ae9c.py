import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _compute_error_fast(U, V, W, n, m, p):
    """Fast vectorized error computation."""
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kin->ijk', U_r, V_r, W_r, optimize=True)
    return np.sum((T - 1.0)**2)

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast numerical verification using numpy. Returns True if likely correct."""
    error = _compute_error_fast(U, V, W, n, m, p)
    return error < tol

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Exact rational verification. Only call on candidates that pass fast check."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(str(U[i*m+j][c]))
                    v_val = Fraction(str(V[j*p+k][c]))
                    w_val = Fraction(str(W[k*n+i][c]))
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _local_search(n, m, p, target_rank, time_limit):
    """
    Perform local search to find a decomposition of target_rank.
    Uses fast vectorized error.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize with small random integers/half-integers
    # Snap to 0.5 grid
    U = np.random.randint(-2, 3, (nm, target_rank)).astype(np.float64) * 0.5
    V = np.random.randint(-2, 3, (mp, target_rank)).astype(np.float64) * 0.5
    W = np.random.randint(-2, 3, (pn, target_rank)).astype(np.float64) * 0.5
    
    current_error = _compute_error_fast(U, V, W, n, m, p)
    
    start_time = time.time()
    no_improvement_count = 0
    max_no_improvement = 1000 # Allow more iterations due to speed
    
    while time.time() - start_time < time_limit:
        # Choose a random matrix and a random entry to perturb
        mat_choice = random.randint(0, 2)
        if mat_choice == 0:
            mat = U
            rows = nm
        elif mat_choice == 1:
            mat = V
            rows = mp
        else:
            mat = W
            rows = pn
        
        row = random.randint(0, rows - 1)
        col = random.randint(0, target_rank - 1)
        
        old_val = mat[row, col]
        
        # Discrete moves on the 0.5 grid
        # Deltas: -1, -0.5, 0.5, 1
        # Or just pick a new random value in [-2, 2] with step 0.5
        # Let's try random replacement first, then local moves
        if random.random() < 0.5:
            new_val = random.choice([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        else:
            new_val = old_val + random.choice([-1.0, -0.5, 0.5, 1.0])
        
        mat[row, col] = new_val
        new_error = _compute_error_fast(U, V, W, n, m, p)
        
        if new_error < current_error:
            current_error = new_error
            no_improvement_count = 0
            if current_error < 1e-6:
                # Snap to nearest half-integer and verify
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                if _verify_decomposition_fast(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                    U_list = U_snapped.tolist()
                    V_list = V_snapped.tolist()
                    W_list = W_snapped.tolist()
                    if _verify_decomposition_exact(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list
        else:
            # Revert
            mat[row, col] = old_val
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                break
        
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Try ranks from lower bound to schoolbook
    # Lower bound is max(nm, mp, pn) = 21
    # But we can start higher if 21-30 is unlikely
    # Let's try 21 to 41
    
    for target_rank in range(21, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        remaining_time = deadline - 10 - time.time()
        if remaining_time <= 0:
            break
        
        # Allocate time based on rank
        # Give more time to lower ranks
        time_per_rank = min(remaining_time * 0.2, 5.0) # 5s per rank max
        
        result = _local_search(n, m, p, target_rank, time_limit=time_per_rank)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return _schoolbook()
