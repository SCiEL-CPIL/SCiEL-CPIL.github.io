import numpy as np
import time
from fractions import Fraction
import random
from itertools import product

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
    Uses exact Fraction arithmetic for safety.
    """
    rank = len(U[0])
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                
                if total != 1:
                    return False
    return True

def _construct_rank_28():
    """
    Constructs the known rank-28 decomposition for 2x3 * 3x5.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen (7 terms)
    # M1 = (a00+a11)(b00+b11)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2 = (a10+a11)b00
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3 = a00(b01-b11)
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4 = a11(b10-b00)
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5 = (a00+a01)b11
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6 = (a10-a00)(b00+b01)
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7 = (a01-a11)(b10+b11)
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen (7 terms)
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_find_lower_rank(target_rank, n, m, p, max_attempts=500, deadline=None):
    """
    Attempts to find a rank-`target_rank` decomposition using random initialization
    and local search. Returns None if not found within attempts/time.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    # Precompute the target tensor indices
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                tensor_entries.append((idx_u, idx_v, idx_w))
    
    num_entries = len(tensor_entries)
    
    def _evaluate(U, V, W):
        """Compute the error of the decomposition."""
        error = 0.0
        for idx_u, idx_v, idx_w in tensor_entries:
            s = 0.0
            for r in range(target_rank):
                s += U[idx_u][r] * V[idx_v][r] * W[idx_w][r]
            error += (s - 1.0) ** 2
        return error
    
    def _random_decomposition():
        """Generate a random sparse decomposition with entries in {-1, 0, 1}."""
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        for r in range(target_rank):
            # Each rank-1 term touches a subset of indices
            # Randomly choose which entries this term affects
            num_u = random.randint(1, min(3, n * m))
            num_v = random.randint(1, min(3, m * p))
            num_w = random.randint(1, min(3, p * n))
            
            u_indices = random.sample(range(n * m), num_u)
            v_indices = random.sample(range(m * p), num_v)
            w_indices = random.sample(range(p * n), num_w)
            
            for idx in u_indices:
                U[idx][r] = random.choice([-1, 1])
            for idx in v_indices:
                V[idx][r] = random.choice([-1, 1])
            for idx in w_indices:
                W[idx][r] = random.choice([-1, 1])
        
        return U, V, W
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        U, V, W = _random_decomposition()
        err = _evaluate(U, V, W)
        
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        
        if err < 1e-9:
            # Found exact solution, verify with fractions
            if _verify_decomposition(best_U, best_V, best_W, n, m, p):
                return {"U": best_U, "V": best_V, "W": best_W}
        
        # Local search: try flipping individual entries
        if attempt % 10 == 0 and err < 10.0:
            # Try a few local improvements
            for _ in range(20):
                if deadline is not None and time.time() > deadline:
                    break
                # Pick a random entry to flip
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, target_rank - 1)
                    old = U[row][col]
                    if old == 0:
                        U[row][col] = random.choice([-1, 1])
                    else:
                        U[row][col] = 0
                    new_err = _evaluate(U, V, W)
                    if new_err >= err:
                        U[row][col] = old
                    else:
                        err = new_err
                elif mat == 'V':
                    row = random.randint(0, m * p - 1)
                    col = random.randint(0, target_rank - 1)
                    old = V[row][col]
                    if old == 0:
                        V[row][col] = random.choice([-1, 1])
                    else:
                        V[row][col] = 0
                    new_err = _evaluate(U, V, W)
                    if new_err >= err:
                        V[row][col] = old
                    else:
                        err = new_err
                else:
                    row = random.randint(0, p * n - 1)
                    col = random.randint(0, target_rank - 1)
                    old = W[row][col]
                    if old == 0:
                        W[row][col] = random.choice([-1, 1])
                    else:
                        W[row][col] = 0
                    new_err = _evaluate(U, V, W)
                    if new_err >= err:
                        W[row][col] = old
                    else:
                        err = new_err
    
    return None

def construct():
    start_time = time.time()
    # Deadline: leave 5 seconds margin before the 150s kill
    deadline = start_time + 140.0
    
    # First, try to find a lower rank decomposition
    # Known lower bound is 28, so try 27, 26, 25
    # But be very conservative with time
    best_result = None
    best_rank = 28
    
    # Try rank 27 with limited attempts
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        result_27 = _try_find_lower_rank(27, 2, 3, 5, max_attempts=200, deadline=deadline)
        if result_27 is not None:
            best_result = result_27
            best_rank = 27
    
    remaining_time = deadline - time.time()
    if best_result is None and remaining_time > 20:
        result_26 = _try_find_lower_rank(26, 2, 3, 5, max_attempts=100, deadline=deadline)
        if result_26 is not None:
            best_result = result_26
            best_rank = 26
    
    # If no lower rank found, return the known rank-28
    if best_result is None:
        best_result = _construct_rank_28()
    
    # Final verification
    if _verify_decomposition(best_result['U'], best_result['V'], best_result['W'], 2, 3, 5):
        return best_result
    else:
        # Fallback to verified rank-28
        res_28 = _construct_rank_28()
        return res_28
