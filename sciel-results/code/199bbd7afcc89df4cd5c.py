"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import random

def _verify_decomposition(U, V, W, n, m, p, tol=1e-12):
    """Verify that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = delta[i,k] for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                expected = 1.0 if (i == k) else 0.0
                if abs(val - expected) > tol:
                    return False
    return True

def _try_round_to_half_integers(U, V, W, max_trials=100):
    """Try to round entries to half-integers while preserving exactness."""
    def round_half(x):
        return round(2*x) / 2.0
    
    def is_half_integer(x, tol=1e-9):
        r = round(2*x)
        return abs(x - r/2.0) < tol
    
    def check_half_int(mat):
        return all(is_half_integer(v) for row in mat for v in row)
    
    # First check if current values are already half-integers
    if check_half_int(U) and check_half_int(V) and check_half_int(W):
        return [int(round(2*v))//2 for row in U for v in row], \
               [int(round(2*v))//2 for row in V for v in row], \
               [int(round(2*v))//2 for row in W for v in row]
    
    # Try small perturbations to find half-integer solutions
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    for trial in range(max_trials):
        U_try = [list(row) for row in U]
        V_try = [list(row) for row in V]
        W_try = [list(row) for row in W]
        
        # Round each entry to nearest half-integer
        for i in range(len(U_try)):
            for j in range(rank):
                U_try[i][j] = round_half(U_try[i][j])
        for i in range(len(V_try)):
            for j in range(rank):
                V_try[i][j] = round_half(V_try[i][j])
        for i in range(len(W_try)):
            for j in range(rank):
                W_try[i][j] = round_half(W_try[i][j])
        
        if _verify_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-9):
            return U_try, V_try, W_try
    
    return None

def _random_rank_reduction(U, V, W, n, m, p, num_trials=2000, seed=42):
    """Try to reduce rank by finding linear dependencies."""
    rng = random.Random(seed)
    rank = len(U[0])
    
    # Work with float arrays for speed
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    for trial in range(num_trials):
        if trial % 100 == 0 and time.time() - start_time > 100:
            break
        
        # Try to eliminate one random term by expressing it as combination of others
        target = rng.randint(0, best_rank - 1)
        
        # Get all other terms
        other_indices = [i for i in range(best_rank) if i != target]
        
        if len(other_indices) < 2:
            continue
        
        # Try to find coefficients alpha such that:
        # U_target = sum_alpha * U_other
        # V_target = sum_alpha * V_other  
        # W_target = sum_alpha * W_other
        # This requires that the 3-tensor term is a linear combination
        
        # Simpler approach: try random linear combinations
        # Create new U, V, W by combining two random terms
        i1, i2 = rng.sample(range(best_rank), 2)
        
        # Try: new_term = U[i1] + c*U[i2], etc.
        # Actually, let's try a different approach: Gaussian elimination on the tensor
        
        # For each pair, try to see if one can be expressed in terms of others
        # This is complex, so let's use a simpler heuristic:
        
        # Random linear combination test
        c = rng.choice([-1, 1, 0.5, -0.5, 2, -2])
        
        U_new = [list(row) for row in U_arr]
        V_new = [list(row) for row in V_arr]
        W_new = [list(row) for row in W_arr]
        
        # Add c*term_i2 to term_i1, then set term_i2 to zero
        for i in range(len(U_new)):
            U_new[i][i1] += c * U_new[i][i2]
        for i in range(len(V_new)):
            V_new[i][i1] += c * V_new[i][i2]
        for i in range(len(W_new)):
            W_new[i][i1] += c * W_new[i][i2]
        
        # Check if term_i2 is now all zeros (or close)
        if (np.allclose(U_new[:, i2], 0, atol=1e-10) and
            np.allclose(V_new[:, i2], 0, atol=1e-10) and
            np.allclose(W_new[:, i2], 0, atol=1e-10)):
            # We can remove this term!
            keep = [i for i in range(best_rank) if i != i2]
            U_reduced = [row[i] for i in keep for row in U_new]
            # Actually need to be careful with shape
            U_reduced = np.array(U_new)[:, keep]
            V_reduced = np.array(V_new)[:, keep]
            W_reduced = np.array(W_new)[:, keep]
            
            if _verify_decomposition(U_reduced.tolist(), V_reduced.tolist(), W_reduced.tolist(), n, m, p, tol=1e-9):
                best_U = U_reduced.tolist()
                best_V = V_reduced.tolist()
                best_W = W_reduced.tolist()
                best_rank = len(keep)
                U_arr = U_reduced
                V_arr = V_reduced
                W_arr = W_reduced
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition
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
    
    # Try rank reduction
    U_red, V_red, W_red, red_rank = _random_rank_reduction(U, V, W, n, m, p, num_trials=3000, seed=42)
    
    # Try to round to half-integers
    rounded = _try_round_to_half_integers(U_red, V_red, W_red, max_trials=50)
    if rounded is not None:
        U_out, V_out, W_out = rounded
    else:
        U_out, V_out, W_out = U_red, V_red, W_red
    
    # Verify final result
    if not _verify_decomposition(U_out, V_out, W_out, n, m, p, tol=1e-9):
        # Fallback to schoolbook
        U_out, V_out, W_out = U, V, W
    
    # Ensure all entries are half-integers (multiply by 2, round, divide by 2)
    def make_half_int(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                r = round(2 * v)
                new_row.append(r / 2.0)
            result.append(new_row)
        return result
    
    U_final = make_half_int(U_out)
    V_final = make_half_int(V_out)
    W_final = make_half_int(W_out)
    
    # Final verification
    if not _verify_decomposition(U_final, V_final, W_final, n, m, p, tol=1e-9):
        # If rounding broke it, use original
        U_final, V_final, W_final = U_out, V_out, W_out
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
