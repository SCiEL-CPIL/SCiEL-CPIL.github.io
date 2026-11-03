"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 5, 5]

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
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

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def _schoolbook(n, m, p):
    """Return schoolbook decomposition as numpy arrays."""
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

def _tensor_error(U, V, W, n, m, p):
    """Compute Frobenius norm of error between decomposition and true tensor."""
    # True tensor: T[i,j,k] = 1 if we're doing standard matmul
    # Decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape to (n, m, p, rank) and contract
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    
    # For each (i,j,k), the decomposition gives sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # We want this to equal 1 for all (i,j,k)
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
                error += (val - 1.0) ** 2
    return np.sqrt(error)

def _try_reduce_rank(U, V, W, n, m, p, max_time=100.0):
    """Try to find a lower-rank decomposition via local search."""
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    best_err = _tensor_error(best_U, best_V, best_W, n, m, p)
    
    # First, try to identify and remove redundant columns
    # A column is redundant if removing it and adjusting others still gives zero error
    
    # Use gradient-based local search to try to find a valid decomposition
    # with fewer terms
    
    # Strategy: try all possible subsets of columns of size r-1, r-2, etc.
    # For each subset, check if we can solve for the remaining entries
    
    current_rank = best_rank
    U_curr, V_curr, W_curr = best_U.copy(), best_V.copy(), best_W.copy()
    
    # Try to reduce rank by 1 at a time
    for target_rank in range(current_rank - 1, max(20, current_rank - 20), -1):
        if time.time() - start_time > max_time - 30:
            break
        
        # Try random column removal and re-solve
        for attempt in range(50):
            if time.time() - start_time > max_time - 30:
                break
            
            # Select a random subset of columns
            if target_rank >= current_rank:
                continue
                
            col_indices = list(range(current_rank))
            np.random.shuffle(col_indices)
            keep = col_indices[:target_rank]
            keep.sort()
            
            U_try = U_curr[:, keep].copy()
            V_try = V_curr[:, keep].copy()
            W_try = W_curr[:, keep].copy()
            
            err = _tensor_error(U_try, V_try, W_try, n, m, p)
            if err < 1e-8:
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                best_err = err
                current_rank = target_rank
                U_curr, V_curr, W_curr = U_try.copy(), V_try.copy(), W_try.copy()
                break
    
    # If we found a valid lower-rank decomposition, try to polish it
    if best_err < 1e-8 and best_rank < U.shape[1]:
        # Try to find exact rational values
        pass
    
    return best_U, best_V, best_W, best_rank

def _to_fraction_list(matrix):
    """Convert numpy array to list of lists of Fractions for exactness."""
    result = []
    for row in matrix:
        frac_row = []
        for val in row:
            if abs(val) < 1e-10:
                frac_row.append(Fraction(0))
            elif abs(val - round(val)) < 1e-10:
                frac_row.append(Fraction(int(round(val))))
            else:
                # Try to find a simple fraction
                frac = Fraction(val).limit_denominator(1000)
                frac_row.append(frac)
        result.append(frac_row)
    return result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Get schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    schoolbook_rank = U.shape[1]
    
    # Try to reduce rank
    U_best, V_best, W_best, best_rank = _try_reduce_rank(U, V, W, n, m, p, max_time=120.0)
    
    # If we found a better decomposition, use it; otherwise use schoolbook
    if best_rank < schoolbook_rank and time.time() < deadline:
        U_final, V_final, W_final = U_best, V_best, W_best
    else:
        U_final, V_final, W_final = U, V, W
    
    # Convert to list of lists with integer/half-integer entries
    # The problem says entries should be integer or half-integer
    # So multiply by 2 to get integers, then divide by 2 in the output
    # Actually, let's just ensure entries are integers or multiples of 0.5
    
    def clean_matrix(mat):
        """Clean matrix entries to be integer or half-integer."""
        cleaned = []
        for row in mat:
            clean_row = []
            for val in row:
                if abs(val) < 1e-10:
                    clean_row.append(0)
                else:
                    # Round to nearest half-integer
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-6:
                        clean_row.append(rounded)
                    else:
                        # Keep as is if it's close enough
                        clean_row.append(rounded)
            cleaned.append(clean_row)
        return cleaned
    
    U_out = clean_matrix(U_final)
    V_out = clean_matrix(V_final)
    W_out = clean_matrix(W_final)
    
    # Verify the decomposition is still valid after cleaning
    # If not, fall back to schoolbook
    if time.time() < deadline:
        U_check = np.array(U_out, dtype=float)
        V_check = np.array(V_out, dtype=float)
        W_check = np.array(W_out, dtype=float)
        err = _tensor_error(U_check, V_check, W_check, n, m, p)
        if err > 1e-6:
            # Fall back to schoolbook
            U_out = clean_matrix(U)
            V_out = clean_matrix(V)
            W_out = clean_matrix(W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
