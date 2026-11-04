"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists with the standard rank n*m*p decomposition."""
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k), else 0."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Efficient vectorized check using einsum
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need to check that T[i,j,k] == 1 for all i,j,k
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # For each (i,j,k), compute sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a bit tricky with einsum because the indices are interleaved.
    # Let's just loop over i,j,k since n*m*p = 40 is small.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p, tol=1e-9):
    """Faster numpy-based verification."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # Build the full tensor T of shape (n, m, p)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can compute this efficiently:
    # For fixed i, j: U3[i,j,:] is a vector of length rank
    # For fixed j, k: V3[j,k,:] is a vector of length rank
    # For fixed k, i: W3[k,i,:] is a vector of length rank
    
    # Let's compute T[i,j,k] for all i,j,k
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
    
    return np.allclose(T, 1.0, atol=tol)

def _rank_of(U, V, W):
    return len(U[0])

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    fx = Fraction(x).limit_denominator(1000)
    num, den = fx.numerator, fx.denominator
    if den == 1:
        return True
    if den == 2:
        return True
    return False

def _all_half_integer(U, V, W, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = _rank_of(U, V, W)
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by removing columns that are linearly dependent or zero in the tensor sense.
    def try_remove_column(U, V, W, col_to_remove):
        rank = len(U[0])
        if rank <= 1:
            return None, None, None, False
        
        # Remove column
        new_rank = rank - 1
        new_U = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
        new_V = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
        new_W = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
        
        if _verify_numpy(new_U, new_V, new_W, n, m, p):
            return new_U, new_V, new_W, True
        return None, None, None, False
    
    # Greedy column removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        rank = len(U[0])
        for c in range(rank):
            if time.time() > deadline:
                break
            new_U, new_V, new_W, ok = try_remove_column(U, V, W, c)
            if ok:
                U, V, W = new_U, new_V, new_W
                best_rank = len(U[0])
                best_U, best_V, best_W = U, V, W
                changed = True
                break  # Restart loop since indices changed
    
    # Try to find a lower rank decomposition using optimization
    # For matrix multiplication, we can try to find a decomposition with rank r < n*m*p
    # by solving a system of equations.
    
    # The tensor decomposition condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    # For r = 20 (the lower bound), we have 20*(8 + 20 + 10) = 20*38 = 760 unknowns
    # and 40 equations. This is underdetermined, so solutions should exist.
    
    # Let's try to find a solution with rank r for r from the lower bound up to the current best.
    # We'll use a simple approach: initialize U, V, W with random values and use gradient descent
    # to minimize the error.
    
    import random
    random.seed(42)
    
    def try_ranked_decomposition(r, max_iter=1000, lr=0.01, seed=42):
        """Try to find a decomposition of rank r using gradient descent."""
        if time.time() > deadline:
            return None, None, None
        
        np.random.seed(seed)
        
        # Initialize U, V, W with small random values
        U_arr = np.random.randn(n * m, r) * 0.1
        V_arr = np.random.randn(m * p, r) * 0.1
        W_arr = np.random.randn(p * n, r) * 0.1
        
        # Compute the error: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1
        # We want to minimize the sum of squares of these errors.
        
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None, None, None
            
            # Compute gradients
            dU = np.zeros_like(U_arr)
            dV = np.zeros_like(V_arr)
            dW = np.zeros_like(W_arr)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
                        # dT/dU3[i,j,c] = V3[j,k,c] * W3[k,i,c]
                        # dT/dV3[j,k,c] = U3[i,j,c] * W3[k,i,c]
                        # dT/dW3[k,i,c] = U3[i,j,c] * V3[j,k,c]
                        # Error = T[i,j,k] - 1
                        # dError/dU3[i,j,c] = V3[j,k,c] * W3[k,i,c]
                        # etc.
                        
                        prod_UVW = U3[i, j, :] * V3[j, k, :] * W3[k, i, :]
                        T_val = np.sum(prod_UVW)
                        err = T_val - 1.0
                        
                        dU3_i_j = V3[j, k, :] * W3[k, i, :] * err
                        dV3_j_k = U3[i, j, :] * W3[k, i, :] * err
                        dW3_k_i = U3[i, j, :] * V3[j, k, :] * err
                        
                        dU[i * m + j, :] += dU3_i_j
                        dV[j * p + k, :] += dV3_j_k
                        dW[k * n + i, :] += dW3_k_i
            
            # Gradient descent step
            U_arr -= lr * dU
            V_arr -= lr * dV
            W_arr -= lr * dW
            
            # Check convergence
            if it % 50 == 0:
                U3 = U_arr.reshape(n, m, r)
                V3 = V_arr.reshape(m, p, r)
                W3 = W_arr.reshape(p, n, r)
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                            max_err = max(max_err, abs(T_val - 1.0))
                if max_err < 1e-6:
                    break
        
        # Final check
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                    max_err = max(max_err, abs(T_val - 1.0))
        
        if max_err < 1e-6:
            # Convert to lists
            U_list = U_arr.tolist()
            V_list = V_arr.tolist()
            W_list = W_arr.tolist()
            return U_list, V_list, W_list
        
        return None, None, None
    
    # Try ranks from the lower bound up to the current best
    lower_bound = max(n * m, m * p, p * n)
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few different seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            U_new, V_new, W_new = try_ranked_decomposition(r, max_iter=500, lr=0.01, seed=seed)
            
            if U_new is not None:
                # Verify the decomposition
                if _verify_numpy(U_new, V_new, W_new, n, m, p):
                    best_rank = r
                    best_U, best_V, best_W = U_new, V_new, W_new
                    break
        
        if best_rank == r:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
