"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 if i,j,k are valid indices (they always are in 0..n-1, 0..m-1, 0..p-1)
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid i,j,k
    
    # Let's verify: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    # U: (n*m, rank) -> for each (i,j), we have a vector in R^rank
    # V: (m*p, rank) -> for each (j,k), we have a vector in R^rank  
    # W: (p*n, rank) -> for each (k,i), we have a vector in R^rank
    
    # The product tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k])
                # Wait, that's not right. The decomposition is:
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # This is a trilinear form, not a bilinear one.
                
                # Actually, for matrix multiplication:
                # (AB)[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor T has T[i,j,k] = 1 (since it's the identity in the middle)
                # And the decomposition is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                
                # Let me reconsider. The standard tensor decomposition for matmul:
                # We want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta (which is 1 for all i,j,k in matmul)
                
                # For 2x3 * 3x7:
                # n=2 (rows of A), m=3 (cols of A / rows of B), p=7 (cols of B)
                # A is 2x3, B is 3x7, C = AB is 2x7
                # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
                # Tensor T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,7)
                
                # Decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                
                # So for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                
                # This is a trilinear form. Let me compute it properly.
                u_vec = U_arr[i*m+j]
                v_vec = V_arr[j*p+k]
                w_vec = W_arr[k*n+i]
                val = np.sum(u_vec * v_vec * w_vec)
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    return max_err < tol

def _try_rank(rank, n, m, p, time_limit=30.0, n_restarts=5):
    """Try to find a rank decomposition using optimization."""
    start_time = time.time()
    
    # Variables: U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Total variables: (n*m + m*p + p*n) * rank
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    total_vars = (nm + mp + pn) * rank
    
    def objective(x):
        """Objective: sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2"""
        # Reshape x into U, V, W
        U_flat = x[:nm*rank]
        V_flat = x[nm*rank:nm*rank+mp*rank]
        W_flat = x[nm*rank+mp*rank:]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j]
                    v_val = V[j*p+k]
                    w_val = W[k*n+i]
                    prod = np.sum(u_val * v_val * w_val)
                    err += (prod - 1.0) ** 2
        
        return err
    
    # Try multiple random initializations
    best_err = float('inf')
    best_x = None
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with small values
        x0 = np.random.randn(total_vars) * 0.1
        
        # Use Nelder-Mead or BFGS
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if best_x is None or best_err > 1e-6:
        return None
    
    # Extract U, V, W
    nm_rank = nm * rank
    mp_rank = mp * rank
    
    U_flat = best_x[:nm_rank]
    V_flat = best_x[nm_rank:nm_rank+mp_rank]
    W_flat = best_x[nm_rank+mp_rank:]
    
    U = U_flat.reshape(nm, rank)
    V = V_flat.reshape(mp, rank)
    W = W_flat.reshape(pn, rank)
    
    # Convert to lists and check for near-integer values
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Check if values are close to integers or half-integers
    def is_close_to_rational(x, tol=1e-4):
        # Check if x is close to an integer or half-integer
        for denom in [1, 2]:
            for num in range(-10, 11):
                if abs(x - num/denom) < tol:
                    return num/denom
        return None
    
    # Try to snap to rational values
    U_snapped = []
    V_snapped = []
    W_snapped = []
    
    all_snapped = True
    for row in U_list:
        snapped_row = []
        for val in row:
            snapped = is_close_to_rational(val)
            if snapped is None:
                all_snapped = False
                break
            snapped_row.append(snapped)
        if not all_snapped:
            break
        U_snapped.append(snapped_row)
    
    if all_snapped:
        for row in V_list:
            snapped_row = []
            for val in row:
                snapped = is_close_to_rational(val)
                if snapped is None:
                    all_snapped = False
                    break
                snapped_row.append(snapped)
            if not all_snapped:
                break
            V_snapped.append(snapped_row)
    
    if all_snapped:
        for row in W_list:
            snapped_row = []
            for val in row:
                snapped = is_close_to_rational(val)
                if snapped is None:
                    all_snapped = False
                    break
                snapped_row.append(snapped)
            if not all_snapped:
                break
            W_snapped.append(snapped_row)
    
    if all_snapped and len(U_snapped) == nm and len(V_snapped) == mp and len(W_snapped) == pn:
        # Verify the snapped decomposition
        if _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p):
            return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, return schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Also, rank >= n*m*p / (some factor) - not straightforward
    
    # Try ranks from 21 upward
    for rank in range(21, best_rank):
        if time.time() > deadline:
            break
        
        # Time budget for each rank attempt
        time_budget = min(20.0, (deadline - time.time()) * 0.8)
        if time_budget < 5.0:
            break
        
        result = _try_rank(rank, n, m, p, time_limit=time_budget, n_restarts=3)
        
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid decomposition, this is minimal since we go upward
    
    return best_result

# EVOLVE-BLOCK-END
