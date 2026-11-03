"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor
    T_target = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Lower bound for rank: max(n*m, m*p, n*p) is not a tight lower bound for tensor rank,
    # but a good heuristic start is often lower. Let's start from a small number.
    # For 3x4x5, rank is likely < 30.
    # We will try ranks from 10 to 60.
    
    for rank in range(10, 61):
        # Try to find a real decomposition
        solution = _find_real_decomposition(n, m, p, rank, T_target)
        if solution is not None:
            U_real, V_real, W_real = solution
            
            # Try to convert to integer or half-integer
            # First, try rounding to integers
            U_int = np.round(U_real)
            V_int = np.round(V_real)
            W_int = np.round(W_real)
            
            if _is_exact_decomposition(U_int, V_int, W_int, n, m, p, T_target):
                return _to_dict(U_int, V_int, W_int)
            
            # Try rounding to half-integers (multiply by 2, round, divide by 2)
            U_half = np.round(U_real * 2) / 2.0
            V_half = np.round(V_real * 2) / 2.0
            W_half = np.round(W_real * 2) / 2.0
            
            if _is_exact_decomposition(U_half, V_half, W_half, n, m, p, T_target):
                return _to_dict(U_half, V_half, W_half)
            
            # If real solution exists but not exact integer, we might need to search nearby integers
            # Or increase rank. For now, if real solution found, it's a good candidate for improvement,
            # but we need exactness.
            
            # If we found a real solution, it suggests rank is possible.
            # We can do a local integer search around this solution.
            int_sol = _local_integer_search(U_real, V_real, W_real, n, m, p, T_target)
            if int_sol is not None:
                return _to_dict(int_sol[0], int_sol[1], int_sol[2])

    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _find_real_decomposition(n, m, p, rank, T_target):
    """Use least_squares to find a real-valued rank decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    total_vars = nm * rank + mp * rank + pn * rank
    
    def residuals(params):
        U_flat = params[:nm*rank]
        V_flat = params[nm*rank:nm*rank+mp*rank]
        W_flat = params[nm*rank+mp*rank:]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        # Compute tensor: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # This is equivalent to flattening the sum of outer products.
        # Efficient way: 
        # T_flat = sum_r (U[:,r] * V[:,r]) outer W[:,r] flattened?
        # Let's use einsum or manual loop for clarity in residual calculation.
        T_computed = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_computed += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        
        return (T_computed - T_target).flatten()
    
    # Initialize with small random values
    np.random.seed(42) # For reproducibility in optimization
    U0 = np.random.randn(nm, rank) * 0.1
    V0 = np.random.randn(mp, rank) * 0.1
    W0 = np.random.randn(pn, rank) * 0.1
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=2000)
        if result.success or result.cost < 1e-8:
            U = result.x[:nm*rank].reshape(nm, rank)
            V = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = result.x[nm*rank+mp*rank:].reshape(pn, rank)
            return U, V, W
    except Exception:
        pass
    return None

def _is_exact_decomposition(U, V, W, n, m, p, T_target):
    """Check if U, V, W form an exact decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    T_computed = np.zeros((nm, mp, pn))
    for r in range(rank):
        T_computed += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    # Check if T_computed is close to T_target
    # Since entries are integers/half-integers, exact match is possible
    if not np.allclose(T_computed, T_target, atol=1e-9):
        return False
    
    # Verify entries are integers or half-integers
    def check_entries(M):
        for val in M:
            if not (abs(val - round(val)) < 1e-9 or abs(val*2 - round(val*2)) < 1e-9):
                return False
        return True
        
    return check_entries(U) and check_entries(V) and check_entries(W)

def _local_integer_search(U_real, V_real, W_real, n, m, p, T_target):
    """Search for integer solution near the real solution."""
    nm, mp, pn = n*m, m*p, p*n
    rank = U_real.shape[1]
    
    # Round to nearest integer
    U_base = np.round(U_real).astype(int)
    V_base = np.round(V_real).astype(int)
    W_base = np.round(W_real).astype(int)
    
    # If the base is already exact, return it
    if _is_exact_decomposition(U_base.astype(float), V_base.astype(float), W_base.astype(float), n, m, p, T_target):
        return U_base.astype(float), V_base.astype(float), W_base.astype(float)
    
    # Try small perturbations? This is expensive. 
    # For now, if real solution exists, it's highly likely the integer rounding works 
    # or we need a slightly higher rank.
    return None

def _to_dict(U, V, W):
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n, m, p):
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
