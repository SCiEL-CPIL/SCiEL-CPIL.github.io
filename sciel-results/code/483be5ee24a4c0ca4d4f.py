"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0
                for c in range(rank):
                    lhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if lhs != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    for i in range(n):
        for j in range(m):
            # Row i*m+j of U, row j*p+k of V for all k
            u_row = U[i * m + j, :]  # (rank,)
            # V rows: j*p to j*p+p-1
            v_rows = V[j * p : (j + 1) * p, :]  # (p, rank)
            # W rows: k*n + i for k in range(p)
            w_vals = np.array([W[k * n + i, :] for k in range(p)])  # (p, rank)
            # For each k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            for k in range(p):
                s = np.sum(u_row * v_rows[k, :] * w_vals[k, :])
                if s != 1:
                    return False
    return True

def _round_to_halfint(arr, tol=1e-6):
    """Round array entries to nearest half-integer or integer if within tol."""
    result = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        val = arr[idx]
        # Try rounding to nearest 0.5
        rounded = np.round(val * 2) / 2.0
        if abs(val - rounded) < tol:
            result[idx] = rounded
        else:
            return None
    return result

def _to_fraction_list(arr):
    """Convert numpy array to list of Fractions."""
    result = []
    for idx in np.ndindex(arr.shape):
        val = arr[idx]
        # Check if it's a half-integer
        if abs(val * 2 - round(val * 2)) < 1e-9:
            result.append(Fraction(int(round(val * 2)), 2))
        else:
            result.append(Fraction(val).limit_denominator(1000000))
    return result

def construct():
    """Find rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 40
    
    # Start with schoolbook as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # The minimum possible rank for matrix multiplication is at least max(n*m, m*p, p*n)
    # For 2x4x5: max(8, 20, 10) = 20
    # Known lower bounds: rank >= n*m*p / max(n,m,p) ... but we just search
    
    # Search ranks from low to high
    min_possible = max(n * m, m * p, p * n)  # 20
    max_search_rank = min(schoolbook_rank - 1, 35)  # Try up to 35
    
    np.random.seed(42)
    
    for target_rank in range(min_possible, max_search_rank + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Multiple random restarts for each rank
        max_restarts = 50 if target_rank <= 25 else 20
        if time.time() > deadline - 10.0:
            max_restarts = min(max_restarts, 5)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 3.0:
                break
                
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Add some structure: ensure each column has non-trivial entries
            for c in range(target_rank):
                U[:, c] *= (1 + np.random.rand(n * m) * 0.5)
                V[:, c] *= (1 + np.random.rand(m * p) * 0.5)
                W[:, c] *= (1 + np.random.rand(p * n) * 0.5)
            
            # Iterative refinement using alternating least squares
            # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
            # This is a trilinear form. We can fix two matrices and solve for the third.
            
            converged = False
            for iteration in range(200):
                if time.time() > deadline - 2.0:
                    break
                    
                # Fix V and W, solve for U
                # For each (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                # This gives p equations per (i,j) pair, but we only need one equation per (i,j)
                # Actually, the constraint is: for ALL i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # Fix V,W: for each (i,j), we need U[i*m+j,:] to satisfy:
                #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for k=0,...,p-1
                # This is a system of p equations in target_rank unknowns (the entries of U[i*m+j,:])
                
                U_new = np.zeros((n * m, target_rank))
                for i in range(n):
                    for j in range(m):
                        # Build system: A * x = b where x = U[i*m+j, :]
                        A = np.zeros((p, target_rank))
                        b = np.ones(p)
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve least squares
                        try:
                            x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                            U_new[i * m + j, :] = x
                        except:
                            U_new[i * m + j, :] = U[i * m + j, :]
                
                # Fix U and W, solve for V
                V_new = np.zeros((m * p, target_rank))
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        b = np.ones(n)
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                            V_new[j * p + k, :] = x
                        except:
                            V_new[j * p + k, :] = V[j * p + k, :]
                
                # Fix U and V, solve for W
                W_new = np.zeros((p * n, target_rank))
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        b = np.ones(m)
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                            W_new[k * n + i, :] = x
                        except:
                            W_new[k * n + i, :] = W[k * n + i, :]
                
                # Check convergence
                U_diff = np.max(np.abs(U_new - U))
                V_diff = np.max(np.abs(V_new - V))
                W_diff = np.max(np.abs(W_new - W))
                
                U, V, W = U_new, V_new, W_new
                
                if max(U_diff, V_diff, W_diff) < 1e-8:
                    converged = True
                    break
            
            # Check if we have an exact solution
            if _verify_fast(U, V, W, n, m, p):
                # Try to round to half-integers
                U_rounded = _round_to_halfint(U, tol=1e-4)
                V_rounded = _round_to_halfint(V, tol=1e-4)
                W_rounded = _round_to_halfint(W, tol=1e-4)
                
                if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                    if _verify_fast(U_rounded, V_rounded, W_rounded, n, m, p):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        break  # Found solution at this rank, move to next
                    else:
                        # Use unrounded but verify
                        if _verify_fast(U, V, W, n, m, p):
                            best_rank = target_rank
                            best_U, best_V, best_W = U, V, W
                            break
                else:
                    if _verify_fast(U, V, W, n, m, p):
                        best_rank = target_rank
                        best_U, best_V, best_W = U, V, W
                        break
        
        # If we found a solution at this rank, we can stop
        if best_rank == target_rank:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Try to convert to fractions for cleanliness
    try:
        U_frac = []
        for row in U_list:
            U_frac.append([Fraction(str(x)).limit_denominator(1000000) for x in row])
        V_frac = []
        for row in V_list:
            V_frac.append([Fraction(str(x)).limit_denominator(1000000) for x in row])
        W_frac = []
        for row in W_list:
            W_frac.append([Fraction(str(x)).limit_denominator(1000000) for x in row])
        
        # Verify with fractions
        if _verify(U_frac, V_frac, W_frac, n, m, p):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    except:
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
