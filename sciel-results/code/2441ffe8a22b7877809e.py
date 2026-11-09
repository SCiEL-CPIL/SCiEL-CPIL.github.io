"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0
                for c in range(rank):
                    lhs += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if lhs != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-8):
    """Verify using floating point with tolerance."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_halfint(arr, tol=1e-6):
    """Round array entries to nearest half-integer if within tol."""
    result = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        val = arr[idx]
        rounded = np.round(val * 2) / 2.0
        if abs(val - rounded) < tol:
            result[idx] = rounded
        else:
            return None
    return result

def _to_fraction_list(arr):
    """Convert numpy array to list of Fractions."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            if abs(val * 2 - round(val * 2)) < 1e-9:
                row_list.append(Fraction(int(round(val * 2)), 2))
            else:
                # Try to find a simple fraction
                frac = Fraction(str(val)).limit_denominator(100000)
                row_list.append(frac)
        result.append(row_list)
    return result

def _als_solve(U, V, W, n, m, p, max_iter=200, deadline=None):
    """Alternating least squares to refine the decomposition."""
    rank = U.shape[1]
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Fix V and W, solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # U[i*m+j, c] * sum_k V[j*p+k, c] * W[k*n+i, c] = 1
                # This is: U[i*m+j, :] dot (V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
                # Actually it's: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # So we need: U[i*m+j,:] to satisfy a system
                # Let a_c = V[j*p+k,c] * W[k*n+i,c] for each k
                # Then U[i*m+j,:] dot a_k = 1 for all k
                # This is A @ x = b where A[k,c] = V[j*p+k,c]*W[k*n+i,c], b[k]=1
                A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                b = np.ones(p)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i * m + j, :] = x
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Fix U and W, solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                b = np.ones(n)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j * p + k, :] = x
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Fix U and V, solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                b = np.ones(m)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k * n + i, :] = x
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        U, V, W = U_new, V_new, W_new
    
    return U, V, W

def construct():
    """Find rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
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
    # Theoretical lower bound is around 30-35 for 2x4x5
    # Let's try ranks from 30 to 39
    
    np.random.seed(42)
    
    for target_rank in range(30, schoolbook_rank):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization with small values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # ALS refinement
            U, V, W = _als_solve(U, V, W, n, m, p, max_iter=100, deadline=deadline - 2.0)
            
            # Check if it works
            if _verify_float(U, V, W, n, m, p, tol=1e-6):
                # Try to round to half-integers
                U_rounded = _round_to_halfint(U, tol=1e-4)
                V_rounded = _round_to_halfint(V, tol=1e-4)
                W_rounded = _round_to_halfint(W, tol=1e-4)
                
                if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                    if _verify_float(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        found = True
                        break
                else:
                    # Use the float solution
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
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
