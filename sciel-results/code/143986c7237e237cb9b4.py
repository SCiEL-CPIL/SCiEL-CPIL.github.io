"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import lstsq

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication.
    
    Uses randomized alternating linear solver with half-integer constraint checking.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    def verify_decomposition(U, V, W, r):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def is_half_integer(arr, tol=1e-6):
        """Check if all entries are close to half-integers."""
        doubled = 2 * arr
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=tol)
    
    def snap_to_half_integer(arr, tol=1e-6):
        """Snap values to nearest half-integers if close enough."""
        if not is_half_integer(arr, tol):
            return None
        doubled = np.round(2 * arr)
        return doubled / 2.0
    
    def solve_for_W(U, V, r):
        """Given U (nm x r) and V (mp x r), solve for W (pn x r).
        
        For each (i,k) pair: W[k*n+i, :] must satisfy
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
        
        This is: for fixed (i,k), we need W_row such that
        sum_j (sum_c U[i*m+j,c] * V[j*p+k,c] * W_row[c]) = 1
        
        Let's define for each (i,k) and each c:
        coeff_c = sum_j U[i*m+j, c] * V[j*p+k, c]
        
        Then we need: sum_c coeff_c * W_row[c] = 1
        
        But this is one equation with r unknowns for each (i,k). We have pn = 12 equations.
        Actually, the constraint is that for each (i,k), the dot product of W[k*n+i,:] 
        with the vector coeff_{i,k} must equal 1.
        
        So we build a system: for each (i,k), sum_c A[(i,k), c] * W[k*n+i, c] = 1
        where A[(i,k), c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        
        This is a block-diagonal system: each row of W is independent!
        For row idx = k*n+i: sum_c A[idx, c] * W[idx, c] = 1
        """
        W = np.zeros((pn, r))
        for i in range(n):
            for k in range(p):
                idx = k * n + i
                # Compute coefficient vector for this (i,k)
                coeff = np.zeros(r)
                for j in range(m):
                    coeff += U[i*m+j, :] * V[j*p+k, :]
                
                # We need: sum_c coeff[c] * W[idx, c] = 1
                # This is underdetermined if r > 1. We want a solution that's close to half-integers.
                # Simple approach: find the minimum norm solution, then try to adjust.
                if np.allclose(coeff, 0):
                    return None  # Cannot solve
                
                # Minimum norm solution: W[idx, :] = coeff / (coeff . coeff)
                norm_sq = np.dot(coeff, coeff)
                if abs(norm_sq) < 1e-12:
                    return None
                W[idx, :] = coeff / norm_sq
        
        return W
    
    def solve_for_U(V, W, r):
        """Given V (mp x r) and W (pn x r), solve for U (nm x r).
        
        For each (i,j) pair: U[i*m+j, :] must satisfy
        sum_k (sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]) = 1
        
        For fixed (i,j) and each c:
        coeff_c = sum_k V[j*p+k, c] * W[k*n+i, c]
        
        Then: sum_c coeff_c * U[i*m+j, c] = 1
        """
        U = np.zeros((nm, r))
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                coeff = np.zeros(r)
                for k in range(p):
                    coeff += V[j*p+k, :] * W[k*n+i, :]
                
                if np.allclose(coeff, 0):
                    return None
                
                norm_sq = np.dot(coeff, coeff)
                if abs(norm_sq) < 1e-12:
                    return None
                U[idx, :] = coeff / norm_sq
        
        return U
    
    def solve_for_V(U, W, r):
        """Given U (nm x r) and W (pn x r), solve for V (mp x r).
        
        For each (j,k) pair: V[j*p+k, :] must satisfy
        sum_i (sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]) = 1
        
        For fixed (j,k) and each c:
        coeff_c = sum_i U[i*m+j, c] * W[k*n+i, c]
        
        Then: sum_c coeff_c * V[j*p+k, c] = 1
        """
        V = np.zeros((mp, r))
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                coeff = np.zeros(r)
                for i in range(n):
                    coeff += U[i*m+j, :] * W[k*n+i, :]
                
                if np.allclose(coeff, 0):
                    return None
                
                norm_sq = np.dot(coeff, coeff)
                if abs(norm_sq) < 1e-12:
                    return None
                V[idx, :] = coeff / norm_sq
        
        return V
    
    def try_rank(r, num_trials=200, seed=42):
        """Try to find a decomposition of rank r using alternating solver."""
        rng = np.random.RandomState(seed)
        
        for trial in range(num_trials):
            # Random initialization with small integers/half-integers
            U = rng.randint(-2, 3, size=(nm, r)) / 2.0
            V = rng.randint(-2, 3, size=(mp, r)) / 2.0
            W = rng.randint(-2, 3, size=(pn, r)) / 2.0
            
            # Alternating minimization
            converged = False
            for iteration in range(50):
                # Solve for W given U, V
                W_new = solve_for_W(U, V, r)
                if W_new is None:
                    break
                W = W_new
                
                # Solve for U given V, W
                U_new = solve_for_U(V, W, r)
                if U_new is None:
                    break
                U = U_new
                
                # Solve for V given U, W
                V_new = solve_for_V(U, W, r)
                if V_new is None:
                    break
                V = V_new
                
                # Check convergence
                if iteration > 5:
                    if (np.allclose(U, U_new, atol=1e-10) and 
                        np.allclose(V, V_new, atol=1e-10) and 
                        np.allclose(W, W_new, atol=1e-10)):
                        converged = True
                        break
            
            if not converged:
                continue
            
            # Check if solution is close to half-integers
            U_snapped = snap_to_half_integer(U, tol=1e-5)
            V_snapped = snap_to_half_integer(V, tol=1e-5)
            W_snapped = snap_to_half_integer(W, tol=1e-5)
            
            if U_snapped is None or V_snapped is None or W_snapped is None:
                continue
            
            # Verify exactness with snapped values
            if verify_decomposition(U_snapped, V_snapped, W_snapped, r):
                # Convert to list format
                U_list = [[clean_val(x) for x in row] for row in U_snapped.tolist()]
                V_list = [[clean_val(x) for x in row] for row in V_snapped.tolist()]
                W_list = [[clean_val(x) for x in row] for row in W_snapped.tolist()]
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def clean_val(v):
        """Clean a value to integer if it's very close to one."""
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
        else:
            # Check if it's a clean half-integer
            doubled = 2 * v
            if abs(doubled - round(doubled)) < 1e-9:
                return round(doubled) / 2
            return v
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: at least max(nm, mp, pn) = 24? Not necessarily.
    # Let's try from 24 to 47
    for r in range(24, 48):
        result = try_rank(r, num_trials=100, seed=42 + r)
        if result is not None:
            return result
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
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
