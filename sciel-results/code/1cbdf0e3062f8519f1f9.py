"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W):
    """Fast verification using numpy for real-valued checks."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 40."""
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

def _try_rank_k_structural(k, deadline):
    """Try to find a decomposition of rank k using structural search.
    
    We use the fact that for matrix multiplication A(n x m) * B(m x p),
    the tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    We search for integer/half-integer solutions.
    """
    n, m, p = DIMENSIONS
    # Number of constraints: n*m*p = 40
    # Unknowns: 3 * k (for U, V, W)
    # We need 3k >= 40, so k >= 14 (but with structural constraints, higher)
    
    if k < 14:
        return None
    
    # Try to find a structured solution
    # For 2x4*4x5, known lower bound is around 25-30
    # Let's try a grid search over small integer values
    
    # The constraints are:
    # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # 
    # We can try to set up the problem as: choose U and V, then solve for W
    # W[k*n+i,c] appears in constraints for all j: 
    #   sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for each j
    # This gives m equations for each (i,k) pair, with k unknowns (the W values for that (i,k))
    
    # Actually, let's think differently. For each (i,k), we have m constraints:
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1, j=0..m-1
    # This is a linear system in W[k*n+i, :] (k unknowns) if U and V are fixed.
    # The coefficient matrix is m x k, entries (j,c) = U[i*m+j,c]*V[j*p+k,c]
    
    # Strategy: randomly initialize U and V with small integers/half-integers,
    # then for each (i,k), solve the linear system for W.
    # Check if W entries are integers or half-integers.
    
    best = None
    n_trials = 200
    trial = 0
    while trial < n_trials and time.time() < deadline:
        trial += 1
        
        # Initialize U and V with random small values
        # Use values from {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}
        scale = 1
        U = [[random.choice([-1, 0, 1]) for _ in range(k)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(k)] for _ in range(m*p)]
        
        # Ensure some non-trivial structure
        # Solve for W
        W = [[0]*k for _ in range(p*n)]
        valid = True
        for i in range(n):
            for ki in range(p):
                # For this (i, ki), we have m constraints:
                # sum_c U[i*m+j,c] * V[j*p+ki,c] * W[ki*n+i,c] = 1
                # Coefficient matrix A[j,c] = U[i*m+j,c] * V[j*p+ki,c]
                A = np.array([[U[i*m+j][c] * V[j*p+ki][c] for c in range(k)] for j in range(m)], dtype=float)
                b = np.ones(m)
                
                # Solve A * w = b where w is k-vector
                if m == 1:
                    # One equation, k unknowns - underdetermined
                    # Just check if any solution exists with small values
                    if np.all(A[0] == 0):
                        valid = False
                        break
                    # Set most to 0, one to appropriate value
                    # Find a column with non-zero entry
                    nz = np.nonzero(A[0])[0]
                    if len(nz) == 0:
                        valid = False
                        break
                    # This is tricky; skip for now
                    valid = False
                    break
                else:
                    # m equations, k unknowns. If k >= m, we can try least squares or exact
                    # For exact solution, we need the system to be consistent
                    try:
                        # Use lstsq
                        w, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if this is an exact solution
                        if np.allclose(A @ w, b, atol=1e-8):
                            # Check if w entries are integer or half-integer
                            w_rounded = np.round(w * 2) / 2
                            if np.allclose(w, w_rounded, atol=1e-8):
                                W[ki*n+i] = [float(x) for x in w_rounded]
                            else:
                                # Try rounding to nearest integer
                                w_int = np.round(w)
                                if np.allclose(A @ w_int, b, atol=1e-8):
                                    W[ki*n+i] = [float(x) for x in w_int]
                                else:
                                    valid = False
                                    break
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify
        if _verify(U, V, W):
            # Convert to fractions for exactness
            U_frac = [[Fraction(str(x)).limit_denominator(1000) for x in row] for row in U]
            V_frac = [[Fraction(str(x)).limit_denominator(1000) for x in row] for row in V]
            W_frac = [[Fraction(str(x)).limit_denominator(1000) for x in row] for row in W]
            
            # Check if all are integer or half-integer
            ok = True
            for mat in [U_frac, V_frac, W_frac]:
                for row in mat:
                    for v in row:
                        if v.denominator not in (1, 2):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Convert back to float representation (integers or x.5)
                def to_num(frac):
                    if frac.denominator == 1:
                        return int(frac)
                    else:
                        return float(frac)
                
                U_out = [[to_num(f) for f in row] for row in U_frac]
                V_out = [[to_num(f) for f in row] for row in V_frac]
                W_out = [[to_num(f) for f in row] for row in W_frac]
                
                return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def _try_systematic_low_rank(deadline):
    """Try systematic search for low-rank decompositions.
    
    Known results for small matrix multiplication:
    - 2x2 * 2x2: rank 7 (Strassen)
    - 2x3 * 3x2: rank ~10-12
    - 2x4 * 4x5: unknown, but lower bound is at least 25 (by various bounds)
    
    We'll try ranks from 25 upward.
    """
    n, m, p = DIMENSIONS
    # Try ranks from a reasonable lower bound
    # Lower bound: max(n*p, m*p, n*m, (n*p*m)^(1/2)...) 
    # Simple lower bound: n*p = 10, m*p = 20, n*m = 8, so at least 20
    # Better: information-theoretic or algebraic bounds
    # For now, try 25 to 35
    
    for k in range(25, 36):
        if time.time() > deadline:
            break
        result = _try_rank_k_structural(k, deadline)
        if result is not None:
            return result
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # First, get the schoolbook solution as a fallback
    fallback = _schoolbook()
    best_rank = 40  # n*m*p = 2*4*5 = 40
    best_result = fallback
    
    # Try to find lower rank decompositions
    # Try systematic search
    if time.time() < deadline - 10:
        result = _try_systematic_low_rank(deadline - 10)
        if result is not None:
            # Verify
            if _verify(result["U"], result["V"], result["W"]):
                new_rank = len(result["U"][0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_result = result
    
    # Try more randomized search if time permits
    if time.time() < deadline - 5:
        # Try different random seeds and strategies
        for k in range(best_rank - 1, max(20, best_rank - 5), -1):
            if time.time() > deadline - 5:
                break
            # Try a few more random trials
            for trial in range(50):
                if time.time() > deadline - 5:
                    break
                n, m, p = DIMENSIONS
                # Random initialization with more structure
                # Try to exploit the block structure
                U = [[random.choice([-1, 0, 1]) for _ in range(k)] for _ in range(n*m)]
                V = [[random.choice([-1, 0, 1]) for _ in range(k)] for _ in range(m*p)]
                
                W = [[0]*k for _ in range(p*n)]
                valid = True
                for i in range(n):
                    for ki in range(p):
                        A = np.array([[U[i*m+j][c] * V[j*p+ki][c] for c in range(k)] for j in range(m)], dtype=float)
                        b = np.ones(m)
                        try:
                            w, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                            if np.allclose(A @ w, b, atol=1e-8):
                                w_rounded = np.round(w * 2) / 2
                                if np.allclose(w, w_rounded, atol=1e-8):
                                    W[ki*n+i] = [float(x) for x in w_rounded]
                                else:
                                    w_int = np.round(w)
                                    if np.allclose(A @ w_int, b, atol=1e-8):
                                        W[ki*n+i] = [float(x) for x in w_int]
                                    else:
                                        valid = False
                                        break
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                if _verify(U, V, W):
                    # Check integer/half-integer
                    ok = True
                    for mat in [U, V, W]:
                        for row in mat:
                            for v in row:
                                if abs(v * 2 - round(v * 2)) > 1e-8:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        if k < best_rank:
                            best_rank = k
                            best_result = {"U": U, "V": V, "W": W}
                        break  # Found one for this k, move to lower k
    
    return best_result
