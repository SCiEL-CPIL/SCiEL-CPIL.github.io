"""Matrix multiplication 3x5x7 - rank minimization via descending search."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_rank(r, num_attempts=100):
        """
        Try to find a decomposition of rank r.
        Strategy: Random V, W. Solve for U. Check if U is half-integer.
        """
        if time.time() > deadline:
            return None
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Random seed based on rank and attempt for reproducibility
            random.seed(12345 + r * 1000 + attempt)
            
            # Generate random V and W with small integer entries
            # Using -1, 0, 1 increases the chance of small integer solutions for U
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
            
            U = [[0.0]*r for _ in range(nm)]
            valid = True
            
            for i in range(n):
                # System for fixed i:
                # Unknowns: U[i*m+j, c] for j in 0..m-1, c in 0..r-1
                # Total unknowns: m * r
                # Equations: for j in 0..m-1, k in 0..p-1:
                #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Total equations: m * p
                
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                # Build A_mat efficiently
                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        v_row = V[j*p+k]
                        w_row = W[k*n+i]
                        # A_mat[eq_idx, j*r : (j+1)*r] = element-wise product of v_row and w_row
                        A_mat[eq_idx, j*r : (j+1)*r] = np.array(v_row) * np.array(w_row)
                
                # Solve least squares
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if solution is close to half-integers
                sol_rounded = np.round(sol * 2.0) / 2.0
                diff = np.max(np.abs(sol - sol_rounded))
                
                if diff > 1e-8:
                    # Try rounding to integers
                    sol_int = np.round(sol)
                    diff_int = np.max(np.abs(sol - sol_int))
                    if diff_int < diff:
                        sol_rounded = sol_int
                        diff = diff_int
                    else:
                        valid = False
                        break
                
                if diff > 1e-6:
                    valid = False
                    break
                
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])
            
            if not valid:
                continue
            
            # Verify the full solution
            if _verify_np(U, V, W, n, m, p):
                # Convert to exact fractions and double-check
                U_frac = [[Fraction(U[i][j]).limit_denominator(100) for j in range(r)] for i in range(nm)]
                V_frac = [[Fraction(V[i][j]).limit_denominator(100) for j in range(r)] for i in range(mp)]
                W_frac = [[Fraction(W[i][j]).limit_denominator(100) for j in range(r)] for i in range(pn)]
                
                if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
        
        return None

    # Start with schoolbook as the guaranteed best
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Search for lower ranks, starting from just below schoolbook
    # Theoretical minimum is max(n*m, m*p, p*n) = 35, but practically higher.
    # We search downwards from 104 to 50.
    for r in range(best_rank - 1, 49, -1):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(r, num_attempts=50)
        if result is not None:
            best_result = result
            best_rank = r
            break
            
    return best_result
