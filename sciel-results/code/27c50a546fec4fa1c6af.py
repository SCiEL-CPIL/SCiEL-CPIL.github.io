"""Matrix multiplication 3x5x7 - rank minimization via structured algebraic search."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
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
    deadline = start_time + 110  # Safety margin before 120s limit
    
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
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_rank_with_solve(r, num_attempts=20, seed_base=0):
        """
        Try to find a decomposition of rank r.
        Strategy: fix V and W with random small integer entries,
        solve for U via least squares, then round and verify.
        """
        if time.time() > deadline:
            return None
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            random.seed(seed_base + r * 1000 + attempt)
            
            # Generate random V and W with small integer entries
            # Use values from {-1, 0, 1} to keep things simple
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
            
            U = [[0.0]*r for _ in range(nm)]
            valid = True
            
            for i in range(n):
                # For fixed i, solve for U[i*m+j, c] for all j, c
                # Constraints: for each j in [0,m), k in [0,p):
                #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = float(V[j*p+k][c] * W[k*n+i][c])
                
                # Solve the linear system
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if the solution is close to half-integers or integers
                sol_rounded = np.round(sol * 2.0) / 2.0
                diff = np.max(np.abs(sol - sol_rounded))
                
                if diff > 1e-8:
                    sol_int = np.round(sol)
                    diff_int = np.max(np.abs(sol - sol_int))
                    if diff_int < diff:
                        sol_rounded = sol_int
                        diff = diff_int
                
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
    
    def try_rank_symmetric(r, num_attempts=15, seed_base=10000):
        """
        Try a more structured approach where V and W have additional symmetry.
        """
        if time.time() > deadline:
            return None
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            random.seed(seed_base + r * 1000 + attempt)
            
            V = [[0]*r for _ in range(mp)]
            W = [[0]*r for _ in range(pn)]
            
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        V[j*p+k][c] = random.choice([-1, 0, 1])
            
            for k in range(p):
                for i in range(n):
                    for c in range(r):
                        W[k*n+i][c] = random.choice([-1, 0, 1])
            
            U = [[0.0]*r for _ in range(nm)]
            valid = True
            
            for i in range(n):
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = float(V[j*p+k][c] * W[k*n+i][c])
                
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                sol_rounded = np.round(sol * 2.0) / 2.0
                diff = np.max(np.abs(sol - sol_rounded))
                
                if diff > 1e-8:
                    sol_int = np.round(sol)
                    diff_int = np.max(np.abs(sol - sol_int))
                    if diff_int < diff:
                        sol_rounded = sol_int
                        diff = diff_int
                
                if diff > 1e-6:
                    valid = False
                    break
                
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])
            
            if not valid:
                continue
            
            if _verify_np(U, V, W, n, m, p):
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
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = 35
    # But realistically, for matrix multiplication, the rank is typically higher.
    # Known results: M(2,2,2)=7, M(3,3,3) is between 19 and 23.
    # For 3x5x7, let's try from a reasonable starting point.
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try ranks from 35 upward
    for r in range(35, 106):
        if time.time() > deadline:
            break
        
        # Try structured approach first
        result = try_rank_with_solve(r, num_attempts=25, seed_base=0)
        if result is not None:
            best_result = result
            best_rank = r
            break
        
        # If we have time, also try the symmetric approach
        if time.time() < deadline - 10:
            result = try_rank_symmetric(r, num_attempts=15, seed_base=10000)
            if result is not None:
                best_result = result
                best_rank = r
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
