"""Matrix multiplication 3x5x7 - rank minimization via linear solve with half-integer constraint."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p):
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
                if abs(val - expected) > 1e-10:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    from fractions import Fraction
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = Fraction(1)
                val = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(U[i*m+j][c])
                    v_val = Fraction(V[j*p+k][c])
                    w_val = Fraction(W[k*n+i][c])
                    val += u_val * v_val * w_val
                if val != target:
                    return False
    return True

def _is_half_integer(x, tol=1e-10):
    """Check if x is close to a half-integer."""
    doubled = x * 2
    nearest = round(doubled)
    return abs(doubled - nearest) < tol

def _round_to_half_integer(x):
    """Round x to nearest half-integer."""
    return round(x * 2) / 2.0

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
    # Schoolbook baseline: rank 105
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_rank(r, num_trials=200):
        """Try to find a decomposition of rank r."""
        nonlocal best_result, best_rank
        if r >= best_rank:
            return False
        
        random.seed(42 + r)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return False
            
            # Generate random V and W with small integer entries
            # Use values from {-2, -1, 0, 1, 2}
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(pn)]
            
            # Convert to numpy for solving
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            U = [[0]*r for _ in range(nm)]
            valid = True
            
            for i in range(n):
                if time.time() > deadline:
                    valid = False
                    break
                
                # Build the system for this i
                # Unknowns: U[i*m+j, c] for j in range(m), c in range(r) -> m*r unknowns
                # Equations: for j in range(m), k in range(p):
                #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j*p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = V_np[j*p+k, c] * W_np[k*n+i, c]
                
                # Solve least squares
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    valid = False
                    break
                
                # Check if solution is close to half-integer
                sol_rounded = np.round(sol * 2) / 2.0
                max_err = np.max(np.abs(sol - sol_rounded))
                
                if max_err > 1e-8:
                    valid = False
                    break
                
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])
            
            if not valid:
                continue
            
            # Verify the full solution
            if _verify_np(U, V, W, n, m, p):
                # Double-check with exact arithmetic
                if _verify_exact(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = r
                    return True
        
        return False
    
    # Try ranks from lower to higher
    # Lower bound considerations:
    # The tensor has 105 non-zero entries, so rank must be at least 1
    # But more realistically, for matrix multiplication, known lower bounds exist
    # Let's try from reasonable values upward
    
    # Start from a lower estimate and go up
    # For 3x5x7, let's try ranks from 50 to 104
    for r in range(50, 105):
        if time.time() > deadline:
            break
        if try_rank(r, num_trials=100):
            # Found a valid decomposition at rank r, try to go lower
            break
    
    # If we found something better, try even lower ranks
    if best_rank < 105:
        for r in range(30, best_rank):
            if time.time() > deadline:
                break
            if try_rank(r, num_trials=50):
                break
    
    return best_result
