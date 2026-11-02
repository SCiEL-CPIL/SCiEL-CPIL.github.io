import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
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

def _verify_fraction(U, V, W, n, m, p):
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

def _snap_to_rational(x, max_denom=8):
    """Snap a float to the nearest rational with denominator <= max_denom."""
    # Use limit_denominator for simplicity, but we want to check if it's close
    frac = Fraction(x).limit_denominator(max_denom)
    if abs(float(frac) - x) < 1e-7:
        return frac
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_rank(r, num_seeds=100):
        """Try to find a decomposition of rank r using random V,W and solving for U."""
        if time.time() > deadline:
            return None
        
        # We will try solving for U given V, W
        # We can also permute which matrix we solve for, but U is smallest (15 vars) vs V (35) or W (21)
        # Solving for U involves 3 independent systems of 35x15 (m*p x m*r)
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                return None
            
            random.seed(12345 + r * 1000 + seed)
            
            # Generate random V and W with small integer entries
            # Values from {-1, 0, 1} is a good start, maybe expand to {-2, -1, 0, 1, 2}
            val_set = [-2, -1, 0, 1, 2]
            
            V = [[random.choice(val_set) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice(val_set) for _ in range(r)] for _ in range(p*n)]
            
            U = [[0]*r for _ in range(n*m)]
            
            valid = True
            for i in range(n):
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
                            A_mat[eq_idx, j*r + c] = V[j*p+k][c] * W[k*n+i][c]
                
                # Solve least squares
                try:
                    # Use lstsq to handle potential rank deficiency or noise
                    sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    valid = False
                    break
                
                # Check if solution is close to rational
                U_row = []
                for j in range(m):
                    for c in range(r):
                        val = sol[j*r + c]
                        frac = _snap_to_rational(val, max_denom=16)
                        if frac is None:
                            valid = False
                            break
                        U_row.append(frac)
                    if not valid:
                        break
                
                if not valid:
                    break
                
                for idx, frac in enumerate(U_row):
                    U[idx // r][idx % r] = frac
                
            if not valid:
                continue
            
            # Verify the full solution with numpy first for speed
            U_np = [[float(x) for x in row] for row in U]
            if _verify_np(U_np, V, W, n, m, p):
                # Convert to exact fractions for final verification
                U_frac = [[Fraction(U[i][j]) for j in range(r)] for i in range(n*m)]
                V_frac = [[Fraction(V[i][j]) for j in range(r)] for i in range(m*p)]
                W_frac = [[Fraction(W[i][j]) for j in range(r)] for i in range(p*n)]
                
                # Double check with fractions
                if _verify_fraction(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
        
        return None
    
    # Try to find a rank 104 decomposition
    # If found, return it. If not, return schoolbook (rank 105).
    result = try_rank(104, num_seeds=200)
    if result is not None:
        return result
        
    # If 104 failed, try 103? Unlikely, but let's try if time permits.
    if time.time() < deadline - 20:
        result = try_rank(103, num_seeds=100)
        if result is not None:
            return result

    return schoolbook()
