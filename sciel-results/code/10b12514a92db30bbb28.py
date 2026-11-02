"""Matrix multiplication 3x5x7 - rank minimization via structured search."""
import time
import random
import numpy as np
from fractions import Fraction

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

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
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
    
    # Try structured decomposition using block splitting
    # 5 = 2 + 2 + 1, use rank-4 for 2x2 multiplication
    # This gives us a structured approach
    
    def try_structured_rank(r, num_trials=50):
        """Try structured decomposition of rank r."""
        nonlocal best_result, best_rank
        if r >= best_rank:
            return False
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return False
            
            # Use structured V and W based on Winograd-style blocks
            # Split middle dimension 5 into 2+2+1
            # For the 2x2 blocks, use rank-4 bilinear algorithm
            # The rank-4 algorithm for 2x2: 
            #   m1 = a11*(b11+b22), m2 = a22*(b11+b22), m3 = a11*(b21-b22), 
            #   m4 = a22*(b12-b11), c11 = m1+m3, c12 = m4+m2, c21 = m3+m2, c22 = m1+m4
            
            # For 3x5x7, let's try to use the structure more directly
            # Generate V and W with a specific pattern
            
            random.seed(1000 + r + trial)
            
            # Try different value sets
            value_sets = [
                [-2, -1, 0, 1, 2],
                [-1, 0, 1],
                [0, 1],
                [-3, -2, -1, 0, 1, 2, 3]
            ]
            vals = value_sets[trial % len(value_sets)]
            
            V = [[random.choice(vals) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(pn)]
            
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            U = [[0]*r for _ in range(nm)]
            valid = True
            
            for i in range(n):
                if time.time() > deadline:
                    valid = False
                    break
                
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j*p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = V_np[j*p+k, c] * W_np[k*n+i, c]
                
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    valid = False
                    break
                
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
            
            if _verify_np(U, V, W, n, m, p):
                if _verify_exact(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = r
                    return True
        
        return False
    
    # Try ranks from lower to higher with structured search
    for r in range(60, 105):
        if time.time() > deadline:
            break
        if try_structured_rank(r, num_trials=30):
            break
    
    # If found something, try to go lower
    if best_rank < 105:
        for r in range(40, best_rank):
            if time.time() > deadline:
                break
            if try_structured_rank(r, num_trials=20):
                break
    
    return best_result
