"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    schoolbook_rank = n * m * p
    # Lower bound from Alon-Boppana: ceil((n+m+p-1)^2 / 4)
    lb = int(np.ceil((n + m + p - 1)**2 / 4.0))
    
    # Fallback: Schoolbook
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140 # 10s margin
    
    def check_decomposition(U, V, W, r, tol=1e-9):
        """Check if U, V, W of rank r exactly decompose the tensor."""
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    res = abs(val - 1.0)
                    if res > max_res:
                        max_res = res
        return max_res < tol

    def is_half_integer(U, V, W, tol=1e-6):
        """Check if entries are close to half-integers."""
        for M in [U, V, W]:
            for val in M:
                # val * 2 should be close to an integer
                if abs(val * 2 - np.round(val * 2)) > tol:
                    return False
        return True

    def round_to_half_integer(M):
        return np.round(M * 2) / 2.0

    def alternating_minimize(U, V, W, r, max_iter=200, tol=1e-10):
        """Alternating minimization to find rank-r decomposition."""
        for iteration in range(max_iter):
            # Fix V, W, optimize U
            new_U = U.copy()
            for i in range(n):
                for j in range(m):
                    M_mat = np.zeros((p, r))
                    for k in range(p):
                        M_mat[k] = V[j*p+k] * W[k*n+i]
                    b = np.ones(p)
                    try:
                        u, _, _, _ = np.linalg.lstsq(M_mat, b, rcond=None)
                        new_U[i*m+j] = u
                    except:
                        pass
            
            # Fix U, W, optimize V
            new_V = V.copy()
            for j in range(m):
                for k in range(p):
                    M_mat = np.zeros((n, r))
                    for i in range(n):
                        M_mat[i] = U[i*m+j] * W[k*n+i]
                    b = np.ones(n)
                    try:
                        v, _, _, _ = np.linalg.lstsq(M_mat, b, rcond=None)
                        new_V[j*p+k] = v
                    except:
                        pass
            
            # Fix U, V, optimize W
            new_W = W.copy()
            for k in range(p):
                for i in range(n):
                    M_mat = np.zeros((m, r))
                    for j in range(m):
                        M_mat[j] = U[i*m+j] * V[j*p+k]
                    b = np.ones(m)
                    try:
                        w, _, _, _ = np.linalg.lstsq(M_mat, b, rcond=None)
                        new_W[k*n+i] = w
                    except:
                        pass
            
            U, V, W = new_U, new_V, new_W
            
            # Check convergence
            # We can check the residual directly
            if check_decomposition(U, V, W, r, tol=1e-8):
                break
        
        return U, V, W

    # Search for lower ranks
    # We start from lb, but lb is 43. Schoolbook is 48.
    # We try 47, 46, 45, 44, 43.
    # It is unlikely that rank 43 is easily found by random init + ALS.
    # Rank 47 or 46 might be.
    
    for r in range(schoolbook_rank - 1, lb - 1, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Number of trials per rank
        # For higher ranks (closer to 48), it's easier.
        num_trials = 20 if r > 45 else 10
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Using small scale to help convergence
            np.random.seed(r * 100 + trial)
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            U, V, W = alternating_minimize(U, V, W, r, max_iter=100)
            
            # Check if the continuous solution is good
            if check_decomposition(U, V, W, r, tol=1e-6):
                # Try rounding
                U_r = round_to_half_integer(U)
                V_r = round_to_half_integer(V)
                W_r = round_to_half_integer(W)
                
                if check_decomposition(U_r, V_r, W_r, r, tol=1e-9):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r
                    found = True
                    break
        
        if found:
            break

    # Convert to list
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
