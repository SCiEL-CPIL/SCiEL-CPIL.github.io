"""Matrix multiplication 2x4 x 4x6 - rank-minimal tensor decomposition search."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]

def _schoolbook():
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

def _verify_np(U, V, W):
    """Vectorized verification using numpy."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Flatten indices
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    
    # Precompute index arrays
    idx_U = []
    idx_V = []
    idx_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U.append(i*m+j)
                idx_V.append(j*p+k)
                idx_W.append(k*n+i)
                
    idx_U = np.array(idx_U)
    idx_V = np.array(idx_V)
    idx_W = np.array(idx_W)
    
    U_sel = np.array(U)[idx_U, :]
    V_sel = np.array(V)[idx_V, :]
    W_sel = np.array(W)[idx_W, :]
    
    # Sum over rank dimension
    result = np.sum(U_sel * V_sel * W_sel, axis=1)
    return np.all(result == 1)

def _try_rank_r(r, deadline, seed_offset=0):
    """
    Attempt to find a rank r decomposition.
    Strategy: Randomly initialize U and V with sparse {-1, 0, 1} entries.
    For each (i, k), solve for W[k*n+i, :] using least squares, then round to integers.
    """
    n, m, p = DIMENSIONS
    rng = random.Random(12345 + seed_offset)
    
    start = time.time()
    
    while time.time() < deadline:
        # Random sparse U and V
        # We use a moderate density to allow for solutions
        U = np.zeros((n*m, r), dtype=int)
        V = np.zeros((m*p, r), dtype=int)
        
        # Fill U: ~1 non-zero per row on average? 
        # n*m = 8. r=47. 
        # Let's try 2 non-zeros per row.
        for _ in range(n*m * 2):
            U[rng.randrange(n*m), rng.randrange(r)] = rng.choice([-1, 1])
            
        # Fill V: m*p = 24. 
        # Let's try 2 non-zeros per row.
        for _ in range(m*p * 2):
            V[rng.randrange(m*p), rng.randrange(r)] = rng.choice([-1, 1])
            
        W = np.zeros((p*n, r), dtype=int)
        feasible = True
        
        # Solve for W rows
        for i in range(n):
            for k in range(p):
                # System: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # This is a linear system A @ w = 1, where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A_sys = np.zeros((m, r), dtype=float)
                for j in range(m):
                    A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                
                # Solve A_sys @ w = ones(m)
                try:
                    # Use lstsq for robustness
                    sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                except:
                    feasible = False
                    break
                
                # Check if solution is close to integer
                sol_rounded = np.round(sol)
                if np.max(np.abs(sol - sol_rounded)) > 1e-6:
                    # Check if the rounded solution actually satisfies the equation
                    if not np.all(np.abs(A_sys @ sol_rounded - 1) < 1e-6):
                        feasible = False
                        break
                W[k*n+i, :] = sol_rounded
        
        if feasible:
            if _verify_np(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        if time.time() > deadline:
            break
    return None

def construct():
    start_time = time.time()
    # Hard deadline: 110 seconds to be safe
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Try to find a lower rank.
    # We limit the number of attempts and time per rank to ensure we return schoolbook if we fail.
    # The search is stochastic and unlikely to succeed for r close to 48 with this simple method.
    # But we must try.
    
    # Try rank 47 and 46 with short time budgets
    ranks_to_try = [47, 46]
    
    for r in ranks_to_try:
        if time.time() > deadline - 20.0: # Leave 20s for verification and fallback
            break
            
        # Give 10 seconds per rank attempt
        time_budget = min(10.0, deadline - time.time() - 10.0)
        if time_budget < 1.0:
            break
            
        result = _try_rank_r(r, time.time() + time_budget, seed_offset=r)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback: schoolbook
    return _schoolbook()
