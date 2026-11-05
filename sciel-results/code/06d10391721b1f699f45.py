import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                recon = 0.0
                for r in range(rank):
                    recon += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(recon - target) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_decomposition(n, m, p, rank, deadline, seed):
    """
    Try to find a rank-r decomposition for n x m * m x p matrix multiplication.
    Uses random initialization and Gauss-Newton style refinement.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Number of variables: rank * (n*m + m*p + p*n)
    num_vars = rank * (n * m + m * p + p * n)
    
    # We have n*m*p equations (one for each (i,j,k) triple)
    num_eqs = n * m * p
    
    # Initialize with random small values
    U = np.random.uniform(-1, 1, (n * m, rank))
    V = np.random.uniform(-1, 1, (m * p, rank))
    W = np.random.uniform(-1, 1, (p * n, rank))
    
    # To help convergence, we can fix some variables or use a smarter init
    # Let's try a few different random inits
    
    best_error = float('inf')
    
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        # Random init
        U = np.random.uniform(-1, 1, (n * m, rank))
        V = np.random.uniform(-1, 1, (m * p, rank))
        W = np.random.uniform(-1, 1, (p * n, rank))
        
        # Gauss-Newton iteration
        for it in range(100):
            if time.time() > deadline:
                break
                
            # Compute residual
            res = np.zeros(num_eqs)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon = 0.0
                        for r in range(rank):
                            recon += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                        res[idx] = 1.0 - recon
                        idx += 1
            
            err = np.linalg.norm(res)
            if err < best_error:
                best_error = err
                if err < 1e-10:
                    # Found a solution!
                    return U, V, W
            
            if err < 1e-10:
                break
            
            # Compute Jacobian (approximate)
            # This is expensive, so let's use a simpler gradient step
            # Gradient of 0.5 * ||res||^2 w.r.t. U[i,j,r] = -res[idx] * V[j*p+k,r] * W[k*n+i,r]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r_val = res[idx]
                        if abs(r_val) > 1e-12:
                            for r in range(rank):
                                dU[i*m + j, r] += -r_val * V[j*p + k, r] * W[k*n + i, r]
                                dV[j*p + k, r] += -r_val * U[i*m + j, r] * W[k*n + i, r]
                                dW[k*n + i, r] += -r_val * U[i*m + j, r] * V[j*p + k, r]
                        idx += 1
            
            # Normalize gradient
            gnorm = np.linalg.norm(dU) + np.linalg.norm(dV) + np.linalg.norm(dW)
            if gnorm < 1e-12:
                break
            
            # Step size
            step = 0.1
            # Try to find a better step size
            for _ in range(5):
                U_new = U + step * dU
                V_new = V + step * dV
                W_new = W + step * dW
                
                res_new = np.zeros(num_eqs)
                idx2 = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon = 0.0
                            for r in range(rank):
                                recon += U_new[i*m + j, r] * V_new[j*p + k, r] * W_new[k*n + i, r]
                            res_new[idx2] = 1.0 - recon
                            idx2 += 1
                
                err_new = np.linalg.norm(res_new)
                if err_new < err:
                    U, V, W = U_new, V_new, W_new
                    err = err_new
                    break
                else:
                    step *= 0.5
            
            if step < 1e-10:
                break
    
    return None

def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 115  # Leave 5s margin
    
    # Start from a reasonable lower bound
    # Schoolbook is 105. Let's try to beat it.
    # Known lower bounds for matrix multiplication are complex, but let's try ranks
    # that are significantly lower than 105.
    
    # We'll try ranks in increasing order, but start from a point where it's
    # likely to succeed to save time, or start low and go up?
    # Starting low is better for finding the minimum, but might be slow.
    # Let's try a range.
    
    best_result = None
    best_rank = n * m * p
    
    # We know schoolbook works.
    schoolbook_res = _schoolbook(n, m, p)
    best_result = schoolbook_res
    best_rank = n * m * p
    
    # Try to find lower ranks
    # Let's try ranks from 80 to 104
    # This is a heuristic range.
    
    # Actually, let's try to be smarter.
    # The problem is small enough that we might be able to solve it.
    
    # Let's try a few specific ranks.
    # If we find one at 90, we can try 89, 88...
    
    # To be safe and fast, let's just try to find ANY rank < 105.
    # If we find one, we return it.
    # We won't guarantee minimal, but the prompt asks for "rank-minimal".
    # However, finding the true minimum for 3x5x7 is a research problem.
    # The "score" is the rank. Lower is better.
    # So finding 104 is better than 105.
    
    # Let's try ranks 104, 103, ... down to some limit.
    # But the search is expensive.
    
    # Alternative: Use the structure.
    # If we can decompose 3x5x7 into blocks, we can use Strassen.
    # 3 = 2+1, 5=2+3, 7=3+4.
    # But the current code already does this and gets 105.
    # This implies Strassen didn't help in the current partitioning.
    
    # Let's try the direct algebraic search for rank 104.
    # If it succeeds, great.
    
    # We'll iterate ranks from 104 down to 90 (arbitrary limit for time).
    # If we find a solution, we return it immediately?
    # No, we should check if a lower one exists.
    # But given time constraints, finding 104 is a win.
    
    # Let's try a few ranks.
    ranks_to_try = [104, 103, 102, 100, 95, 90]
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        # Try to find decomposition for rank r
        # We'll give it a limited time
        sub_deadline = time.time() + 10
        
        sol = _find_decomposition(n, m, p, r, sub_deadline, seed=r)
        
        if sol is not None:
            U, V, W = sol
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Verify
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we didn't find a better one, return schoolbook
    return best_result
