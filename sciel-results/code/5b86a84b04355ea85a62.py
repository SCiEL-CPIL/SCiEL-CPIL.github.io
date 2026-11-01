import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    def _verify_fast(U, V, W):
        """
        Fast verification: check if U,V,W with half-integer entries exactly decompose the tensor.
        Uses numpy for speed, falls back to exact Fraction check only if close.
        """
        R = U.shape[1]
        
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check if snap was close (within 1e-6)
        if not (np.allclose(U, U_snap, atol=1e-6) and 
                np.allclose(V, V_snap, atol=1e-6) and 
                np.allclose(W, W_snap, atol=1e-6)):
            return False, None
            
        # Quick numpy check: compute all tensor entries
        # Build index arrays for efficiency
        tensor_vals = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    # Compute sum_r U[u,r]*V[v,r]*W[w,r]
                    tensor_vals[idx] = np.sum(U_snap[u_idx] * V_snap[v_idx] * W_snap[w_idx])
                    idx += 1
        
        if not np.allclose(tensor_vals, 1.0, atol=1e-6):
            return False, None
            
        # Exact verification with Fractions (only called when numpy check passes)
        U_frac = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in U_snap]
        V_frac = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in V_snap]
        W_frac = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in W_snap]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = Fraction(0)
                    for r in range(R):
                        s += U_frac[u_idx][r] * V_frac[v_idx][r] * W_frac[w_idx][r]
                    if s != 1:
                        return False, None
        return True, (U_snap, V_snap, W_snap)

    def _schoolbook():
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

    def _residual_vectorized(U, V, W):
        """Vectorized residual computation."""
        R = U.shape[1]
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    res[idx] = np.sum(U[u_idx] * V[v_idx] * W[w_idx]) - 1.0
                    idx += 1
        return res

    def _try_rank_numerical(R, max_restarts=30, time_budget=20.0):
        """
        Attempts to find a decomposition of rank R using iterative optimization.
        Uses a custom Gauss-Newton style approach for speed.
        """
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            return None
            
        time_per_restart = min(time_budget / max_restarts, 3.0)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
                
            np.random.seed(restart * 104729 + R * 7919)
            
            # Smart initialization: start from a structured pattern
            if restart % 5 == 0:
                # Near-schoolbook with perturbations
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                # Initialize first min(R, n*m*p) columns like schoolbook
                for r in range(min(R, n*m*p)):
                    # Map r to (i,j,k)
                    k = r % p
                    j = (r // p) % m
                    i = (r // (p*m)) % n
                    U[i*m + j, r] = 1.0
                    V[j*p + k, r] = 1.0
                    W[k*n + i, r] = 1.0
                # Add small random perturbations to remaining columns
                for r in range(min(R, n*m*p), R):
                    U[:, r] = np.random.randn(n*m) * 0.1
                    V[:, r] = np.random.randn(m*p) * 0.1
                    W[:, r] = np.random.randn(p*n) * 0.1
            elif restart % 5 == 1:
                # Random with structure: each column has limited support
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for r in range(R):
                    # Each tensor component involves specific (i,j,k)
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    U[i*m + j, r] = 1.0
                    V[j*p + k, r] = 1.0
                    W[k*n + i, r] = 1.0
                    # Add some cross terms
                    if r < R // 2:
                        U[(i+1)%m * m + (j+1)%m, r] = 0.5
                        V[(j+1)%m * p + (k+1)%p, r] = 0.5
                        W[(k+1)%p * n + (i+1)%n, r] = 0.5
            elif restart % 5 == 2:
                # Random sparse
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for r in range(R):
                    u_rows = np.random.choice(n*m, size=min(2, n*m), replace=False)
                    v_rows = np.random.choice(m*p, size=min(2, m*p), replace=False)
                    w_rows = np.random.choice(p*n, size=min(2, p*n), replace=False)
                    U[u_rows, r] = np.random.choice([-1, 1], size=len(u_rows)) * np.random.uniform(0.5, 1.5, size=len(u_rows))
                    V[v_rows, r] = np.random.choice([-1, 1], size=len(v_rows)) * np.random.uniform(0.5, 1.5, size=len(v_rows))
                    W[w_rows, r] = np.random.choice([-1, 1], size=len(w_rows)) * np.random.uniform(0.5, 1.5, size=len(w_rows))
            else:
                # Pure random
                scale = 1.0 / np.sqrt(R)
                U = np.random.randn(n*m, R) * scale
                V = np.random.randn(m*p, R) * scale
                W = np.random.randn(p*n, R) * scale
            
            # Gauge fixing
            U[0, :] = 1.0
            
            # Custom optimization: alternating minimization
            best_residual = np.inf
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Compute residual
                res = _residual_vectorized(U, V, W)
                current_res = np.linalg.norm(res)
                
                if current_res < 1e-10:
                    break
                if current_res < best_residual:
                    best_residual = current_res
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Try to update one factor at a time
                # Update U: for each u_idx, solve least squares
                # This is complex; instead use a simple gradient step
                
                # Compute gradients numerically (expensive but simple)
                # Instead, use a random restart within this iteration
                if it % 10 == 9 and current_res > best_residual * 1.01:
                    # Perturb slightly
                    U += np.random.randn(*U.shape) * 0.01
                    V += np.random.randn(*V.shape) * 0.01
                    W += np.random.randn(*W.shape) * 0.01
                    U[0, :] = 1.0
                    continue
                
                # Simple coordinate descent: update one random column at a time
                r_idx = np.random.randint(R)
                # Optimize column r_idx of U, V, W
                # For fixed V[:,r] and W[:,r], optimal U[:,r] can be derived
                # But this is coupled. Use a small step instead.
                
                # Compute gradient for U[:, r_idx]
                # d/dU[u,r] = sum over (j,k) where u=i*m+j of V[v,r]*W[w,r] * (tensor_val - 1)
                # This is complex. Use a heuristic update.
                
                # Instead, do a simple line search on a random direction
                direction_U = np.random.randn(n*m, 1) * 0.1
                direction_V = np.random.randn(m*p, 1) * 0.1
                direction_W = np.random.randn(p*n, 1) * 0.1
                
                # Try a few step sizes
                for alpha in [0.5, 0.3, 0.1, -0.1, -0.3]:
                    U_trial = U + alpha * direction_U
                    V_trial = V + alpha * direction_V
                    W_trial = W + alpha * direction_W
                    U_trial[0, :] = 1.0
                    res_trial = _residual_vectorized(U_trial, V_trial, W_trial)
                    if np.linalg.norm(res_trial) < current_res:
                        U, V, W = U_trial, V_trial, W_trial
                        break
            
            # Check if we found a solution
            res_final = _residual_vectorized(U, V, W)
            if np.linalg.norm(res_final) < 1e-8:
                valid, result = _verify_fast(U, V, W)
                if valid:
                    return result
            
            # Also check best found
            if best_residual < 1e-8:
                valid, result = _verify_fast(best_U, best_V, best_W)
                if valid:
                    return result
        
        return None

    def _try_rank_direct_construction(R):
        """
        Try to construct a low-rank decomposition using known patterns.
        For 2x3x9, try structured decompositions.
        """
        if time.time() > deadline - 10:
            return None
            
        # Try a construction based on the structure of the problem
        # Each output element C[i,k] = sum_j A[i,j]*B[j,k]
        # We need: C[i,k] = sum_r U[i*m+j_r, r] * V[j_r*p+k_r, r] * W[k_r*n+i_r, r]
        # where the indices must be consistent
        
        # Simple construction: use the schoolbook but merge some terms
        # For now, just try random structured approaches
        for attempt in range(10):
            if time.time() > deadline - 5:
                return None
                
            np.random.seed(attempt * 12345 + R)
            
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # Each rank-1 term corresponds to a "path" through the tensor
            for r in range(R):
                # Choose a random (i,j,k) as the main term
                i = r % n
                j = (r // n) % m  
                k = (r // (n*m)) % p
                
                # Base term
                U[i*m + j, r] = 1.0
                V[j*p + k, r] = 1.0
                W[k*n + i, r] = 1.0
                
                # Add auxiliary terms to help with cancellation
                if attempt % 3 == 0 and R > n*m*p//2:
                    # Add a negative term for a different (i',j',k')
                    i2 = (i + 1) % n
                    j2 = (j + 1) % m
                    k2 = (k + 1) % p
                    if r < R - 1:
                        U[i2*m + j2, r+1] = -1.0
                        V[j2*p + k2, r+1] = -1.0
                        W[k2*n + i2, r+1] = -1.0
                        r += 1  # Skip next r since we used it
            
            if R > n*m*p:
                return None  # Can't do better than schoolbook with this approach
            
            valid, result = _verify_fast(U, V, W)
            if valid:
                return result
        
        return None

    # Try direct construction first (fast)
    for R in range(45, 55):
        if time.time() > deadline - 20:
            break
        result = _try_rank_direct_construction(R)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try numerical optimization for lower ranks
    for R in range(48, 42, -1):
        if time.time() > deadline - 15:
            break
        result = _try_rank_numerical(R, max_restarts=15, time_budget=15.0)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return _schoolbook()
