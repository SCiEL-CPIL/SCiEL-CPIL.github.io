import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # Indices: i (0..n-1), j (0..m-1), j2 (0..m-1), k (0..p-1), i2 (0..n-1), k2 (0..p-1)
    # T[i, j, j2, k, i2, k2] = 1 if j==j2 and i==i2 and k==k2 else 0
    # Shape: (n, m, m, p, n, p)
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
                
    T_flat = T.flatten()
    
    # Precompute sizes
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    def residual_fn(params, R):
        # params: flattened U, V, W
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        
        len_U = size_U * R
        len_V = size_V * R
        len_W = size_W * R
        
        U_flat = params[:len_U]
        V_flat = params[len_U:len_U + len_V]
        W_flat = params[len_U + len_V:]
        
        U = U_flat.reshape((size_U, R)).T # (R, n*m)
        V = V_flat.reshape((size_V, R)).T # (R, m*p)
        W = W_flat.reshape((size_W, R)).T # (R, p*n)
        
        U_reshaped = U.reshape((R, n, m))
        V_reshaped = V.reshape((R, m, p))
        W_reshaped = W.reshape((R, p, n))
        
        recon = np.zeros_like(T)
        for r in range(R):
            u = U_reshaped[r] # (n, m)
            v = V_reshaped[r] # (m, p)
            w = W_reshaped[r] # (p, n)
            
            # uv[i, j, j2, k] = u[i, j] * v[j2, k]
            uv = np.einsum('ij,jk->ijjk', u, v)
            # recon[i, j, j2, k, k2, i2] += uv[i, j, j2, k] * w[k2, i2]
            # w is (p, n) with indices (k2, i2)
            recon += np.einsum('ijjk,ki->ijkiki', uv, w)
            
        return (recon.flatten() - T_flat)

    def get_schoolbook_solution(R_target=None):
        """Return schoolbook solution padded to R_target if needed, or full if R_target is None"""
        full_rank = n * m * p
        if R_target is None or R_target >= full_rank:
            R = full_rank
        else:
            R = R_target
        
        U = np.zeros((size_U, R))
        V = np.zeros((size_V, R))
        W = np.zeros((size_W, R))
        
        for r in range(min(R, full_rank)):
            i = r // (m*p)
            j = (r // p) % m
            k = r % p
            U[i*m + j, r] = 1.0
            V[j*p + k, r] = 1.0
            W[k*n + i, r] = 1.0
            
        return U, V, W

    def generate_sparse_init(R, sparsity=0.1, scale=1.0):
        """Generate a sparse random initialization"""
        total_size = (size_U + size_V + size_W) * R
        x0 = np.zeros(total_size)
        num_non_zero = int(total_size * sparsity)
        indices = np.random.choice(total_size, num_non_zero, replace=False)
        # Use integer/half-integer values to help convergence to exact solutions
        values = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], num_non_zero)
        x0[indices] = values * scale
        return x0

    best_solution = None
    best_rank = n * m * p
    
    # We know the rank is at most 84.
    # Let's try to find a rank lower than 84.
    # Start from a reasonable lower bound. 
    # For 3x4x7, the rank is likely not dramatically lower than 84, but let's search.
    # If no lower rank is found, we return 84.
    
    # To save time, we will try a few ranks below 84.
    # If we don't find anything, we return the schoolbook solution.
    
    # Strategy:
    # 1. Try R = 84 with sparse init to see if we can find a "cleaner" solution (unlikely to beat 84 rank, but good for verification).
    # 2. Try R from 70 down to 50 with multiple restarts.
    
    # Given the time limit, we should prioritize checking if a lower rank is possible.
    # If not, return 84.
    
    ranks_to_try = list(range(50, 84))
    # Reverse to find the lowest rank first? Or highest?
    # Finding a low rank is harder. Finding a high rank (close to 84) is easier.
    # But we want the MINIMAL rank.
    # So we should try low ranks first? No, low ranks are very constrained.
    # It is better to try ranks in increasing order? No, if 84 works, and 83 doesn't, we want 84.
    # But if 70 works, we want 70.
    # So we should try 50, 51, ... until we find a solution?
    # That might take too long if 50-70 are impossible.
    
    # Let's try a mix. First, ensure we have the 84 solution.
    # Then try to improve.
    
    # Fallback: Schoolbook
    U_sb, V_sb, W_sb = get_schoolbook_solution()
    best_solution = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = 84

    # Search for lower ranks
    # We will try ranks from 83 down to 60.
    # For each rank, we do a few restarts.
    
    # To maximize the chance of finding a solution, we use sparse initializations.
    
    for R in range(83, 59, -1):
        if time.time() > deadline:
            break
            
        # If we found a solution at R+1, we only need to check R if we want to go lower.
        # Since we are iterating downwards, if we find a solution at R, we break.
        
        found = False
        num_restarts = 10
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Sparse initialization
            x0 = generate_sparse_init(R, sparsity=0.05, scale=1.0)
            
            try:
                res = least_squares(residual_fn, x0, args=(R,), method='lm', max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                # Check if we can snap to half-integers
                len_U = size_U * R
                len_V = size_V * R
                len_W = size_W * R
                
                U_flat = res.x[:len_U]
                V_flat = res.x[len_U:len_U + len_V]
                W_flat = res.x[len_U + len_V:]
                
                U_snap = np.round(U_flat * 2) / 2
                V_snap = np.round(V_flat * 2) / 2
                W_snap = np.round(W_flat * 2) / 2
                
                # Verify snapped solution
                x_snapped = np.concatenate([U_snap, V_snap, W_snap])
                res_snapped = residual_fn(x_snapped, R)
                
                if np.max(np.abs(res_snapped)) < 1e-9:
                    # Valid!
                    best_rank = R
                    best_solution = {
                        "U": U_snap.reshape((size_U, R)).tolist(),
                        "V": V_snap.reshape((size_V, R)).tolist(),
                        "W": W_snap.reshape((size_W, R)).tolist()
                    }
                    found = True
                    break
        
        if found:
            break

    return best_solution
