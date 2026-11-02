import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute index arrays for vectorized residual computation
    # For each (i,j,k) entry, we need indices into U, V, W
    idx_u = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    idx_v = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    idx_w = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    def residual_vectorized(params, R):
        """Vectorized residual computation using matrix operations."""
        size_U = nm * R
        size_V = mp * R
        
        U = params[:size_U].reshape(nm, R)
        V = params[size_U:size_U+size_V].reshape(mp, R)
        W = params[size_U+size_V:].reshape(pn, R)
        
        # For each (i,j,k), compute sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
        # This is equivalent to: sum_r (U[idx_u] * V[idx_v] * W[idx_w])
        # We can compute this as: for each r, sum over the selected indices
        # More efficiently: (U[idx_u] * V[idx_v]) @ W[idx_w].T but that's not quite right
        
        # Actually: for each (i,j,k), we need sum_r U[idx_u[r_i], r] * V[idx_v[r_i], r] * W[idx_w[r_i], r]
        # Let's compute element-wise: U_sel = U[idx_u, :], V_sel = V[idx_v, :], W_sel = W[idx_w, :]
        # Then res[i,j,k] = sum_r U_sel[i,j,k,r] * V_sel[i,j,k,r] * W_sel[i,j,k,r]
        
        U_sel = U[idx_u, :]  # shape (n*m*p, R)
        V_sel = V[idx_v, :]  # shape (n*m*p, R)
        W_sel = W[idx_w, :]  # shape (n*m*p, R)
        
        # res = sum over last axis of U_sel * V_sel * W_sel
        res = np.sum(U_sel * V_sel * W_sel, axis=1)  # shape (n*m*p,)
        
        return (res - 1.0).flatten()

    def verify_solution(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    def try_solve_rank(R, max_attempts=10, seed_offset=0):
        """Try to find a rank-R decomposition."""
        rng = np.random.RandomState(seed_offset)
        
        for attempt in range(max_attempts):
            # Different initialization strategies
            if attempt % 3 == 0:
                # Small random
                scale = 0.1
            elif attempt % 3 == 1:
                # Medium random
                scale = 1.0
            else:
                # Large random
                scale = 5.0
                
            x0 = rng.uniform(-scale, scale, R * (nm + mp + pn))
            
            try:
                result = least_squares(
                    residual_vectorized, 
                    x0, 
                    args=(R,), 
                    method='lm', 
                    max_nfev=2000,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10
                )
                
                if result.cost < 1e-12:
                    U = result.x[:nm*R].reshape(nm, R)
                    V = result.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    W = result.x[nm*R+mp*R:].reshape(pn, R)
                    
                    # Try rounding to half-integers
                    for round_scale in [2, 4, 8]:
                        U_rounded = np.round(U * round_scale) / round_scale
                        V_rounded = np.round(V * round_scale) / round_scale
                        W_rounded = np.round(W * round_scale) / round_scale
                        
                        if verify_solution(U_rounded, V_rounded, W_rounded):
                            return U_rounded, V_rounded, W_rounded
                    
                    # Check if the float solution itself works (should be exact or near-exact)
                    if verify_solution(U, V, W):
                        # Try to find a rational representation
                        # For now, if it's very close to half-integers, use that
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        if verify_solution(U_rounded, V_rounded, W_rounded):
                            return U_rounded, V_rounded, W_rounded
                            
            except Exception:
                continue
                
        return None

    start_time = time.time()
    timeout = 120.0
    
    # Try ranks from lower bound upward
    # Lower bound for matrix multiplication tensor M_{n,m,p} is max(n*m, m*p, p*n)
    # For 2x3x5: max(6, 15, 10) = 15
    lower_bound = max(n*m, m*p, p*n)
    
    for R in range(lower_bound, n*m*p + 1):
        if time.time() - start_time > timeout - 15:
            break
            
        # Try multiple attempts with different seeds
        result = try_solve_rank(R, max_attempts=15, seed_offset=R*100)
        
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback: schoolbook algorithm (rank = n*m*p)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
