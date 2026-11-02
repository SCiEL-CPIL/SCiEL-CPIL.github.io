import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Target Tensor Indices
    targets_alpha = []
    targets_beta = []
    targets_gamma = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets_alpha.append(a)
                targets_beta.append(b)
                targets_gamma.append(c)
                
    targets_alpha = np.array(targets_alpha)
    targets_beta = np.array(targets_beta)
    targets_gamma = np.array(targets_gamma)
    
    # Precompute flat indices for targets to speed up residual check
    # But we need full tensor for Jacobian?
    # Actually, we only need to satisfy the full tensor being equal to Target.
    # Target is 1 at (a,b,c) in targets, 0 elsewhere.
    
    # Precompute target tensor as a flat vector for quick comparison?
    # T_target_flat is 900 long.
    T_target_flat = np.zeros(size_U * size_V * size_W)
    for a, b, c in zip(targets_alpha, targets_beta, targets_gamma):
        idx = a * size_V * size_W + b * size_W + c
        T_target_flat[idx] = 1.0

    def residuals(x, R):
        U = x[:size_U*R].reshape(size_U, R)
        V = x[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
        W = x[(size_U+size_V)*R:].reshape(size_W, R)
        
        # Compute full tensor
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        
        return (T_full - T_target_flat).flatten()

    def jacobian(x, R):
        U = x[:size_U*R].reshape(size_U, R)
        V = x[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
        W = x[(size_U+size_V)*R:].reshape(size_W, R)
        
        J = np.zeros((size_U * size_V * size_W, (size_U + size_V + size_W) * R))
        
        # Derivatives w.r.t U
        # dT[a,b,c]/dU[a',r'] = delta(a,a') * V[b,r'] * W[c,r']
        # We can compute this by broadcasting.
        # V[b, r'] and W[c, r']
        # For each r', we have a matrix V_r' (size_V x 1) and W_r' (size_W x 1)
        # The contribution to J for variable U[a', r'] is V[b, r'] * W[c, r'] if a==a'
        
        # This is tricky to vectorize directly into the flat J matrix without loops.
        # Given the small size, a loop over R might be acceptable.
        
        for r in range(R):
            # U part
            # Variables: size_U
            # For each a', the derivative is V[:, r] * W[:, r] broadcast to (size_U, size_V, size_W)
            # Actually, dT[a,b,c]/dU[a',r] is non-zero only if a==a'.
            # So for a fixed a', the slice J[:, offset_U + a'] is V[:, r] * W[:, r] (flattened)
            
            # Let's fill the J matrix.
            # J shape: (N, M) where N=900, M=31*R
            
            # U variables: indices 0 .. size_U*R - 1
            # For a in range(size_U):
            #   var_idx = a * R + r
            #   J[:, var_idx] = (V[:, r] * W[:, r]).flatten()
            
            # This is still a loop over size_U (6).
            for a in range(size_U):
                var_idx = a * R + r
                J[:, var_idx] = (V[:, r] * W[:, r]).flatten()
            
            # V variables: indices size_U*R .. size_U*R + size_V*R - 1
            # For b in range(size_V):
            #   var_idx = size_U*R + b * R + r
            #   J[:, var_idx] = (U[:, r] * W[:, r]).flatten()
            
            for b in range(size_V):
                var_idx = size_U*R + b * R + r
                J[:, var_idx] = (U[:, r] * W[:, r]).flatten()
            
            # W variables: indices size_U*R + size_V*R .. end
            # For c in range(size_W):
            #   var_idx = (size_U + size_V)*R + c * R + r
            #   J[:, var_idx] = (U[:, r] * V[:, r]).flatten()
            
            for c in range(size_W):
                var_idx = (size_U + size_V)*R + c * R + r
                J[:, var_idx] = (U[:, r] * V[:, r]).flatten()
                
        return J

    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    def try_rank(R, num_restarts=3):
        size_U = n * m
        size_V = m * p
        size_W = p * n
        num_vars = (size_U + size_V + size_W) * R
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Initialization
            # Try random Gaussian
            np.random.seed(42 + restart + R) # Reproducibility for debugging, maybe remove later
            x0 = np.random.randn(num_vars) * 0.1
            
            try:
                res = least_squares(residuals, x0, jac=jacobian, args=(R,), method='trf', max_nfev=100)
                
                if res.success or res.cost < 1e-6:
                    x_sol = res.x
                    # Snap to half-integers
                    x_snapped = np.round(x_sol * 2) / 2.0
                    
                    # Verify
                    res_check = residuals(x_snapped, R)
                    if np.max(np.abs(res_check)) < 1e-9:
                        U = x_snapped[:size_U*R].reshape(size_U, R)
                        V = x_snapped[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
                        W = x_snapped[(size_U+size_V)*R:].reshape(size_W, R)
                        
                        return U.tolist(), V.tolist(), W.tolist()
                        
            except Exception as e:
                pass
        return None

    # Try ranks
    # Lower bound for rank of 2x3x5 is likely around 14-16.
    # Let's try 12 to 20.
    for R in range(12, 21):
        if time.time() > deadline:
            break
        
        res = try_rank(R, num_restarts=2)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
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

# EVOLVE-BLOCK-END
