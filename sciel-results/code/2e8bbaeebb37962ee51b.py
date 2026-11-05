import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor indices
    # T is non-zero at (i*m+j, j*p+k, k*n+i) for all i,j,k
    # Value is 1
    
    # Precompute the list of non-zero indices and values
    # We will represent the tensor as a dense array for easy access in ALS
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    def verify(U, V, W):
        # Check if U, V, W exactly decompose T
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # res[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        res = np.einsum('ar,br,cr->abc', U, V, W)
        # Since entries are integers/half-integers, we can check exact equality
        # But floating point might have errors.
        return np.allclose(res, T, atol=1e-9)

    def get_schoolbook():
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

    def als_step(U, V, W, T, max_iters=50, tol=1e-10):
        # ALS for CP decomposition
        # U: (I, R), V: (J, R), W: (K, R)
        I, J, K, R = U.shape[0], V.shape[0], W.shape[0], U.shape[1]
        
        # Precompute Gram matrices
        VV = V.T @ V  # R x R
        WW = W.T @ W  # R x R
        UU = U.T @ U  # R x R
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            # Numerator: T x_2 V x_3 W
            # T x_2 V: (I, K, R)
            # (T x_2 V) x_3 W: (I, R)
            # Equivalent to: np.einsum('ijk,jr,kr->ir', T, V, W)
            num_U = np.einsum('ijk,jr,kr->ir', T, V, W)
            den_U = VV * WW  # Hadamard product
            try:
                U_new = np.linalg.solve(den_U, num_U.T).T
            except np.linalg.LinAlgError:
                U_new = num_U / np.maximum(np.diag(den_U), 1e-9)[:, None]
            
            # Update V
            num_V = np.einsum('ijk,ir,kr->jr', T, U, W)
            den_V = UU * WW
            try:
                V_new = np.linalg.solve(den_V, num_V.T).T
            except np.linalg.LinAlgError:
                V_new = num_V / np.maximum(np.diag(den_V), 1e-9)[:, None]
                
            # Update W
            num_W = np.einsum('ijk,ir,jr->kr', T, U, V)
            den_W = UU * VV
            try:
                W_new = np.linalg.solve(den_W, num_W.T).T
            except np.linalg.LinAlgError:
                W_new = num_W / np.maximum(np.diag(den_W), 1e-9)[:, None]
                
            # Check convergence
            diff = np.linalg.norm(U - U_new) + np.linalg.norm(V - V_new) + np.linalg.norm(W - W_new)
            U, V, W = U_new, V_new, W_new
            if diff < tol:
                break
                
        return U, V, W

    start_time = time.time()
    deadline = 145.0 # Leave 5s margin for safety
    
    best_result = get_schoolbook()
    
    # Try ranks from low to high
    # Known lower bounds for 3x3x4 matrix multiplication are around 24-25?
    # Let's try 24 to 36.
    for R in range(24, 37):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        # Lower ranks are harder, so more attempts
        attempts = 100 if R < 30 else 50
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random values
            # Using small integers or half-integers can help find exact solutions
            # Random uniform in [-1, 1]
            U = np.random.uniform(-1, 1, (n*m, R))
            V = np.random.uniform(-1, 1, (m*p, R))
            W = np.random.uniform(-1, 1, (p*n, R))
            
            # Run ALS
            U, V, W = als_step(U, V, W, T, max_iters=50)
            
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify(U_rounded, V_rounded, W_rounded):
                # Found a valid decomposition
                # Check if it's better than current best
                if R < len(best_result['U'][0]):
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                # Since we iterate R from low to high, the first found is the best for this R
                # We can break out of attempts loop for this R
                break
                
    return best_result
