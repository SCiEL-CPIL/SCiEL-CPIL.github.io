import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 23
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Construct target tensor T of shape (9, 9, 9)
    # T[alpha, beta, gamma] = 1 if alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    # Note: The task says W[k*n+i, r].
    # Standard convention: A is n x m, B is m x p, C is n x p.
    # A index: i*m + j
    # B index: j*p + k
    # C index: k*n + i  (Wait, C is n x p, so index is i*p + k? Or k*n + i?)
    # Task says: W[k*n+i, r]. This implies C is stored column-major? Or just a specific indexing.
    # Let's check the seed: W[k*n+i, column] = 1.
    # In seed: for i in n, for j in m, for k in p:
    # U[i*m+j] = 1
    # V[j*p+k] = 1
    # W[k*n+i] = 1
    # This matches C index = k*n + i.
    # So C is effectively stored as (p, n) matrix? Or just index mapping.
    # I will follow the seed's indexing exactly.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    # Function to compute error
    def error(params):
        # params is flat array of length 3 * R * 9
        R = target_rank
        U = params[:R*9].reshape(R, 9).T # 9 x R
        V = params[R*9:2*R*9].reshape(R, 9).T # 9 x R
        W = params[2*R*9:3*R*9].reshape(R, 9).T # 9 x R
        
        # Compute decomposition
        # Decomposed[alpha, beta, gamma] = sum_r U[alpha, r] * V[beta, r] * W[gamma, r]
        # This is an outer product sum.
        # Can be computed as:
        # Temp = np.einsum('ar,br,cr->abc', U, V, W)
        # But U, V, W are 9xR.
        # Let's use einsum on transposed or just loop?
        # einsum is fast.
        # U is 9xR, V is 9xR, W is 9xR
        # We want T_decomp[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        decomp = np.einsum('ar,br,cr->abc', U, V, W)
        diff = decomp - T
        return np.sum(diff * diff)

    # Initial best result (schoolbook)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1

    # Try to find rank 23
    R = target_rank
    found = False
    
    # Multiple random restarts
    while time.time() < deadline and not found:
        # Random initialization
        # Use small random values
        np.random.seed(int(time.time() * 1000) % 10000) # Simple seed variation
        init_params = np.random.uniform(-1, 1, size=3 * R * 9)
        
        # Optimization
        try:
            res = minimize(error, init_params, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < 1e-6:
                # Extract U, V, W
                U = res.x[:R*9].reshape(R, 9).T
                V = res.x[R*9:2*R*9].reshape(R, 9).T
                W = res.x[2*R*9:3*R*9].reshape(R, 9).T
                
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                decomp = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                if np.allclose(decomp, T, atol=1e-9):
                    # Found valid decomposition
                    found = True
                    best_rank = R
                    # Convert to lists
                    best_U = U_rounded.T.tolist()
                    best_V = V_rounded.T.tolist()
                    best_W = W_rounded.T.tolist()
        except Exception:
            pass

    if not found:
        # Fallback to schoolbook
        best_rank = n * m * p
        # Reconstruct schoolbook if not found
        best_U = [[0] * best_rank for _ in range(n * m)]
        best_V = [[0] * best_rank for _ in range(m * p)]
        best_W = [[0] * best_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][column] = 1
                    best_V[j * p + k][column] = 1
                    best_W[k * n + i][column] = 1
                    column += 1

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
