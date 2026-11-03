import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    # We know 36 works. 30 works (27+3).
    # Try to find lower.
    
    def verify(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check: sum_c W[j*n+i, c] * U[i*m+alpha, c] * V[beta*p+j, c] == delta[alpha, beta]
        # for all i, j, alpha, beta
        r = U.shape[1]
        error = 0.0
        for i in range(n):
            for j in range(p):
                for alpha in range(m):
                    for beta in range(m):
                        val = 0.0
                        for c in range(r):
                            val += W[j*n+i, c] * U[i*m+alpha, c] * V[beta*p+j, c]
                        target = 1.0 if alpha == beta else 0.0
                        error += (val - target)**2
        return error

    def try_rank(r):
        # Initialize
        np.random.seed(42) # For reproducibility in search
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Flatten
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_ = x[:n*m*r].reshape(n*m, r)
            V_ = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_ = x[n*m*r + m*p*r:].reshape(p*n, r)
            return verify(U_, V_, W_)
            
        # Use Nelder-Mead or L-BFGS-B
        # Nelder-Mead is derivative-free, might be robust
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < 1e-6:
                U_ = res.x[:n*m*r].reshape(n*m, r)
                V_ = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_ = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Check for integer/half-integer
                # Scale if needed?
                # If values are like 0.5, 1.0, -0.5
                U_int = np.round(U_ * 2) / 2
                V_int = np.round(V_ * 2) / 2
                W_int = np.round(W_ * 2) / 2
                
                if verify(U_int, V_int, W_int) < 1e-6:
                    return U_int, V_int, W_int
        except Exception:
            pass
        return None

    # Try ranks from low to high
    for r in range(24, 37):
        start_time = time.time()
        result = try_rank(r)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Safety break if taking too long
        if time.time() - start_time > 10:
            continue
            
    # Fallback to known rank 30 construction (27 + 3)
    # Or just return schoolbook 36 if search fails
    # Let's construct rank 30 explicitly
    rank = 30
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # 3x3x3 part (27 terms)
    for i in range(n):
        for j in range(m):
            for k in range(m): # p=3 for this part
                U[i*m+j, col] = 1
                V[j*m+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # 3x3x1 part (3 terms) for the 4th column (index 3 in p)
    # C[i, 3] = sum_alpha A[i, alpha] B[alpha, 3]
    # We need 3 terms.
    # Term for alpha=0: U[i,0] B[0,3] -> C[i,3]
    # But U must be for A.
    # We can just use the schoolbook for the 4th column.
    for i in range(n):
        for j in range(m):
            # k=3
            U[i*m+j, col] = 1
            V[j*m+3, col] = 1
            W[3*n+i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
