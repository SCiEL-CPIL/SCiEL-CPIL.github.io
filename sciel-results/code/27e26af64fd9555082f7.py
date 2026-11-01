"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T for 2x3x6
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def alt_minimize(U, V, W, T, n_iter=3000, deadline=None):
        """Alternating minimization for tensor decomposition."""
        for it in range(n_iter):
            if deadline is not None and time.time() > deadline:
                break
            # Update U
            for r in range(U.shape[1]):
                vw = V[:, r] * W[:, r]
                num = np.einsum('abc,bc->a', T, vw)
                den = np.dot(vw, vw)
                if den > 1e-12:
                    U[:, r] = num / den
            # Update V
            for r in range(V.shape[1]):
                uw = U[:, r] * W[:, r]
                num = np.einsum('abc,ac->b', T, uw)
                den = np.dot(uw, uw)
                if den > 1e-12:
                    V[:, r] = num / den
            # Update W
            for r in range(W.shape[1]):
                uv = U[:, r] * V[:, r]
                num = np.einsum('abc,ab->c', T, uv)
                den = np.dot(uv, uv)
                if den > 1e-12:
                    W[:, r] = num / den
        return U, V, W
    
    def check(U, V, W, T):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T))
    
    # Try to find low-rank decomposition
    best_result = None
    
    for rank in [18, 19, 20, 21, 22, 24, 28, 32, 36]:
        if time.time() > deadline - 15:
            break
        
        for seed in range(6):
            if time.time() > deadline - 15:
                break
            
            rng = np.random.RandomState(seed * 1000 + rank)
            U = rng.randn(n*m, rank) * 0.05
            V = rng.randn(m*p, rank) * 0.05
            W = rng.randn(p*n, rank) * 0.05
            
            U, V, W = alt_minimize(U, V, W, T, n_iter=2000, deadline=deadline - 15)
            err = check(U, V, W, T)
            
            if err < 1e-8:
                best_result = (U, V, W, rank)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W, _ = best_result
    else:
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
# EVOLVE-BLOCK-END
