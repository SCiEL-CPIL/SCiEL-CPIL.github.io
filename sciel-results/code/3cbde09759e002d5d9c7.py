import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def schoolbook():
        U = np.zeros((nm, nm * mp * pn // (m * p * n) * m * p * n))  # just n*m*p
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def compute_residual(U, V, W):
        R = np.zeros_like(T)
        for c in range(U.shape[1]):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return np.linalg.norm(T - R)
    
    deadline = time.time() + 100
    
    # Fallback
    U, V, W, best_rank = schoolbook()
    
    # Try local search for lower ranks
    # Use alternating least squares (ALS)
    def als_decompose(rank, max_iter=500, tol=1e-8):
        """Try to find rank-r decomposition using ALS"""
        # Initialize
        U = np.random.randn(nm, rank) * 0.3
        V = np.random.randn(mp, rank) * 0.3
        W = np.random.randn(pn, rank) * 0.3
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            old_res = compute_residual(U, V, W)
            
            # Update U
            for c in range(rank):
                # Fix V[:,c] and W[:,c], optimize U[:,c]
                vc = V[:, c]
                wc = W[:, c]
                # We want to minimize ||T - (R_other + U[:,c]⊗vc⊗wc)||
                # = ||T - R_other - U[:,c]⊗(vc⊗wc)||
                # Let X = vc⊗wc (shape mp*pn), then we're doing rank-1 approx of (T-R_other) with second factor X
                # Optimal U[:,c] = (T-R_other) contracted with X, divided by ||X||^2
                R_other = np.zeros_like(T)
                for c2 in range(rank):
                    if c2 != c:
                        R_other += np.einsum('a,b,c->abc', U[:, c2], V[:, c2], W[:, c2])
                target = T - R_other
                # U[:,c] should satisfy: for all a,b,c: target[a,b,c] ≈ U[a,c]*vc[b]*wc[c]
                # Sum over b,c: target[a,b,c]*vc[b]*wc[c] / sum(vc[b]^2*wc[c]^2)
                denom = np.sum(vc**2) * np.sum(wc**2)
                if denom > 1e-12:
                    U[:, c] = np.einsum('abc,b,c->a', target, vc, wc) / denom
            
            # Update V similarly
            for c in range(rank):
                uc = U[:, c]
                wc = W[:, c]
                R_other = np.zeros_like(T)
                for c2 in range(rank):
                    if c2 != c:
                        R_other += np.einsum('a,b,c->abc', U[:, c2], V[:, c2], W[:, c2])
                target = T - R_other
                denom = np.sum(uc**2) * np.sum(wc**2)
                if denom > 1e-12:
                    V[:, c] = np.einsum('abc,a,c->b', target, uc, wc) / denom
            
            # Update W
            for c in range(rank):
                uc = U[:, c]
                vc = V[:, c]
                R_other = np.zeros_like(T)
                for c2 in range(rank):
                    if c2 != c:
                        R_other += np.einsum('a,b,c->abc', U[:, c2], V[:, c2], W[:, c2])
                target = T - R_other
                denom = np.sum(uc**2) * np.sum(vc**2)
                if denom > 1e-12:
                    W[:, c] = np.einsum('abc,a,b->c', target, uc, vc) / denom
            
            new_res = compute_residual(U, V, W)
            if abs(old_res - new_res) < tol:
                break
        
        return U, V, W, compute_residual(U, V, W)
    
    # Try ranks from low to high
    for r in range(12, n * m * p + 1):
        if time.time() > deadline - 25:
            break
        best_res = np.inf
        for trial in range(3):
            if time.time() > deadline - 25:
                break
            U_t, V_t, W_t, res = als_decompose(r)
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U_t, V_t, W_t
        if best_res < 1e-6:
            U, V, W, best_rank = best_U, best_V, best_W, r
            break
    
    # Round to half-integers
    def round_half(arr):
        return np.round(arr * 2) / 2
    
    U_r = round_half(U)
    V_r = round_half(V)
    W_r = round_half(W)
    
    # Verify
    res = compute_residual(U_r, V_r, W_r)
    if res > 1e-6:
        # Fall back to schoolbook
        U, V, W, best_rank = schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
