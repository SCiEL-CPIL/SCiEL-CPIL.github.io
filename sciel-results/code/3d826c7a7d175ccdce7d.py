"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before the 150s wall-clock limit
    rng = random.Random(42)
    np.random.seed(42)

    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    nm, mp, pn = n * m, m * p, p * n

    # Build the full tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition of rank n*m*p (guaranteed valid baseline).
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 32

    # Verify a decomposition: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]
    def verify(U, V, W, tol=1e-8):
        rank = U.shape[1]
        # Compute tensor from decomposition
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_decomp, T, atol=tol)

    # Improved ALS with proper handling and convergence checks
    def als_decompose(T_tensor, rank, max_iter=100, tol=1e-10, seed=42):
        """
        Improved ALS decomposition: T ≈ sum_r U_r ⊗ V_r ⊗ W_r
        Uses proper initialization and convergence criteria.
        """
        local_rng = np.random.RandomState(seed)
        nm, mp, pn = T_tensor.shape
        
        # Initialize with structured pattern based on tensor sparsity
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        
        # Structured initialization: use the non-zero pattern of T
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if idx < rank:
                        U[i * m + j, idx] = 1.0
                        V[j * p + k, idx] = 1.0
                        W[k * n + i, idx] = 1.0
                        idx += 1
        # Fill remaining with small random values
        if idx < rank:
            U[idx:, :] = local_rng.randn(nm - idx, rank - idx) * 0.01
            V[idx:, :] = local_rng.randn(mp - idx, rank - idx) * 0.01
            W[idx:, :] = local_rng.randn(pn - idx, rank - idx) * 0.01
        
        prev_residual = np.inf
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            VW = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, rank)
            VWt_VW = VW.T @ VW + 1e-10 * np.eye(rank)
            for a in range(nm):
                target = T_tensor[a, :, :].flatten()
                b_vec = VW.T @ target
                try:
                    U[a, :] = np.linalg.solve(VWt_VW, b_vec)
                except np.linalg.LinAlgError:
                    U[a, :] = np.linalg.lstsq(VWt_VW, b_vec, rcond=None)[0]
            
            # Update V
            UW = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, rank)
            UWt_UW = UW.T @ UW + 1e-10 * np.eye(rank)
            for b in range(mp):
                target = T_tensor[:, b, :].flatten()
                b_vec = UW.T @ target
                try:
                    V[b, :] = np.linalg.solve(UWt_UW, b_vec)
                except np.linalg.LinAlgError:
                    V[b, :] = np.linalg.lstsq(UWt_UW, b_vec, rcond=None)[0]
            
            # Update W
            UV = np.einsum('ar,br->abr', U, V).reshape(nm * mp, rank)
            UVt_UV = UV.T @ UV + 1e-10 * np.eye(rank)
            for c in range(pn):
                target = T_tensor[:, :, c].flatten()
                b_vec = UV.T @ target
                try:
                    W[c, :] = np.linalg.solve(UVt_UV, b_vec)
                except np.linalg.LinAlgError:
                    W[c, :] = np.linalg.lstsq(UVt_UV, b_vec, rcond=None)[0]
            
            # Compute residual
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.sum((T_decomp - T_tensor) ** 2)
            
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if abs(prev_residual - residual) < tol:
                break
            prev_residual = residual
        
        return best_U, best_V, best_W, best_residual

    # Try various ranks from low to high
    for target_rank in range(16, best_rank + 1):
        if time.time() > deadline:
            break
        
        rank = target_rank
        found = False
        
        # Multiple restarts with different seeds
        num_restarts = min(15, max(5, int((deadline - time.time()) / 8)))
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Use different seeds for diversity
            seed = 42 + restart * 137
            
            U_opt, V_opt, W_opt, residual = als_decompose(T, rank, max_iter=50, tol=1e-12, seed=seed)
            
            if residual < 1e-8:
                # Verify
                if verify(U_opt, V_opt, W_opt, tol=1e-7):
                    # Try to round to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded, tol=1e-6):
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = target_rank
                        found = True
                        break
                    else:
                        # Use unrounded if it verifies
                        best_U = U_opt
                        best_V = V_opt
                        best_W = W_opt
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            # Found a valid decomposition at this rank
            # Continue to see if we can find a lower rank
            pass
    
    # If no better rank was found, return schoolbook
    if best_rank == n * m * p:
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
