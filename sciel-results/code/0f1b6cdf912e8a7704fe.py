import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Construct the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, o] = 1 if there is a term A[a] B[b] contributing to C[o]
    # Based on schoolbook:
    # a = i*m + j
    # b = j*p + k
    # o = k*n + i
    # T[a, b, o] = 1 for all valid i, j, k.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                o = k * n + i
                T[a, b, o] = 1.0
    
    # Helper to verify a decomposition
    def verify(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Check if sum_c U[:,c] V[:,c] W[:,c] == T
        # We can do this by unfolding or direct summation.
        # Direct summation is O(nm * mp * pn * R) which is 45 * 45 * 15 * R?
        # nm=9, mp=15, pn=15. 9*15*15 = 2025. R~45. 2025*45 ~ 90k ops. Very fast.
        R = U.shape[1]
        T_recon = np.zeros_like(T)
        for c in range(R):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            # T_recon += np.einsum('a,b,o->abo', u, v, w)
            # np.einsum is fast
            T_recon += np.einsum('a,b,o->abo', u, v, w)
        
        # Check if T_recon is close to T
        # Since we snap to half-integers, we should check for exact equality
        # or very close. The problem asks for "integer/half-integer entries that exactly decompose".
        # So T_recon should be exactly T (which is 0 or 1).
        if not np.allclose(T_recon, T, atol=1e-9):
            return False
        # Also check if entries are half-integers
        # The problem says "with integer/half-integer entries".
        # We will enforce this in the snapping step.
        return True

    def snap_to_half_integers(U, V, W):
        # Snap values to nearest 0.5
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    def als_decompose(R, max_iters=100, tol=1e-6):
        """
        Attempt to find a rank-R decomposition using ALS.
        """
        if time.time() > deadline - 5:
            return None
        
        # Initialize with schoolbook columns
        # We need a base schoolbook decomposition
        U_sb = np.zeros((n * m, n * m * p))
        V_sb = np.zeros((m * p, n * m * p))
        W_sb = np.zeros((p * n, n * m * p))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        
        # Select R random columns from schoolbook to initialize
        # Or just the first R? Random might be better to escape local minima.
        # Let's try first R for simplicity, then random if it fails?
        # Actually, if R < 45, the first R columns are a subset.
        # Let's just use the first R columns.
        U = U_sb[:, :R].copy()
        V = V_sb[:, :R].copy()
        W = W_sb[:, :R].copy()
        
        # Add small random noise to break symmetry?
        # U += np.random.randn(*U.shape) * 1e-3
        # V += np.random.randn(*V.shape) * 1e-3
        # W += np.random.randn(*W.shape) * 1e-3
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            # T_unfold1 (nm x mp*pn) = U (Z_VW)^T
            # Z_VW (mp*pn x R), Z_VW[(b,o), c] = V[b,c] W[o,c]
            # We can compute this efficiently.
            # Z_VW = np.einsum('bc,oc->bo c', V, W).reshape(-1, R) ?
            # V is (mp, R), W is (pn, R).
            # Outer product of V and W for each c?
            # Z_VW[b, o, c] = V[b, c] * W[o, c]
            # Z_VW_flat = Z_VW.reshape(-1, R)
            Z_VW = np.einsum('bc,oc->boc', V, W).reshape(-1, R)
            # T_unfold1 = T.reshape(nm, -1)
            T_unfold1 = T.reshape(n * m, -1)
            # U = T_unfold1 @ pinv(Z_VW.T)
            # Use least squares: U = np.linalg.lstsq(Z_VW.T, T_unfold1.T, rcond=None)[0].T
            # Z_VW.T is (R, mp*pn). T_unfold1.T is (mp*pn, nm).
            # Result is (R, nm). Transpose to (nm, R).
            try:
                U_new = np.linalg.lstsq(Z_VW.T, T_unfold1.T, rcond=None)[0].T
            except:
                return None
            U = U_new
            
            # Update V
            # T_unfold2 (mp x nm*pn) = V (Z_UW)^T
            # Z_UW[(a,o), c] = U[a,c] W[o,c]
            Z_UW = np.einsum('ac,oc->aoc', U, W).reshape(-1, R)
            T_unfold2 = T.transpose(1, 0, 2).reshape(m * p, -1) # (mp, nm*pn)
            try:
                V_new = np.linalg.lstsq(Z_UW.T, T_unfold2.T, rcond=None)[0].T
            except:
                return None
            V = V_new
            
            # Update W
            # T_unfold3 (pn x nm*mp) = W (Z_UV)^T
            # Z_UV[(a,b), c] = U[a,c] V[b,c]
            Z_UV = np.einsum('ac,bc->abc', U, V).reshape(-1, R)
            T_unfold3 = T.transpose(2, 0, 1).reshape(p * n, -1) # (pn, nm*mp)
            try:
                W_new = np.linalg.lstsq(Z_UV.T, T_unfold3.T, rcond=None)[0].T
            except:
                return None
            W = W_new
            
            # Check convergence
            # err = np.linalg.norm(T - np.einsum('ac,bc,oc->abo', U, V, W))
            # This is expensive. Check every 10 iterations.
            if it % 10 == 0:
                T_recon = np.einsum('ac,bc,oc->abo', U, V, W)
                err = np.linalg.norm(T - T_recon)
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
        
        # Snap to half-integers
        U, V, W = snap_to_half_integers(U, V, W)
        
        # Verify
        if verify(U, V, W):
            return U, V, W
        
        return None

    # Try ranks from 45 downwards
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # We know 45 works. Let's try to find a better one.
    # Start from 45 and go down? Or start from a lower estimate?
    # Known lower bound for matrix mult rank is not trivial.
    # Let's just try 45, 44, 43... down to maybe 30.
    # If we find one, we return it.
    # If we don't, we return 45.
    
    # To save time, we can just verify the schoolbook solution first.
    # It is always valid.
    U_sb_full = np.zeros((n * m, best_rank))
    V_sb_full = np.zeros((m * p, best_rank))
    W_sb_full = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb_full[i * m + j, col] = 1.0
                V_sb_full[j * p + k, col] = 1.0
                W_sb_full[k * n + i, col] = 1.0
                col += 1
    if verify(U_sb_full, V_sb_full, W_sb_full):
        best_U, best_V, best_W = U_sb_full, V_sb_full, W_sb_full
    
    # Try to improve
    for R in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations?
        # The ALS above uses fixed initialization.
        # Let's try it.
        result = als_decompose(R, max_iters=50)
        if result:
            U, V, W = result
            best_rank = R
            best_U, best_V, best_W = U, V, W
            # If we find a lower rank, we can break or continue.
            # Let's break to save time, as lower ranks are harder to find.
            # Actually, if we find R, R-1 might be possible.
            # But given the time, let's break.
            break
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
