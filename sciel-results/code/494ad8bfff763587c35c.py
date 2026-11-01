"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from lower bound upward
    # Lower bound: max(nm, mp, pn) = max(4, 26, 26) = 26
    lower_bound = max(nm, mp, pn)
    
    best_result = None
    
    for rank in range(lower_bound, nm * mp):  # up to schoolbook
        if time.time() > deadline - 30:
            break
        
        success = _try_decompose(T, rank, nm, mp, pn, n, m, p, deadline)
        if success is not None:
            U, V, W = success
            # Verify exactness
            if _verify(T, U, V, W, nm, mp, pn):
                best_result = (U, V, W, rank)
                break
        if time.time() > deadline - 30:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = nm * mp
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
        best_result = (U, V, W, rank)
    
    U, V, W, rank = best_result
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _try_decompose(T, rank, nm, mp, pn, n, m, p, deadline):
    """Try to find a rank-`rank` decomposition of T via alternating least squares."""
    np.random.seed(42)
    
    # Initialize with random matrices
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    # Scale to have similar magnitude to T
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    max_iters = 50
    for it in range(max_iters):
        if time.time() > deadline - 30:
            return None
        
        # Update U: for each a, U[a,:] should satisfy sum_b,c T[a,b,c] = sum_r U[a,r] * (V[:,r] * W[:,r])[b,c]
        # Reshape: T_a = T[a,:,:] shape (mp, pn). We need U[a,:] such that
        # T_a ≈ sum_r U[a,r] * V[:,r] ⊗ W[:,r]
        # This is: T_a.reshape(mp*pn) ≈ V_flat @ (U[a,:] ⊗ W_flat) ... 
        # Actually: sum_r U[a,r] * (V[:,r] * W[:,r]) where V[:,r]*W[:,r] is element-wise product reshaped
        # Let M_a = T[a,:,:].reshape(mp*pn, 1)
        # Let A_a = np.column_stack([V[:,r] * W[:,r] for r in range(rank)]).reshape(mp*pn, rank)
        # Then U[a,:] = argmin ||M_a - A_a @ U[a,:]||^2
        for a in range(nm):
            if time.time() > deadline - 30:
                return None
            M_a = T[a, :, :].reshape(-1)
            # A_a[b] = sum_r U[a,r] * V[b,r] * W[b,r]... no
            # The reconstruction is: sum_r U[a,r] * V[:,r] ⊗ W[:,r]
            # So for fixed a, we want: M_a ≈ sum_r U[a,r] * (V[:,r] ⊗ W[:,r]).flatten()
            A_a = np.zeros((mp * pn, rank))
            for r in range(rank):
                A_a[:, r] = (V[:, r] * W[:, r]).reshape(-1) if False else np.outer(V[:, r], W[:, r]).reshape(-1)
            # Solve least squares: U[a,:] = argmin ||M_a - A_a @ u||^2
            try:
                u, _, _, _ = np.linalg.lstsq(A_a, M_a, rcond=None)
                U[a, :] = u
            except:
                return None
        
        # Update V similarly
        for b in range(mp):
            if time.time() > deadline - 30:
                return None
            M_b = T[:, b, :].reshape(-1)  # shape (nm * pn,)
            A_b = np.zeros((nm * pn, rank))
            for r in range(rank):
                A_b[:, r] = np.outer(U[:, r], W[:, r]).reshape(-1)
            try:
                v, _, _, _ = np.linalg.lstsq(A_b, M_b, rcond=None)
                V[b, :] = v
            except:
                return None
        
        # Update W similarly
        for c in range(pn):
            if time.time() > deadline - 30:
                return None
            M_c = T[:, :, c].reshape(-1)  # shape (nm * mp,)
            A_c = np.zeros((nm * mp, rank))
            for r in range(rank):
                A_c[:, r] = np.outer(U[:, r], V[:, r]).reshape(-1)
            try:
                w, _, _, _ = np.linalg.lstsq(A_c, M_c, rcond=None)
                W[c, :] = w
            except:
                return None
    
    # Check if we got a good solution
    recon = np.zeros((nm, mp, pn))
    for r in range(rank):
        recon += np.outer(U[:, r], V[:, r])[:, :, None] * W[None, None, r]
    
    err = np.linalg.norm(recon - T) / np.linalg.norm(T)
    if err < 1e-10:
        return (U, V, W)
    return None

def _verify(T, U, V, W, nm, mp, pn):
    """Verify that U, V, W exactly decompose T."""
    recon = np.zeros((nm, mp, pn))
    for r in range(U.shape[1]):
        recon += np.outer(U[:, r], V[:, r])[:, :, None] * W[None, None, r]
    err = np.linalg.norm(recon - T) / (np.linalg.norm(T) + 1e-30)
    return err < 1e-10
# EVOLVE-BLOCK-END

