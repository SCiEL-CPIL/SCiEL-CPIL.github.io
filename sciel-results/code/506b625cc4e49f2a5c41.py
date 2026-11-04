import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 20, 30, 24

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition (rank 120)
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    # Verify a decomposition
    def verify(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:, t], V[:, t])
            T_recon += np.einsum('ab,c->abc', uv, W[:, t])
        return np.max(np.abs(T_recon - T))

    # Try to find a lower-rank decomposition using ALS
    def try_als(target_rank, max_iters=200, tol=1e-12, seed=42):
        if time.time() > deadline - 10:
            return None, None, None, float('inf')
        
        np.random.seed(seed)
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        # Precompute flattened T
        T_flat_a = T.reshape(nm, -1)  # (nm, mp*pn)
        T_flat_b = T.reshape(mp, -1)  # (mp, nm*pn)
        T_flat_c = T.reshape(pn, -1)  # (pn, nm*mp)
        
        best_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Fix V, W: solve for U
            X = np.einsum('bt,ct->bct', V, W).reshape(target_rank, mp*pn).T
            for a in range(nm):
                U[a, :] = np.linalg.lstsq(X, T_flat_a[a, :], rcond=None)[0]
            
            # Fix U, W: solve for V
            X = np.einsum('at,ct->act', U, W).reshape(target_rank, nm*pn).T
            for b in range(mp):
                V[b, :] = np.linalg.lstsq(X, T_flat_b[b, :], rcond=None)[0]
            
            # Fix U, V: solve for W
            X = np.einsum('at,bt->abt', U, V).reshape(target_rank, nm*mp).T
            for c in range(pn):
                W[c, :] = np.linalg.lstsq(X, T_flat_c[c, :], rcond=None)[0]
            
            # Check error
            err = verify(U, V, W)
            if err < best_err:
                best_err = err
                if err < tol:
                    return U.copy(), V.copy(), W.copy(), err
        
        return U, V, W, best_err

    # Try partitioned approach with Strassen for 4x4*4x4 core
    def try_partitioned():
        """
        Partition: A (4x5) = [A1(4x4) | A2(4x1)]
                   B (5x6) = [B1(4x6); B2(1x6)]
        C = A1*B1 + A2*B2
        
        A1*B1: (4x4)*(4x6) -> can be split as (4x4*4x4) + (4x4*4x2)
        A2*B2: (4x1)*(1x6) -> rank 4 schoolbook
        
        For 4x4*4x4, use Strassen (rank 28 via 4 sub-Strassen)
        For 4x4*4x2, schoolbook rank 32
        Total: 28 + 32 + 4 = 64
        """
        # This is complex to implement correctly, so let's try a simpler partition
        # A (4x5) = [A1(3x5); A2(1x5)]
        # B (5x6) = [B1(5x6)]
        # C1 (3x6) = A1*B1, C2 (1x6) = A2*B1
        # C1: (3x5)*(5x6) -> rank 90 schoolbook
        # C2: (1x5)*(5x6) -> rank 30 schoolbook
        # Total: 120 (no improvement)
        
        # Let's try: A (4x5) = [A1(2x5); A2(2x5)]
        # C1 (2x6) = A1*B, C2 (2x6) = A2*B
        # Each: (2x5)*(5x6) -> rank 60 schoolbook
        # Total: 120 (no improvement)
        
        # The key insight is that we need to use sub-quadratic algorithms for the core.
        # Let's just try ALS with various target ranks.
        return None, None, None, float('inf')

    # Try various target ranks with ALS
    best_rank = n * m * p
    best_decomp = None
    best_err = float('inf')
    
    # First, get schoolbook as baseline
    U_sb, V_sb, W_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try target ranks from 60 to 119
    target_ranks = list(range(60, 120, 5)) + [119, 118, 117, 116, 115, 114, 113, 112, 111, 110]
    target_ranks = sorted(set(target_ranks))
    
    for tr in target_ranks:
        if time.time() > deadline - 30:
            break
        
        if tr >= best_rank:
            continue
        
        for seed in [42, 123, 777]:
            if time.time() > deadline - 10:
                break
            
            U, V, W, err = try_als(tr, max_iters=100, tol=1e-12, seed=seed)
            if err < 1e-10:
                if tr < best_rank:
                    best_rank = tr
                    best_decomp = (U.copy(), V.copy(), W.copy())
                    best_err = err
                break  # Found a valid decomposition for this rank, try lower
    
    if best_decomp is None:
        best_decomp = (U_sb, V_sb, W_sb)
        best_rank = n * m * p
    
    U, V, W = best_decomp
    
    # Final verification
    err = verify(U, V, W)
    if err > 1e-8:
        # Fallback to schoolbook
        U, V, W = schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
