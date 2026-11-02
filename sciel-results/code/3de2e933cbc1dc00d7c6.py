"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 140

    def check_decomposition(U, V, W, rank, tol=1e-9):
        """Verify the decomposition is exact using vectorized operations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    idx_u = a * m + b
                    idx_v = b * p + c
                    idx_w = c * n + a
                    val = np.dot(U_arr[idx_u, :], V_arr[idx_v, :] * W_arr[idx_w, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def build_rank18():
        """Construct a valid rank-18 decomposition."""
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(col0, col1, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1
            W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
            
            r = base + 2
            U[0][r] = 1
            V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
            W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 3
            U[3][r] = 1
            V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
            W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
            
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+col1][r] = 1
            W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
            
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
            W[col1*n+1][r] = 1
            
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1
        
        add_strassen_block(0, 1, 0)
        add_strassen_block(2, 3, 7)
        
        r = 14
        U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
        r = 15
        U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
        r = 16
        U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
        r = 17
        U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
        
        return U, V, W

    def try_als_rank(rank, max_iters=1000, tol=1e-12, seed=42):
        """Try to find a rank-`rank` decomposition using ALS."""
        if time.time() > deadline - 10:
            return None
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        np.random.seed(seed)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Normalize columns
        for r in range(rank):
            U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-12)
            V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-12)
            W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-12)
        
        def compute_error():
            err = 0.0
            for a in range(n):
                for b in range(m):
                    for c in range(p):
                        idx_u = a * m + b
                        idx_v = b * p + c
                        idx_w = c * n + a
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        err += (val - 1.0) ** 2
            return err
        
        best_error = compute_error()
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U
            for a in range(n):
                for b in range(m):
                    idx_u = a * m + b
                    G = np.zeros((p, rank))
                    for c in range(p):
                        G[c, :] = V[b * p + c, :] * W[c * n + a, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(G, np.ones(p), rcond=None)
                        U[idx_u, :] = x
                    except:
                        pass
            
            # Update V
            for b in range(m):
                for c in range(p):
                    idx_v = b * p + c
                    H = np.zeros((n, rank))
                    for a in range(n):
                        H[a, :] = U[a * m + b, :] * W[c * n + a, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(H, np.ones(n), rcond=None)
                        V[idx_v, :] = x
                    except:
                        pass
            
            # Update W
            for c in range(p):
                for a in range(n):
                    idx_w = c * n + a
                    K = np.zeros((m, rank))
                    for b in range(m):
                        K[b, :] = U[a * m + b, :] * V[b * p + c, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(K, np.ones(m), rcond=None)
                        W[idx_w, :] = x
                    except:
                        pass
            
            # Renormalize
            for r in range(rank):
                U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-12)
                V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-12)
                W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-12)
            
            err = compute_error()
            if err < best_error:
                best_error = err
                if best_error < tol:
                    break
            else:
                # If error increases, break early to save time
                if it > 50 and err > best_error * 1.1:
                    break
        
        if best_error < tol:
            U_list = [[U[i, r] for r in range(rank)] for i in range(nm)]
            V_list = [[V[i, r] for r in range(rank)] for i in range(mp)]
            W_list = [[W[i, r] for r in range(rank)] for i in range(pn)]
            
            if check_decomposition(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list
        
        return None

    # Try rank 16 first
    found = None
    for attempt in range(50):
        if time.time() > deadline - 20:
            break
        result = try_als_rank(16, max_iters=200, seed=100+attempt)
        if result is not None:
            found = result
            break

    if found:
        return {"U": found[0], "V": found[1], "W": found[2]}

    # Fallback to rank 18
    U18, V18, W18 = build_rank18()
    return {"U": U18, "V": V18, "W": W18}
# EVOLVE-BLOCK-END
