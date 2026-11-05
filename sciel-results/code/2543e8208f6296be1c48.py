"""Matrix multiplication 3,3,6 rank-minimal exact decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    start_time = time.time()
    deadline = 120.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = 3, 3, 6
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if not U or not V or not W:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def snap_and_verify(U, V, W):
        """Try to snap to half-integers and verify."""
        # Try half-integer snap first
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_r.tolist()
        if verify(U_list, V_list, W_list):
            return (U_list, V_list, W_list)
        
        # Try integer snap
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_r.tolist()
        if verify(U_list, V_list, W_list):
            return (U_list, V_list, W_list)
        
        return None

    def try_rank(r, n_restarts=10, n_steps=100):
        """Try to find a rank-r decomposition via random restarts + local optimization."""
        if time.time() > deadline - 5:
            return None
        best_err = float('inf')
        best_U = best_V = best_W = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            
            for step in range(n_steps):
                if time.time() > deadline - 5:
                    break
                
                # Compute residuals E[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1
                # E = einsum('ijr, jkr, kir -> ijk', U_r, V_r, W_r) - 1
                E = np.einsum('ijr, jkr, kir -> ijk', U_r, V_r, W_r) - 1.0
                err = np.sum(E**2)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-12:
                    break
                
                # Gradient step
                lr = 0.05 / (1 + step * 0.02)
                
                # G_U[i, j, r] = sum_k E[i, j, k] V[j, k, r] W[k, i, r]
                G_U = np.einsum('ijk, jkr, kir -> ijr', E, V_r, W_r)
                # G_V[j, k, r] = sum_i E[i, j, k] U[i, j, r] W[k, i, r]
                G_V = np.einsum('ijk, ijr, kir -> jkr', E, U_r, W_r)
                # G_W[k, i, r] = sum_j E[i, j, k] U[i, j, r] V[j, k, r]
                G_W = np.einsum('ijk, ijr, jkr -> kir', E, U_r, V_r)
                
                U_r -= lr * G_U
                V_r -= lr * G_V
                W_r -= lr * G_W
                
                U = U_r.reshape(n * m, r)
                V = V_r.reshape(m * p, r)
                W = W_r.reshape(p * n, r)
                
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10:
            result = snap_and_verify(best_U, best_V, best_W)
            if result is not None:
                return result
        return None

    # Try ranks from low to high
    # Prioritize lower ranks with fewer restarts, higher ranks (closer to 54) with more restarts
    for r in range(14, 54):
        if time.time() > deadline - 10:
            break
        
        # Heuristic: more effort for ranks closer to 54, as they are more likely to exist
        if r < 40:
            res = try_rank(r, n_restarts=5, n_steps=50)
        else:
            res = try_rank(r, n_restarts=20, n_steps=200)
            
        if res is not None:
            U_list, V_list, W_list = res
            return {"U": U_list, "V": V_list, "W": W_list}

    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
