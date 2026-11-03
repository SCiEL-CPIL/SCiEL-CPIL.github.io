"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]
DEADLINE = 150.0

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for c in range(rank):
        T += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    T_true = _build_tensor(n, m, p)
    return np.allclose(T, T_true, atol=1e-9)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _compute_error(U, V, W, T_true):
    """Compute squared error using vectorized ops."""
    n_m, r = U.shape
    m_p = V.shape[0]
    n_p = W.shape[0]
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Reshape: for each r, outer product
    T = np.zeros((n_m, m_p, n_p), dtype=np.float64)
    for c in range(r):
        T += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.sum((T - T_true)**2)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE - 5
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    T_true = _build_tensor(n, m, p)
    
    # Try to find lower-rank decompositions using ALS (Alternating Least Squares)
    def _als_decompose(r, n_iters=200, seed=0):
        """Try ALS to find rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(n*p, r) * 0.5
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each a, solve least squares
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
            # Let M[a] be a (m*p*n_p) x r matrix where M[a][b*n_p+c, r] = V[b,r]*W[c,r]
            # Then T[a].reshape(-1) = M[a] @ U[a]
            
            # Build M once (doesn't depend on a)
            M = np.zeros((m*p*n*p, r), dtype=np.float64)
            for b in range(m*p):
                for c in range(n*p):
                    for r_idx in range(r):
                        M[b*(n*p)+c, r_idx] = V[b, r_idx] * W[c, r_idx]
            
            # For each a, solve: M @ U[a] = T_true[a].reshape(-1)
            for a in range(n*m):
                rhs = T_true[a].reshape(-1)
                # Least squares: U[a] = pinv(M) @ rhs
                U[a] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            # Update V similarly
            M2 = np.zeros((n*m*n*p, r), dtype=np.float64)
            for a in range(n*m):
                for c in range(n*p):
                    for r_idx in range(r):
                        M2[a*(n*p)+c, r_idx] = U[a, r_idx] * W[c, r_idx]
            for b in range(m*p):
                rhs = T_true[:, b, :].reshape(-1)
                V[b] = np.linalg.lstsq(M2, rhs, rcond=None)[0]
            
            # Update W
            M3 = np.zeros((n*m*m*p, r), dtype=np.float64)
            for a in range(n*m):
                for b in range(m*p):
                    for r_idx in range(r):
                        M3[a*(m*p)+b, r_idx] = U[a, r_idx] * V[b, r_idx]
            for c in range(n*p):
                rhs = T_true[:, :, c].reshape(-1)
                W[c] = np.linalg.lstsq(M3, rhs, rcond=None)[0]
            
            # Compute error
            T_est = np.zeros((n*m, m*p, n*p), dtype=np.float64)
            for r_idx in range(r):
                T_est += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            err = np.sum((T_est - T_true)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-12:
                break
        
        if best_err < 1e-9:
            # Try to round to half-integers
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            T_check = np.zeros((n*m, m*p, n*p), dtype=np.float64)
            for r_idx in range(r):
                T_check += np.einsum('a,b,c->abc', U_r[:, r_idx], V_r[:, r_idx], W_r[:, r_idx])
            if np.allclose(T_check, T_true, atol=1e-9):
                return (U_r.tolist(), V_r.tolist(), W_r.tolist())
            # Also try rounding to integers
            U_i = np.round(best_U)
            V_i = np.round(best_V)
            W_i = np.round(best_W)
            T_check2 = np.zeros((n*m, m*p, n*p), dtype=np.float64)
            for r_idx in range(r):
                T_check2 += np.einsum('a,b,c->abc', U_i[:, r_idx], V_i[:, r_idx], W_i[:, r_idx])
            if np.allclose(T_check2, T_true, atol=1e-9):
                return (U_i.tolist(), V_i.tolist(), W_i.tolist())
        
        return None
    
    # Try ranks below schoolbook
    for r in range(best_rank - 1, max(best_rank - 30, 20), -1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            result = _als_decompose(r, n_iters=100, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                best_rank = r
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                found = True
                break
        
        if found:
            continue  # try even lower
        else:
            break  # this rank didn't work, stop
    
    return best_result
# EVOLVE-BLOCK-END
