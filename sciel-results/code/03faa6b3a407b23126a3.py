import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    R = U.shape[1]
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(S, np.ones((n, m, p)), atol=tol):
        return False
    if not np.allclose(U * 2, np.round(U * 2), atol=1e-8):
        return False
    if not np.allclose(V * 2, np.round(V * 2), atol=1e-8):
        return False
    if not np.allclose(W * 2, np.round(W * 2), atol=1e-8):
        return False
    return True

def _als_step(U, V, W, n, m, p):
    """Perform one step of Alternating Least Squares."""
    R = U.shape[1]
    
    U_new = U.copy()
    for i in range(n):
        for j in range(m):
            M = np.zeros((p, R))
            for k in range(p):
                M[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                U_new[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
    U = U_new
    
    V_new = V.copy()
    for j in range(m):
        for k in range(p):
            M = np.zeros((n, R))
            for i in range(n):
                M[i, :] = U[i*m+j, :] * W[k*n+i, :]
            try:
                V_new[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
    V = V_new
    
    W_new = W.copy()
    for k in range(p):
        for i in range(n):
            M = np.zeros((m, R))
            for j in range(m):
                M[j, :] = U[i*m+j, :] * V[j*p+k, :]
            try:
                W_new[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
    W = W_new
    
    return U, V, W

def _try_rank_als(n, m, p, rank, seed, max_iters=200, tol=1e-10):
    """Try to find a rank-r decomposition using ALS."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_iters):
        U, V, W = _als_step(U, V, W, n, m, p)
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.sum((S - 1.0)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < tol:
            break
    
    if best_err < 1e-8:
        # Try snapping to half-integers
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        if _verify(U_round, V_round, W_round, n, m, p, tol=1e-8):
            return U_round, V_round, W_round
        
        # Try snapping to integers
        U_int = np.round(best_U)
        V_int = np.round(best_V)
        W_int = np.round(best_W)
        if _verify(U_int, V_int, W_int, n, m, p, tol=1e-8):
            return U_int, V_int, W_int
    
    return None, None, None

def _try_rank_gradient(n, m, p, rank, seed, max_iters=500, tol=1e-10):
    """Try to find a rank-r decomposition using gradient descent."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    lr = 0.01
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_iters):
        # Compute loss and gradients
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        loss = np.sum((S - 1.0)**2)
        
        if loss < best_err:
            best_err = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < tol:
            break
        
        # Compute gradients
        dS = 2 * (S - 1.0)
        
        # Gradient for U
        dU = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                for r in range(rank):
                    grad = 0.0
                    for k in range(p):
                        grad += dS[i,j,k] * V_3d[j,k,r] * W_3d[k,i,r]
                    dU[i*m+j, r] = grad
        U -= lr * dU
        
        # Gradient for V
        dV = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    grad = 0.0
                    for i in range(n):
                        grad += dS[i,j,k] * U_3d[i,j,r] * W_3d[k,i,r]
                    dV[j*p+k, r] = grad
        V -= lr * dV
        
        # Gradient for W
        dW = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    grad = 0.0
                    for j in range(m):
                        grad += dS[i,j,k] * U_3d[i,j,r] * V_3d[j,k,r]
                    dW[k*n+i, r] = grad
        W -= lr * dW
        
        # Decay learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.5
    
    if best_err < 1e-8:
        # Try snapping to half-integers
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        if _verify(U_round, V_round, W_round, n, m, p, tol=1e-8):
            return U_round, V_round, W_round
        
        # Try snapping to integers
        U_int = np.round(best_U)
        V_int = np.round(best_V)
        W_int = np.round(best_W)
        if _verify(U_int, V_int, W_int, n, m, p, tol=1e-8):
            return U_int, V_int, W_int
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Fallback: schoolbook
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_schoolbook
    
    # Try to find lower ranks using ALS and gradient descent
    ranks_to_try = list(range(38, 17, -1))  # Try from 38 down to 18
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        if R >= best_rank:
            continue
            
        found = False
        # Try ALS first
        for seed in range(50):
            if time.time() > deadline:
                break
            U, V, W = _try_rank_als(n, m, p, R, seed, max_iters=300)
            if U is not None:
                best_rank = R
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            continue
        
        # Try gradient descent
        for seed in range(20):
            if time.time() > deadline:
                break
            U, V, W = _try_rank_gradient(n, m, p, R, seed, max_iters=300)
            if U is not None:
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break
    
    def to_list(mat):
        return mat.tolist()
        
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
