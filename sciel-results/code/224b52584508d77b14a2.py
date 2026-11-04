"""Matrix multiplication 3x4x6 - structural decomposition + targeted optimization."""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-4):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    expected = np.ones((n, m, p))
    return np.max(np.abs(T - expected)) < tol

def _verify_fast(U, V, W, n, m, p, tol=1e-3):
    """Faster verification with looser tolerance."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    return np.max(np.abs(T - 1.0)) < tol

def _als_step(U, V, W, n, m, p, r, max_iter=100, tol=1e-10, deadline=None):
    """Alternating least squares for tensor decomposition."""
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Fix V, W -> solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A = V[j, :, :] * W[:, i, :]  # shape (p, r)
                try:
                    u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        if np.max(np.abs(U_new - U)) < tol:
            U = U_new
            break
        U = U_new
        
        # Fix U, W -> solve for V
        V_new = np.zeros_like(V)
        Ut = U.reshape(n, m, r)
        for j in range(m):
            for k in range(p):
                A = Ut[:, j, :] * W[k, :, :]  # shape (n, r)
                try:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
        
        # Fix U, V -> solve for W
        W_new = np.zeros_like(W)
        Vt = V.reshape(m, p, r)
        for k in range(p):
            for i in range(n):
                A = Ut[i, :, :] * Vt[:, k, :]  # shape (m, r)
                try:
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        if np.max(np.abs(W_new - W)) < tol:
            W = W_new
            break
        W = W_new
    
    return U, V, W

def _polish_scipy(U, V, W, n, m, p, r, max_iter=200, deadline=None):
    """Polish with scipy L-BFGS-B."""
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def loss_and_grad(x):
        Ux = x[:n*m*r].reshape(n, m, r)
        Vx = x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
        Wx = x[n*m*r + m*p*r :].reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', Ux, Vx, Wx)
        err = T - 1.0
        loss = np.sum(err ** 2)
        
        dU = np.einsum('ijk,jkc,kic->ijc', err, Vx, Wx) * 2
        dV = np.einsum('ijk,ijc,kic->jkc', err, Ux, Wx) * 2
        dW = np.einsum('ijk,ijc,jkc->kic', err, Ux, Vx) * 2
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad
    
    try:
        result = opt.minimize(loss_and_grad, x0, method='L-BFGS-B',
                              jac=True, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.fun < 1e-6:
            U_opt = result.x[:n*m*r].reshape(n*m, r)
            V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
            return U_opt, V_opt, W_opt, result.fun
    except:
        pass
    
    return U, V, W, np.inf

def _find_333_decomposition(deadline):
    """
    Try to find a low-rank decomposition for 3x3x3 matrix multiplication.
    Known optimal rank is 23, but we'll try from 23 upward.
    """
    n = m = p = 3
    for r in range(23, 28):
        if time.time() > deadline - 10:
            return None
        
        for restart in range(5):
            if time.time() > deadline - 5:
                return None
            
            seed = 1000 + restart * 73
            np.random.seed(seed)
            scale = 1.0 / np.sqrt(r)
            U = np.random.randn(n * m, r) * scale
            V = np.random.randn(m * p, r) * scale
            W = np.random.randn(p * n, r) * scale
            
            U, V, W = _als_step(U, V, W, n, m, p, r, max_iter=50, deadline=deadline)
            
            if _verify_fast(U, V, W, n, m, p, tol=1e-3):
                # Polish
                U, V, W, loss = _polish_scipy(U, V, W, n, m, p, r, max_iter=100, deadline=deadline)
                if _verify(U, V, W, n, m, p, tol=1e-4):
                    return U, V, W, r
            
            # Direct scipy from random
            np.random.seed(seed + 500)
            U2 = np.random.randn(n * m, r) * scale
            V2 = np.random.randn(m * p, r) * scale
            W2 = np.random.randn(p * n, r) * scale
            U2, V2, W2, loss2 = _polish_scipy(U2, V2, W2, n, m, p, r, max_iter=100, deadline=deadline)
            if _verify(U2, V2, W2, n, m, p, tol=1e-4):
                return U2, V2, W2, r
    
    return None

def _construct_from_333(U333, V333, W333, r333):
    """
    Construct 3x4x6 decomposition from 3x3x3 decomposition.
    
    The 3x4x6 tensor can be decomposed as two 3x4x3 blocks.
    Each 3x4x3 block = 3x3x3 part (rank r333) + outer product (rank 1).
    
    For 3x4x3:
    - U[i,j,c]: for j<3, use U333[i,j,c]; for j=3, use 0 except one column
    - V[j,k,c]: for j<3, use V333[j,k,c]; for j=3, use 0 except one column  
    - W[k,i,c]: same as W333[k,i,c]
    
    The outer product term (rank 1) handles the j=3 case:
    U[i,3,c_new] = 1, V[3,k,c_new] = 1, W[k,i,c_new] = 1
    """
    n, m, p = 3, 4, 3
    r = r333 + 1
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Copy 3x3x3 part
    U333_2d = U333.reshape(n, 3, r333)
    V333_2d = V333.reshape(3, 3, r333)
    W333_2d = W333.reshape(3, n, r333)
    
    for i in range(n):
        for j in range(3):
            for c in range(r333):
                U[i * m + j, c] = U333_2d[i, j, c]
                V[j * p + 0, c] = V333_2d[j, 0, c]  # k=0 placeholder
        # Actually need to do this properly for all k
    
    # Redo: proper embedding
    for i in range(n):
        for j in range(3):
            for c in range(r333):
                U[i * m + j, c] = U333_2d[i, j, c]
    
    for j in range(3):
        for k in range(3):
            for c in range(r333):
                V[j * p + k, c] = V333_2d[j, k, c]
    
    for k in range(3):
        for i in range(n):
            for c in range(r333):
                W[k * n + i, c] = W333_2d[k, i, c]
    
    # Outer product term for j=3
    c_new = r333
    for i in range(n):
        U[i * m + 3, c_new] = 1.0
    for k in range(p):
        V[3 * p + k, c_new] = 1.0
    for k in range(p):
        for i in range(n):
            W[k * n + i, c_new] = 1.0
    
    return U, V, W, r

def _try_direct_rank(n, m, p, r, deadline, max_restarts=8):
    """Try to find a rank-r decomposition directly using ALS + scipy."""
    for restart in range(max_restarts):
        if time.time() > deadline - 8:
            return None
        
        seed = 777 + restart * 131
        np.random.seed(seed)
        scale = 1.0 / np.sqrt(r)
        U = np.random.randn(n * m, r) * scale
        V = np.random.randn(m * p, r) * scale
        W = np.random.randn(p * n, r) * scale
        
        # ALS
        U, V, W = _als_step(U, V, W, n, m, p, r, max_iter=30, deadline=deadline)
        
        if _verify_fast(U, V, W, n, m, p, tol=1e-3):
            U, V, W, loss = _polish_scipy(U, V, W, n, m, p, r, max_iter=150, deadline=deadline)
            if _verify(U, V, W, n, m, p, tol=1e-4):
                return U, V, W
        
        # Direct scipy
        np.random.seed(seed + 300)
        U2 = np.random.randn(n * m, r) * scale
        V2 = np.random.randn(m * p, r) * scale
        W2 = np.random.randn(p * n, r) * scale
        U2, V2, W2, loss2 = _polish_scipy(U2, V2, W2, n, m, p, r, max_iter=150, deadline=deadline)
        if _verify(U2, V2, W2, n, m, p, tol=1e-4):
            return U2, V2, W2
    
    return None

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 140.0  # Safety margin before 150s
    n, m, p = DIMENSIONS
    
    # Fallback: trivial decomposition
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = n * m * p  # 72
    
    # Strategy 1: Try to build from 3x3x3 decomposition
    # 3x4x6 = two copies of 3x4x3, each = 3x3x3 + outer product
    # If we find rank-r 3x3x3, total rank = 2*(r+1)
    result_333 = _find_333_decomposition(_deadline)
    if result_333 is not None:
        U333, V333, W333, r333 = result_333
        # Build one 3x4x3 block
        U_block, V_block, W_block, r_block = _construct_from_333(U333, V333, W333, r333)
        
        # For 3x4x6, we need two such blocks (k=0..2 and k=3..5)
        # The k dimension splits into two groups of 3
        # U is same for both blocks (same i,j)
        # V splits: first 3 k's use V_block, next 3 k's use V_block
        # W: first block uses W_block for k=0..2, second for k=3..5
        
        # Actually, let's think more carefully.
        # The tensor T[i,j,k] for 3x4x6.
        # Split k into k1 in {0,1,2} and k2 in {3,4,5}.
        # For k1: T[i,j,k1] = sum_c U[i,j,c] * V[j,k1,c] * W[k1,i,c]  (first r_block terms)
        # For k2: T[i,j,k2] = sum_c U[i,j,c] * V[j,k2,c] * W[k2,i,c]  (next r_block terms)
        
        total_r = 2 * r_block
        U_full = np.zeros((n * m, total_r))
        V_full = np.zeros((m * p, total_r))
        W_full = np.zeros((p * n, total_r))
        
        # First block (k=0,1,2)
        for c in range(r_block):
            U_full[:, c] = U_block[:, c]
            for j in range(m):
                for k in range(3):
                    V_full[j * p + k, c] = V_block[j * p + k, c]
            for k in range(3):
                for i in range(n):
                    W_full[k * n + i, c] = W_block[k * n + i, c]
        
        # Second block (k=3,4,5) - same structure, offset in V and W
        for c in range(r_block):
            c2 = r_block + c
            U_full[:, c2] = U_block[:, c]
            for j in range(m):
                for k in range(3):
                    V_full[j * p + (k + 3), c2] = V_block[j * p + k, c]
            for k in range(3):
                for i in range(n):
                    W_full[(k + 3) * n + i, c2] = W_block[k * n + i, c]
        
        if _verify(U_full, V_full, W_full, n, m, p, tol=1e-4):
            best = {'U': U_full, 'V': V_full, 'W': W_full}
            best_rank = total_r
    
    # Strategy 2: Try direct optimization for ranks below current best
    if best_rank > 50:
        for r in range(min(best_rank - 1, 65), 48, -1):
            if time.time() > _deadline - 15:
                break
            
            result = _try_direct_rank(n, m, p, r, _deadline, max_restarts=4)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p, tol=1e-4):
                    best = {'U': U, 'V': V, 'W': W}
                    best_rank = r
                    break
    
    return {'U': best['U'].tolist(), 'V': best['V'].tolist(), 'W': best['W'].tolist()}
# EVOLVE-BLOCK-END
