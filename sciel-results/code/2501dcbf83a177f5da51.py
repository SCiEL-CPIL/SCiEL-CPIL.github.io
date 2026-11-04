"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 6]

def _construct_target_tensor(n, m, p):
    """Construct the target multiplication tensor T."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, T, tol=1e-8):
    """
    Verify if U, V, W decompose T.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _try_rational_snapping(U, V, W, T, max_denom=4):
    """
    Try to snap the float solution to rational numbers with small denominators.
    Returns True if a snapped version verifies exactly, and updates U, V, W in place.
    """
    candidates = [
        (1, 1.0),
        (2, 0.5),
        (4, 0.25)
    ]
    
    for denom, step in candidates:
        U_snap = np.round(U / step) * step
        V_snap = np.round(V / step) * step
        W_snap = np.round(W / step) * step
        
        if _verify_fast(U_snap, V_snap, W_snap, T, tol=1e-9):
            U[:] = U_snap
            V[:] = V_snap
            W[:] = W_snap
            return True
            
    return False

def _als_polish(U, V, W, T, n, m, p, iterations=10, tol=1e-10):
    """
    Polish the solution using Alternating Least Squares.
    Uses vectorized operations for speed.
    """
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    prev_loss = np.inf
    
    for it in range(iterations):
        # Solve for U: min || T.reshape(nm, mp*pn) - U @ (V⊗W).T ||_F
        # (V⊗W) has shape (mp*pn, r)
        V_flat = np.kron(V, np.ones((pn, 1))).reshape(mp*pn, r)  # not quite right
        # Better: construct properly
        VW_mat = np.zeros((mp*pn, r))
        for b in range(mp):
            for c in range(pn):
                VW_mat[b*pn+c, :] = V[b, :] * W[c, :]
        
        T_mat = T.reshape(nm, mp*pn)
        try:
            U_new = T_mat @ VW_mat @ np.linalg.pinv(VW_mat.T @ VW_mat)
        except np.linalg.LinAlgError:
            U_new = U
        
        # Solve for V
        UW_mat = np.zeros((nm*pn, r))
        for a in range(nm):
            for c in range(pn):
                UW_mat[a*pn+c, :] = U_new[a, :] * W[c, :]
        
        T_mat_v = T.reshape(mp, nm*pn)
        try:
            V_new = T_mat_v @ UW_mat @ np.linalg.pinv(UW_mat.T @ UW_mat)
        except np.linalg.LinAlgError:
            V_new = V
        
        # Solve for W
        UV_mat = np.zeros((nm*mp, r))
        for a in range(nm):
            for b in range(mp):
                UV_mat[a*mp+b, :] = U_new[a, :] * V_new[b, :]
        
        T_mat_w = T.reshape(pn, nm*mp)
        try:
            W_new = T_mat_w @ UV_mat @ np.linalg.pinv(UV_mat.T @ UV_mat)
        except np.linalg.LinAlgError:
            W_new = W
        
        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
        loss = np.sum((T - recon)**2)
        
        U, V, W = U_new, V_new, W_new
        
        if abs(prev_loss - loss) < tol and loss < 1e-8:
            break
        prev_loss = loss

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = _construct_target_tensor(n, m, p)
    
    # Schoolbook fallback (Rank 60)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    np.random.seed(42)
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try ranks from 30 to 59
    for rank in range(30, best_rank):
        if time.time() > deadline - 30:
            break
            
        found = False
        max_restarts = 15
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Adam Optimizer
            v = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
            m_ = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
            beta1, beta2 = 0.9, 0.999
            lr = 0.01
            t = 0
            best_loss = np.inf
            
            for step in range(300):
                if time.time() > deadline - 5:
                    break
                    
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T - recon
                loss = np.sum(R**2)
                
                if loss < best_loss:
                    best_loss = loss
                    if loss < 1e-10:
                        break
                        
                # Gradients
                gU = -np.einsum('abc,br,cr->ar', R, V, W)
                gV = -np.einsum('abc,ar,cr->br', R, U, W)
                gW = -np.einsum('abc,ar,br->cr', R, U, V)
                
                grads = [gU, gV, gW]
                params = [U, V, W]
                
                t += 1
                for i in range(3):
                    m_[i] = beta1 * m_[i] + (1 - beta1) * grads[i]
                    v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
                    m_hat = m_[i] / (1 - beta1 ** t)
                    v_hat = v[i] / (1 - beta2 ** t)
                    params[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                    
                U = params[0]
                V = params[1]
                W = params[2]
                
                if step % 100 == 0 and step > 0:
                    lr *= 0.5
                
                if step % 50 == 0 and loss < 1e-6:
                    if _verify_fast(U, V, W, T, tol=1e-8):
                        found = True
                        break
                        
            if found:
                # Polish with ALS
                U, V, W = _als_polish(U, V, W, T, n, m, p, iterations=5)
                if _verify_fast(U, V, W, T, tol=1e-8):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    _try_rational_snapping(U, V, W, T)
                    break

        if found:
            break

    U_final = best_U
    V_final = best_V
    W_final = best_W
    
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
