"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 5, 6]

def _construct_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, T, tol=1e-8):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _als_solve(U, V, W, T, n, m, p, max_iters=30, tol=1e-10):
    """ALS polish with vectorized solves."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    T_mat = T.reshape(nm, mp*pn)
    
    for it in range(max_iters):
        # Solve for U
        VW_mat = np.zeros((mp*pn, r))
        for b in range(mp):
            for c in range(pn):
                VW_mat[b*pn+c, :] = V[b, :] * W[c, :]
        try:
            U_new = np.linalg.lstsq(VW_mat, T_mat.T, rcond=None)[0].T
        except:
            U_new = U
        # Solve for V
        UW_mat = np.zeros((nm*pn, r))
        for a in range(nm):
            for c in range(pn):
                UW_mat[a*pn+c, :] = U_new[a, :] * W[c, :]
        T_mat_v = T.reshape(mp, nm*pn)
        try:
            V_new = np.linalg.lstsq(UW_mat, T_mat_v.T, rcond=None)[0].T
        except:
            V_new = V
        # Solve for W
        UV_mat = np.zeros((nm*mp, r))
        for a in range(nm):
            for b in range(mp):
                UV_mat[a*mp+b, :] = U_new[a, :] * V_new[b, :]
        T_mat_w = T.reshape(pn, nm*mp)
        try:
            W_new = np.linalg.lstsq(UV_mat, T_mat_w.T, rcond=None)[0].T
        except:
            W_new = W
        U, V, W = U_new, V_new, W_new
    return U, V, W

def _structured_init(n, m, p, rank, seed):
    """Initialize exploiting tensor structure."""
    np.random.seed(seed)
    nm, mp, pn = n*m, m*p, p*n
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    total = len(triples)
    if rank <= total:
        for r in range(rank):
            start = int(r * total / rank)
            end = int((r + 1) * total / rank)
            if r == rank - 1:
                end = total
            for idx in range(start, end):
                i, j, k = triples[idx]
                U[i*m+j, r] += 1.0 / np.sqrt(end - start)
                V[j*p+k, r] += 1.0 / np.sqrt(end - start)
                W[k*n+i, r] += 1.0 / np.sqrt(end - start)
    
    U += np.random.randn(nm, rank) * 0.05
    V += np.random.randn(mp, rank) * 0.05
    W += np.random.randn(pn, rank) * 0.05
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = _construct_target_tensor(n, m, p)
    
    # Schoolbook fallback (rank 60)
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
    
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Try ranks from 30 up to 59
    for rank in range(30, best_rank):
        if time.time() > deadline - 30:
            break
        
        max_restarts = max(5, 25 - rank // 2)
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + restart * 7919)
            # Random init with structure
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Adam optimizer
            m_ = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
            v = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
            beta1, beta2 = 0.9, 0.999
            lr = 0.01
            t = 0
            best_loss = np.inf
            best_U_c, best_V_c, best_W_c = U.copy(), V.copy(), W.copy()
            
            for step in range(600):
                if time.time() > deadline - 10:
                    break
                    
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T - recon
                loss = np.sum(R**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U_c, best_V_c, best_W_c = U.copy(), V.copy(), W.copy()
                    if loss < 1e-12:
                        break
                
                gU = -np.einsum('abc,br,cr->ar', R, V, W)
                gV = -np.einsum('abc,ar,cr->br', R, U, W)
                gW = -np.einsum('abc,ar,br->cr', R, U, V)
                
                t += 1
                for i, (g, param) in enumerate(zip([gU, gV, gW], [U, V, W])):
                    m_[i] = beta1 * m_[i] + (1 - beta1) * g
                    v[i] = beta2 * v[i] + (1 - beta2) * (g ** 2)
                    m_hat = m_[i] / (1 - beta1 ** t)
                    v_hat = v[i] / (1 - beta2 ** t)
                    param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                
                if step % 100 == 0 and step > 0:
                    lr *= 0.75
                
                if step % 50 == 0 and loss < 1e-8:
                    if _verify_fast(U, V, W, T, tol=1e-8):
                        found = True
                        break
            
            if not found and best_loss < 1e-4:
                U, V, W = best_U_c, best_V_c, best_W_c
                U, V, W = _als_solve(U, V, W, T, n, m, p, max_iters=15)
                if _verify_fast(U, V, W, T, tol=1e-8):
                    found = True
            
            if found:
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if found:
            break
    
    U_final, V_final, W_final = best_U, best_V, best_W
    return {
        "U": [[float(x) for x in row] for row in U_final],
        "V": [[float(x) for x in row] for row in V_final],
        "W": [[float(x) for x in row] for row in W_final]
    }
