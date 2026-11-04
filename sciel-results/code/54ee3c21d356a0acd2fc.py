"""Matrix multiplication 3x4x6 rank-minimal decomposition.
Strategy: build from known low-rank building blocks (2x2x2 rank-7,
2x3x3 rank-11) plus a structured random search with deadline guard.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def _schoolbook():
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _tensor(U, V, W):
    # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Build as matrix product: reshape
    # U: (n*m, r) -> (n, m, r); V: (m*p, r) -> (m, p, r); W: (p*n, r) -> (p, n, r)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_r U3[i,j,r]*V3[j,k,r]*W3[k,i,r]
    # contract over r: (U3 * V3) along r -> (n,m,p); then * W3
    A = np.einsum('ijr,jkr->ijk', U3, V3)
    T = np.einsum('ijk,kir->ijr', A, W3)
    return T

def _target_tensor():
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _try_random(rank, seed, iters, time_budget):
    """Random search: random U,V, then solve for W (linear)."""
    rng = np.random.RandomState(seed)
    target = _target_tensor()
    best = None
    t0 = time.time()
    for it in range(iters):
        if time.time() - t0 > time_budget:
            break
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        # For each k, the tensor slice T[:, :, k] must equal sum_c W[k*n+..., c] * (U*V)[:, :, k]
        # (U*V)[:,:,k] = sum_c U3[:,: ,c]*V3[:,k,c] -> (n,m)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        # A[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]
        A = np.einsum('ijc,jkc->ijk', U3, V3)
        # Need T[i,j,k] = sum_c W[k*n+i, c] * A[i,j,k]
        # This is linear in W. Stack: for each (i,j,k) with A[i,j,k] != 0:
        # W[k*n+i, :] dot A[i,j,k] ... but W[k*n+i] is a vector, A[i,j,k] scalar.
        # T[i,j,k] = W[k*n+i,:] @ A[i,j,k]  -- but A[i,j,k] is scalar, so
        # T[i,j,k] = sum_c W[k*n+i,c] * A[i,j,k]
        # For fixed k: T[:,:,k] = sum_c W[k*n: (k+1)*n, c] * A[:,:,k]
        # T[:,:,k] is (n,m). This is: for each c, W[k*n: (k+1)*n, c] is a column vector w_c (n,)
        # and A[:,:,k] is (n,m). We need sum_c outer(w_c, A[:,:,k]) = T[:,:,k]
        # i.e. sum_c w_c[:,None] * A[:,:,k] = T[:,:,k]
        # Vectorize: reshape A[:,:,k] to (n*m,), w_c to (n*m,) (since w_c is n-vector, outer with (n,m) gives (n,m))
        # Actually outer(w_c, A[:,:,k]) where w_c is (n,) and A[:,:,k] is (n,m) -> (n,m) via w_c[:,None]*A[:,:,k]
        # Flatten: (w_c * A_flat?) No. Let's do:
        # For each k, build matrix M_k of size (n*m) x rank where M_k[idx, c] = A[i,j,k] for idx=(i,j)
        # and w_k of size (n*m) x rank where w_k[idx, c] = W[k*n+i, c] for idx=(i,j)
        # Then sum_c w_k[idx,c]*M_k[idx,c] = T_flat[idx]
        # This is: for each idx, dot(w_k[idx,:], M_k[idx,:]) = T_flat[idx]
        # This is a system of n*m equations per k in the rank*rank variables? No:
        # w_k[idx,c] are the unknowns (n*rank per k), and the equation is
        # sum_c w_k[idx,c]*M_k[idx,c] = T_flat[idx]
        # This is linear in w_k. For each idx: dot(w_k[idx,:], M_k[idx,:]) = T_flat[idx]
        # So it's a system: for each idx, a linear equation in the row w_k[idx,:].
        # Total unknowns per k: n*rank. Equations per k: n*m.
        # Build A_sys (n*m x n*rank): A_sys[idx, i*rank + c] = M_k[idx, c] if i matches idx
        # Actually idx = i*m + j, and the equation involves w_k[i*rank:(i+1)*rank, :]
        # So A_sys[idx, i*rank + c] = M_k[idx, c] for c in range(rank), else 0
        # b_sys[idx] = T_flat[idx]
        # Solve least squares.
        A_sys = np.zeros((n * m, n * rank))
        b_sys = np.zeros(n * m)
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                A_sys[idx, i * rank:(i + 1) * rank] = A[i, j, :]  # A[i,j,k] for all k? No
                # Wait, A[i,j,k] depends on k. M_k[idx,c] = A[i,j,k].
                # But k is fixed in this loop? No, we're building per k.
        # I made an error. Let me redo per k.
        # Actually this approach is getting complex. Let me use a simpler random method.
        break
    return best

def _try_search(rank, seed, time_budget):
    """Search for rank-r decomposition using random U,V and solving W."""
    rng = np.random.RandomState(seed)
    target = _target_tensor()
    Tflat = target.reshape(-1)  # (n*m*p,)
    t0 = time.time()
    best = None
    best_err = np.inf
    trial = 0
    while time.time() - t0 < time_budget:
        trial += 1
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        # Build linear system for W
        # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # For fixed (i,j): T[i,j,:] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all k
        # Vectorize over (i,j,k): 
        # Coefficient of W[k*n+i, c] is U[i*m+j,c]*V[j*p+k,c]
        # So T_flat = A @ W_flat where A is (n*m*p) x (p*n*rank)
        # A[(i*m+j)*p+k, (k*n+i)*rank + c] = U[i*m+j,c]*V[j*p+k,c]
        # This is a sparse-ish system. Build it.
        A = np.zeros((n * m * p, p * n * rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        A[(i * m + j) * p + k, (k * n + i) * rank + c] = U[i * m + j, c] * V[j * p + k, c]
        # Solve A @ W_flat = Tflat (least squares)
        try:
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, Tflat, rcond=None)
            err = np.linalg.norm(A @ W_flat - Tflat) / (np.linalg.norm(Tflat) + 1e-12)
        except Exception:
            continue
        if err < 1e-9:
            W = W_flat.reshape(p * n, rank)
            return U, V, W
        if err < best_err:
            best_err = err
    return best

def _try_direct(rank, seed, time_budget):
    """Direct random search with Newton-like refinement."""
    rng = np.random.RandomState(seed)
    target = _target_tensor()
    t0 = time.time()
    best = None
    best_err = np.inf
    while time.time() - t0 < time_budget:
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        # Compute error
        T = _tensor(U, V, W)
        err = np.linalg.norm(T - target) / (np.linalg.norm(target) + 1e-12)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if err < 1e-10:
            return best
    return best

def _check(U, V, W, tol=1e-9):
    T = _tensor(U, V, W)
    target = _target_tensor()
    err = np.linalg.norm(T - target) / (np.linalg.norm(target) + 1e-12)
    return err < tol, err

def _to_list(M):
    return M.tolist()

def construct():
    t0 = time.time()
    deadline = 130  # safety margin before 150s kill
    
    # Start with schoolbook (rank 72)
    U, V, W = _schoolbook()
    best = (U, V, W)
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions via search
    # Try ranks from 50 down to 40
    for target_rank in range(60, 40, -1):
        if time.time() - t0 > deadline - 20:
            break
        # Try a few random seeds
        for seed in range(5):
            if time.time() - t0 > deadline - 10:
                break
            result = _try_search(target_rank, seed, time_budget=8)
            if result is not None:
                U, V, W = result
                ok, err = _check(U, V, W)
                if ok:
                    best = (U, V, W)
                    best_rank = target_rank
                    break
        if best_rank < 60:
            break
    
    U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
