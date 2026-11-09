"""Matrix multiplication 2x5 x 5x5 - rank-minimal tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n*m, m*p, p*n  # 10, 25, 10

def _build_tensor():
    """T[a,b,c] where a=(i,j), b=(j',k), c=(k',i'); T = delta_{j,j'}*delta_{k,k'}."""
    T = np.zeros((NM, MP, PN), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            a = i*m + j
            for jp in range(m):   # j'
                for k in range(p):
                    b = jp*p + k
                    for kp in range(p):  # k'
                        for ip in range(n):  # i'
                            c = kp*n + ip
                            if jp == j and kp == k:
                                T[a, b, c] = 1.0
    return T

TENSOR = _build_tensor()

def _verify_fast(U, V, W, tol=1e-9):
    """Fast vectorized verification: sum_r U[a,r]V[b,r]W[c,r] == T[a,b,c]."""
    R = U.shape[1]
    # Compute UV[a,b,r] = U[a,r]*V[b,r]
    UV = np.einsum('ar,br->abr', U, V)
    # Result[a,b,c] = sum_r UV[a,b,r]*W[c,r]
    result = np.einsum('abr,cr->abc', UV, W)
    return np.allclose(result, TENSOR, atol=tol)

def _verify_exact(U, V, W):
    """Exact verification using Fractions (for half-integer values)."""
    R = U.shape[1]
    Uq = [[Fraction(round(2*x), 2) for x in row] for row in U]
    Vq = [[Fraction(round(2*x), 2) for x in row] for row in V]
    Wq = [[Fraction(round(2*x), 2) for x in row] for row in W]
    for a in range(NM):
        for b in range(MP):
            prod = [Uq[a][r] * Vq[b][r] for r in range(R)]
            for c in range(PN):
                val = sum(prod[r] * Wq[c][r] for r in range(R))
                if val != TENSOR[a, b, c]:
                    return False
    return True

def _als(R, iters=200, seed=0, tol=1e-8, deadline=None):
    """Alternating least squares for rank-R decomposition with time guard."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((NM, R))
    V = rng.standard_normal((MP, R))
    W = rng.standard_normal((PN, R))
    U /= (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
    V /= (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        # Update U
        M = np.einsum('br,cr->bcr', V, W).reshape(MP*PN, R)
        t = TENSOR.reshape(NM, MP*PN)
        U_new = np.zeros((NM, R))
        for a in range(NM):
            sol, *_ = np.linalg.lstsq(M, t[a], rcond=None)
            U_new[a] = sol
        U = U_new
        # Update V
        M2 = np.einsum('ar,cr->acr', U, W).reshape(NM*PN, R)
        V_new = np.zeros((MP, R))
        for b in range(MP):
            t2b = TENSOR[:, b, :].reshape(NM*PN)
            sol, *_ = np.linalg.lstsq(M2, t2b, rcond=None)
            V_new[b] = sol
        V = V_new
        # Update W
        Mc = np.einsum('ar,br->abr', U, V).reshape(NM*MP, R)
        W_new = np.zeros((PN, R))
        for c in range(PN):
            tc = TENSOR[:, :, c].reshape(NM*MP)
            sol, *_ = np.linalg.lstsq(Mc, tc, rcond=None)
            W_new[c] = sol
        W = W_new
    return U, V, W

def _local_search(U, V, W, steps=50, lr=0.1, deadline=None):
    """Local search: try small perturbations and accept if residual decreases."""
    R = U.shape[1]
    def residual(U, V, W):
        UV = np.einsum('ar,br->abr', U, V)
        result = np.einsum('abr,cr->abc', UV, W)
        return np.sum((result - TENSOR)**2)
    
    cur_res = residual(U, V, W)
    rng = np.random.default_rng(42)
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        # Random perturbation
        dU = rng.standard_normal((NM, R)) * lr
        dV = rng.standard_normal((MP, R)) * lr
        dW = rng.standard_normal((PN, R)) * lr
        new_res = residual(U + dU, V + dV, W + dW)
        if new_res < cur_res:
            U, V, W = U + dU, V + dV, W + dW
            cur_res = new_res
    return U, V, W

def _round_half(U, V, W):
    """Round to nearest half-integer."""
    return (2*U).round()/2.0, (2*V).round()/2.0, (2*W).round()/2.0

def _to_list(U):
    """Convert numpy array to list of lists with clean half-integers."""
    def clean(x):
        hx = round(2*x)
        if abs(hx - round(hx)) < 1e-6:
            return int(hx)/2
        return float(x)
    return [[clean(x) for x in row] for row in U]

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    SAFETY = 5.0
    deadline = t0 + DEADLINE
    
    # Schoolbook fallback (rank 50)
    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(NM)]
        V = [[0]*rank for _ in range(MP)]
        W = [[0]*rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    U0, V0, W0, r0 = schoolbook()
    best = (U0, V0, W0)
    best_rank = r0
    
    # Try ranks from 30 to 49
    for R in range(30, 50):
        if time.time() - t0 > DEADLINE - SAFETY:
            break
        for seed in range(0, 8):
            if time.time() - t0 > DEADLINE - SAFETY:
                break
            try:
                U, V, W = _als(R, iters=40, seed=seed, deadline=deadline - SAFETY)
                # Local search polish
                U, V, W = _local_search(U, V, W, steps=20, lr=0.05, deadline=deadline - SAFETY)
                # Round to half-integers
                U2, V2, W2 = _round_half(U, V, W)
                # Fast verify
                if _verify_fast(U2, V2, W2, tol=1e-9):
                    # Exact verify
                    if _verify_exact(U2, V2, W2):
                        best = (U2, V2, W2)
                        best_rank = R
                        break
            except Exception:
                pass
    
    U, V, W = best
    Uf = _to_list(U)
    Vf = _to_list(V)
    Wf = _to_list(W)
    return {"U": Uf, "V": Vf, "W": Wf}
