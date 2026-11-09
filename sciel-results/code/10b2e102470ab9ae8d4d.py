"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
DEADLINE = 130.0  # wall-clock seconds; safety margin under the 150s limit

# EVOLVE-BLOCK-START

def _tensor_positions(n, m, p):
    pos = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pos.append((i * m + j, j * p + k, k * n + i))
    return pos

def _make_tensor(n, m, p):
    Udim, Vdim, Wdim = n * m, m * p, p * n
    T = np.zeros((Udim, Vdim, Wdim), dtype=float)
    for (a, b, c) in _tensor_positions(n, m, p):
        T[a, b, c] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fractions. U, V, W are lists of lists of floats (half-integers)."""
    Udim, Vdim, Wdim = n * m, m * p, p * n
    rank = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    posset = set(_tensor_positions(n, m, p))
    
    for a in range(Udim):
        for b in range(Vdim):
            for d in range(Wdim):
                s = Fraction(0)
                for t in range(rank):
                    if Uf[a][t] == 0 or Vf[b][t] == 0 or Wf[d][t] == 0:
                        continue
                    s += Uf[a][t] * Vf[b][t] * Wf[d][t]
                target = 1 if (a, b, d) in posset else 0
                if s != target:
                    return False
    return True

def _snap_and_verify(U, V, W, n, m, p):
    """Snap to half-integers and verify exactly."""
    Udim, Vdim, Wdim = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    
    Us = np.round(U * 2) / 2.0
    Vs = np.round(V * 2) / 2.0
    Ws = np.round(W * 2) / 2.0
    
    Ul = Us.tolist()
    Vl = Vs.tolist()
    Wl = Ws.tolist()
    
    if _verify_exact(Ul, Vl, Wl, n, m, p):
        return Ul, Vl, Wl
    return None

def _search_tensor(T, dims, nmp, rank, time_budget, seed):
    Udim, Vdim, Wdim = dims
    n, m, p = nmp
    rng = np.random.default_rng(seed)
    total = Udim * rank + Vdim * rank + Wdim * rank
    start = time.time()
    
    def obj(flat):
        U = flat[0:Udim * rank].reshape(Udim, rank)
        V = flat[Udim * rank:Udim * rank + Vdim * rank].reshape(Vdim, rank)
        W = flat[Udim * rank + Vdim * rank:].reshape(Wdim, rank)
        Trec = np.einsum('ic,jc,kc->ijk', U, V, W)
        R = Trec - T
        return float((R * R).sum())
        
    def grad(flat):
        U = flat[0:Udim * rank].reshape(Udim, rank)
        V = flat[Udim * rank:Udim * rank + Vdim * rank].reshape(Vdim, rank)
        W = flat[Udim * rank + Vdim * rank:].reshape(Wdim, rank)
        Trec = np.einsum('ic,jc,kc->ijk', U, V, W)
        R = Trec - T
        gU = 2.0 * np.einsum('ijk,jc,kc->ic', R, V, W)
        gV = 2.0 * np.einsum('ijk,ic,kc->jc', R, U, W)
        gW = 2.0 * np.einsum('ijk,ic,jc->kc', R, U, V)
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])

    for restart in range(1000):
        if time.time() - start > time_budget:
            break
        scale = float(rng.choice([0.5, 1.0, 1.5, 2.0]))
        flat0 = rng.standard_normal(total) * scale
        
        try:
            res = minimize(obj, flat0, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 2000, 'maxfun': 4000, 'gtol': 1e-12})
        except Exception:
            continue
            
        if res.fun < 1e-5:
            flat = res.x
            U = flat[0:Udim * rank].reshape(Udim, rank)
            V = flat[Udim * rank:Udim * rank + Vdim * rank].reshape(Vdim, rank)
            W = flat[Udim * rank + Vdim * rank:].reshape(Wdim, rank)
            
            sol = _snap_and_verify(U, V, W, n, m, p)
            if sol is not None:
                return sol
                
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    fallback = _schoolbook(n, m, p)
    best_rank = len(fallback["U"][0])
    best_sol = fallback
    
    def remaining():
        return max(0.0, DEADLINE - (time.time() - t0))
        
    T6 = _make_tensor(n, m, p)
    dims6 = (n * m, m * p, p * n)
    
    for r in (13, 14, 15, 16):
        if r >= best_rank:
            continue
        if remaining() < 10.0:
            break
            
        time_budget = min(remaining() - 5.0, 60.0)
        sol = _search_tensor(T6, dims6, (n, m, p), r, time_budget, seed=100 + r)
        if sol is not None:
            U, V, W = sol
            if _verify_exact(U, V, W, n, m, p):
                best_sol = {"U": U, "V": V, "W": W}
                best_rank = r
                if r == 13:
                    return best_sol
                break
            
    return best_sol

# EVOLVE-BLOCK-END
