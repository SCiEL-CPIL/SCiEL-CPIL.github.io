import time
import numpy as np
from scipy.optimize import differential_evolution

N = 50

def _score(h):
    """Compute the ratio of squared L2 norm to (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h)
    if np.any(h < -1e-10):
        return -1.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-10:
        return -1.0
    l2_sq = np.sum(h ** 2)
    # Autoconvolution via FFT
    h_ext = np.zeros(2 * N - 1)
    h_ext[:N] = h
    conv = np.fft.irfft(np.fft.rfft(h_ext, n=2*N-1) ** 2, n=2*N-1)
    sup_conv = np.max(conv)
    if sup_conv < 1e-10:
        return -1.0
    return l2_sq / (l1 * sup_conv)

def _decode(params):
    """Decode parameter vector into a height vector.
    params: first k are positions (integers 0..N-1), next k are weights (positive).
    """
    k = len(params) // 2
    positions = np.round(params[:k]).astype(int)
    weights = params[k:]
    h = np.zeros(N)
    for pos, w in zip(positions, weights):
        if 0 <= pos < N:
            h[pos] += max(w, 0.0)
    return h

def _objective(params):
    h = _decode(params)
    return -_score(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try several different sparsities k
    best_h = None
    best_score = -np.inf
    
    # Start with the Sidon set as initial guess
    S = _get_sidon_set(8)
    h_init = np.zeros(N)
    for pos in S:
        h_init[pos] = 1.0
    h_init = h_init / np.sum(h_init)
    best_h = h_init.copy()
    best_score = _score(h_init)
    
    # For each sparsity level, run differential evolution
    for k in [6, 7, 8, 9, 10]:
        if time.time() > deadline - 20.0:
            break
        
        # Bounds: positions in [0, N-1], weights in [0.01, 1.0]
        bounds = [(0, N - 1)] * k + [(0.01, 1.0)] * k
        
        # Initial population: use Sidon set positions with varied weights
        def _init_pop(size, bounds):
            pop = np.zeros((size, 2 * k))
            for i in range(size):
                # Use Sidon positions as base, add some perturbation
                for j in range(k):
                    if j < len(S):
                        pop[i, j] = S[j] + np.random.uniform(-1, 1)
                        pop[i, j] = np.clip(pop[i, j], 0, N - 1)
                    else:
                        pop[i, j] = np.random.uniform(0, N - 1)
                    pop[i, k + j] = np.random.uniform(0.1, 0.5)
            return pop
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=30,
                popsize=15,
                tol=1e-4,
                seed=42,
                init=_init_pop(15, bounds),
                polish=True,
                update_interval=1
            )
            
            if result.success or result.fun < -best_score:
                h_candidate = _decode(result.x)
                s = _score(h_candidate)
                if s > best_score:
                    best_score = s
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Local polish: try adjusting weights with Nelder-Mead
    if time.time() < deadline - 10.0 and best_h is not None:
        h_flat = best_h.copy()
        # Only optimize non-zero entries
        nonzero_idx = np.where(h_flat > 1e-6)[0]
        if len(nonzero_idx) > 0:
            w_init = h_flat[nonzero_idx]
            from scipy.optimize import minimize
            res = minimize(
                lambda w: -_score(np.where(np.isin(np.arange(N), nonzero_idx), np.maximum(w, 0), 0)),
                w_init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < -best_score:
                h_candidate = np.zeros(N)
                h_candidate[nonzero_idx] = np.maximum(res.x, 0)
                s = _score(h_candidate)
                if s > best_score:
                    best_score = s
                    best_h = h_candidate.copy()
    
    if time.time() > deadline:
        return best_h.tolist()
        
    return best_h.tolist()

def _get_sidon_set(n):
    """
    Returns a maximal Sidon set in [0, n-1].
    A Sidon set (B_2 sequence) has the property that all pairwise sums 
    (or differences) are unique.
    For this purpose, we want distinct differences to keep side-peaks low.
    """
    S = [0]
    for i in range(1, n):
        diffs = set()
        ok = True
        for j in S:
            d = i - j
            if d in diffs:
                ok = False
                break
            diffs.add(d)
        existing_diffs = set()
        for a in range(len(S)):
            for b in range(a+1, len(S)):
                existing_diffs.add(S[b] - S[a])
        
        if ok:
            for d in diffs:
                if d in existing_diffs:
                    ok = False
                    break
        
        if ok:
            S.append(i)
            
    return S
