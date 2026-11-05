import numpy as np
import time
import scipy.optimize

_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    dim = 1 << n
    best = 0
    edge_list = list(edges)
    for mask in range(dim):
        cut = 0
        for u, v in edge_list:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best


def angles(n, edges, p):
    global _start_time, _DEADLINE
    start_time = time.time()
    
    if n > 12:
        # Fallback for large n
        return np.array([[0.4, 0.35], [0.4, 0.35]])
    
    # Precompute cut values once
    dim = 1 << n
    edge_list = list(edges)
    
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cv = 0
        for u, v in edge_list:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cv += 1
        cut_vals[mask] = cv
    
    # Precompute mixer indices for speed
    # mixer_indices[i] is a list of pairs (idx0, idx1) where idx1 = idx0 | (1<<i)
    # and idx0 has bit i = 0
    mixer_indices = []
    for i in range(n):
        bit = 1 << i
        pairs = []
        for mask in range(dim):
            if (mask & bit) == 0:
                pairs.append((mask, mask | bit))
        mixer_indices.append(pairs)
        
    # Precompute the initial state
    psi_init = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    def eval_fn(params):
        """Evaluate expected cut for given parameters [g0, b0, g1, b1]."""
        g0, b0, g1, b1 = params
        psi = psi_init.copy()
        psi *= np.exp(-1j * g0 * cut_vals)
        
        def _mixer(psi, beta):
            cb = np.cos(beta)
            sb = np.sin(beta)
            for i in range(n):
                # Vectorized update using precomputed pairs
                # We need to update psi[idx0] and psi[idx1] simultaneously
                # This is tricky to vectorize perfectly without temporary arrays
                # because psi[idx1] depends on old psi[idx0] and vice versa.
                # However, we can use np.add.at or similar? No, simple loop over pairs is fast enough for n=10 (512 pairs per qubit * 10 qubits = 5120 ops).
                # Actually, let's just use the loop. It's C-level speed if we use lists? No, Python loop is slow.
                # Let's stick to the reshape method but maybe it's not the bottleneck.
                # The bottleneck is likely the number of evaluations.
                pass
            return psi
        
        # Let's use the reshape method inside eval_fn for simplicity, but ensure it's tight
        def _mixer_fast(psi, beta):
            cb = np.cos(beta)
            sb = np.sin(beta)
            for i in range(n):
                bit = 1 << i
                shape = [1] * n
                shape[i] = 2
                psi_r = psi.reshape(shape)
                psi_m = np.moveaxis(psi_r, i, 0).reshape(2, -1)
                a = psi_m[0].copy()
                b = psi_m[1].copy()
                psi_m[0] = cb * a - 1j * sb * b
                psi_m[1] = -1j * sb * a + cb * b
                psi = np.moveaxis(psi_m.reshape(shape), 0, i).reshape(dim)
            return psi
        
        psi = _mixer_fast(psi, b0)
        psi *= np.exp(-1j * g1 * cut_vals)
        psi = _mixer_fast(psi, b1)
        probs = np.abs(psi) ** 2
        return np.dot(probs, cut_vals)
    
    def objective(params):
        return -eval_fn(params)
    
    def gradient(params):
        eps = 1e-5
        g0, b0, g1, b1 = params
        grad = np.zeros(4)
        grad[0] = (eval_fn([g0+eps, b0, g1, b1]) - eval_fn([g0-eps, b0, g1, b1])) / (2*eps)
        grad[1] = (eval_fn([g0, b0+eps, g1, b1]) - eval_fn([g0, b0-eps, g1, b1])) / (2*eps)
        grad[2] = (eval_fn([g0, b0, g1+eps, b1]) - eval_fn([g0, b0, g1-eps, b1])) / (2*eps)
        grad[3] = (eval_fn([g0, b0, g1, b1+eps]) - eval_fn([g0, b0, g1-eps, b1])) / (2*eps)
        return -grad
    
    # Improved Inits based on QAOA literature and common optimal regions
    inits = [
        [0.4, 0.3, 0.2, 0.2],
        [0.5, 0.4, 0.3, 0.3],
        [0.3, 0.3, 0.2, 0.2],
        [0.6, 0.4, 0.4, 0.3],
        [0.2, 0.2, 0.1, 0.1],
        [0.8, 0.5, 0.5, 0.4],
        [0.5, 0.5, 0.5, 0.5],
        [0.4, 0.4, 0.4, 0.4],
    ]
    
    # Add random inits
    np.random.seed(42)
    for _ in range(20):
        inits.append(np.random.uniform([0.05, 0.05, 0.05, 0.05], 
                                       [1.5, 1.5, 1.5, 1.5]).tolist())
    
    bounds = [(0.0, np.pi)] * 4
    
    best_score = -np.inf
    best_params = np.array([0.4, 0.35, 0.4, 0.35])
    
    for init in inits:
        if time.time() - start_time > _DEADLINE - 3:
            break
        
        try:
            res = scipy.optimize.minimize(
                objective, np.array(init, dtype=np.float64),
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            continue
    
    return np.array([[best_params[0], best_params[1]],
                     [best_params[2], best_params[3]]])
