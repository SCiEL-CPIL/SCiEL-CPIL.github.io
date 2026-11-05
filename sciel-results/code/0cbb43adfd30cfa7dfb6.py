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


def _qaoa_expectation_v2(n, edges, g0, b0, g1, b1, cut_vals=None, dim=None):
    """
    Compute expected cut for p=2 QAOA starting from |+>^n.
    Uses vectorized mixer application via reshape/moveaxis.
    """
    if dim is None:
        dim = 1 << n
    if cut_vals is None:
        edge_list = list(edges)
        cut_vals = np.zeros(dim, dtype=np.float64)
        for mask in range(dim):
            cv = 0
            for u, v in edge_list:
                if ((mask >> u) ^ (mask >> v)) & 1:
                    cv += 1
            cut_vals[mask] = cv

    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    # Cost layer 0
    psi *= np.exp(-1j * g0 * cut_vals)

    # Mixer layer 0
    def _mixer(psi, beta):
        cb = np.cos(beta)
        sb = np.sin(beta)
        for i in range(n):
            bit = 1 << i
            # Vectorized: reshape to (2, dim//2) along axis i
            shape = [1] * n
            shape[i] = 2
            psi_r = psi.reshape(shape)
            # Move axis i to front
            psi_m = np.moveaxis(psi_r, i, 0)
            psi_m = psi_m.reshape(2, -1)
            a = psi_m[0].copy()
            b = psi_m[1].copy()
            psi_m[0] = cb * a - 1j * sb * b
            psi_m[1] = -1j * sb * a + cb * b
            # Reshape back
            psi_m = psi_m.reshape(shape)
            psi = np.moveaxis(psi_m, 0, i).reshape(dim)
        return psi

    psi = _mixer(psi, b0)

    # Cost layer 1
    psi *= np.exp(-1j * g1 * cut_vals)

    # Mixer layer 1
    psi = _mixer(psi, b1)

    probs = np.abs(psi) ** 2
    return np.dot(probs, cut_vals)


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

    # Precompute the initial state
    psi_init = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    def eval_fn(params):
        """Evaluate expected cut for given parameters [g0, b0, g1, b1]."""
        g0, b0, g1, b1 = params
        psi = psi_init.copy()
        psi *= np.exp(-1j * g0 * cut_vals)

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
        [0.55, 0.35, 0.35, 0.25],
        [0.65, 0.45, 0.45, 0.35],
        [0.45, 0.25, 0.25, 0.15],
        [0.35, 0.45, 0.45, 0.35],
    ]

    # Add random inits with better coverage
    np.random.seed(42)
    for _ in range(30):
        inits.append(np.random.uniform([0.05, 0.05, 0.05, 0.05], 
                                       [1.5, 1.5, 1.5, 1.5]).tolist())

    bounds = [(0.0, np.pi)] * 4

    best_score = -np.inf
    best_params = np.array([0.4, 0.35, 0.4, 0.35])

    # First pass: quick L-BFGS-B with limited iterations
    for init in inits:
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        try:
            res = scipy.optimize.minimize(
                objective, np.array(init, dtype=np.float64),
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-7, 'gtol': 1e-5, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            continue

    # Second pass: refine top candidates with more iterations
    # Use the best found so far and a few nearby points
    if time.time() - start_time < _DEADLINE - 20:
        # Create refined inits around the best solution
        refined_inits = []
        refined_inits.append(best_params.tolist())
        
        # Add perturbations around best
        for _ in range(10):
            perturbation = np.random.uniform(-0.1, 0.1, 4)
            candidate = np.clip(best_params + perturbation, 0, np.pi)
            refined_inits.append(candidate.tolist())
        
        # Also try a few from the original inits that were close
        for init in inits[:5]:
            refined_inits.append(init)
        
        for init in refined_inits:
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            try:
                res = scipy.optimize.minimize(
                    objective, np.array(init, dtype=np.float64),
                    jac=gradient,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7, 'disp': False}
                )
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                continue

    return np.array([[best_params[0], best_params[1]],
                     [best_params[2], best_params[3]]])
