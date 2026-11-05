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


def _precompute_cut_values(n, edges):
    """Precompute cut values for all basis states."""
    dim = 1 << n
    edge_list = list(edges)
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cv = 0
        for u, v in edge_list:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cv += 1
        cut_vals[mask] = cv
    return cut_vals


def _apply_mixer_fast(psi, beta, n):
    """Apply the mixer exp(-i*beta*sum X) to state vector psi."""
    cb = np.cos(beta)
    sb = np.sin(beta)
    for i in range(n):
        shape = [1] * n
        shape[i] = 2
        psi_r = psi.reshape(shape)
        psi_m = np.moveaxis(psi_r, i, 0).reshape(2, -1)
        a = psi_m[0].copy()
        b = psi_m[1].copy()
        psi_m[0] = cb * a - 1j * sb * b
        psi_m[1] = -1j * sb * a + cb * b
        psi = np.moveaxis(psi_m.reshape(shape), 0, i).reshape(-1)
    return psi


def _qaoa_state(psi_init, cut_vals, g0, b0, g1, b1, n):
    """Compute the full QAOA state for p=2."""
    psi = psi_init.copy()
    psi *= np.exp(-1j * g0 * cut_vals)
    psi = _apply_mixer_fast(psi, b0, n)
    psi *= np.exp(-1j * g1 * cut_vals)
    psi = _apply_mixer_fast(psi, b1, n)
    return psi


def _qaoa_expectation(psi_init, cut_vals, g0, b0, g1, b1, n):
    """Compute expected cut value for p=2 QAOA."""
    psi = _qaoa_state(psi_init, cut_vals, g0, b0, g1, b1, n)
    probs = np.abs(psi) ** 2
    return np.dot(probs, cut_vals)


def _qaoa_gradient(psi_init, cut_vals, g0, b0, g1, b1, n):
    """Compute gradient of expected cut with respect to [g0, b0, g1, b1]."""
    eps = 1e-6
    vals = np.array([
        _qaoa_expectation(psi_init, cut_vals, g0, b0, g1, b1, n),
        _qaoa_expectation(psi_init, cut_vals, g0 + eps, b0, g1, b1, n),
        _qaoa_expectation(psi_init, cut_vals, g0 - eps, b0, g1, b1, n),
        _qaoa_expectation(psi_init, cut_vals, g0, b0 + eps, g1, b1, n),
        _qaoa_expectation(psi_init, cut_vals, g0, b0 - eps, g1, b1, n),
        _qaoa_expectation(psi_init, cut_vals, g0, b0, g1 + eps, b1, n),
        _qaoa_expectation(psi_init, cut_vals, g0, b0, g1 - eps, b1, n),
        _qaoa_expectation(psi_init, cut_vals, g0, b0, g1, b1 + eps, n),
        _qaoa_expectation(psi_init, cut_vals, g0, b0, g1, b1 - eps, n),
    ])
    grad = np.array([
        (vals[1] - vals[2]) / (2 * eps),
        (vals[3] - vals[4]) / (2 * eps),
        (vals[5] - vals[6]) / (2 * eps),
        (vals[7] - vals[8]) / (2 * eps),
    ])
    return grad


def angles(n, edges, p):
    global _start_time, _DEADLINE
    start_time = time.time()

    if n > 12:
        return np.array([[0.4, 0.35], [0.4, 0.35]])

    dim = 1 << n
    cut_vals = _precompute_cut_values(n, edges)
    psi_init = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    def objective(params):
        return -_qaoa_expectation(psi_init, cut_vals, params[0], params[1], params[2], params[3], n)

    def gradient(params):
        return -_qaoa_gradient(psi_init, cut_vals, params[0], params[1], params[2], params[3], n)

    # Global grid search first to avoid local minima
    grid_range = np.linspace(0.05, 1.5, 20)
    best_score = -np.inf
    best_params = np.array([0.4, 0.35, 0.4, 0.35])

    # Coarse grid search
    for g0 in grid_range:
        for b0 in grid_range:
            for g1 in grid_range:
                for b1 in grid_range:
                    if time.time() - start_time > _DEADLINE - 30:
                        break
                    score = _qaoa_expectation(psi_init, cut_vals, g0, b0, g1, b1, n)
                    if score > best_score:
                        best_score = score
                        best_params = np.array([g0, b0, g1, b1])
                if time.time() - start_time > _DEADLINE - 30:
                    break
            if time.time() - start_time > _DEADLINE - 30:
                break
        if time.time() - start_time > _DEADLINE - 30:
            break

    # Refine with L-BFGS-B from multiple starting points
    inits = [best_params.tolist()]
    
    # Add some literature-based inits
    inits.extend([
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
    ])
    
    # Add random inits
    np.random.seed(42)
    for _ in range(20):
        inits.append(np.random.uniform([0.05, 0.05, 0.05, 0.05], 
                                       [1.5, 1.5, 1.5, 1.5]).tolist())

    bounds = [(0.0, np.pi)] * 4

    # First pass: quick L-BFGS-B
    for init in inits:
        if time.time() - start_time > _DEADLINE - 20:
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

    # Second pass: refine top candidates
    if time.time() - start_time < _DEADLINE - 10:
        refined_inits = [best_params.tolist()]
        for _ in range(10):
            perturbation = np.random.uniform(-0.1, 0.1, 4)
            candidate = np.clip(best_params + perturbation, 0, np.pi)
            refined_inits.append(candidate.tolist())
        
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
