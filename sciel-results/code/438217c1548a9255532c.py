"""Model-and-optimize: parameterize the 3-layer Ry/CZ circuit and directly
minimize the exact mean energy gap per spin against the exact ground state.
Pre-computed CZ diagonal, heuristic init, Nelder-Mead + L-BFGS-B hybrid with
diverse restarts for better global optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize

_GLOBAL_DEADLINE = time.time() + 140.0
_CACHE = {}

# Pre-computed structures
_PRECOMP = {}


def _precompute(n, coupling, field):
    key = (n, coupling, field)
    if key in _PRECOMP:
        return _PRECOMP[key]
    
    dim = 2 ** n
    
    # Build Hamiltonian
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        ma = 1 << i
        mb = 1 << (i + 1)
        for state in range(dim):
            za = 1.0 if ((state & ma) == 0) else -1.0
            zb = 1.0 if ((state & mb) == 0) else -1.0
            H[state, state] += -coupling * za * zb
    for i in range(n):
        m = 1 << i
        for state in range(dim):
            H[state, state ^ m] += -field
    
    # Ground energy
    eigvals = np.linalg.eigvalsh(H)
    E0 = float(eigvals[0])
    
    # Pre-compute CZ_all diagonal: product of CZ on (0,1), (1,2), ..., (n-2,n-1)
    cz_diag = np.ones(dim, dtype=complex)
    for i in range(n - 1):
        mask = (1 << i) | (1 << (i + 1))
        for state in range(dim):
            if (state & mask) == mask:
                cz_diag[state] *= -1.0
    
    # Pre-compute Ry index arrays
    ry_indices = {}
    for q in range(n):
        mask = 1 << q
        idx = np.arange(dim)
        zero_mask = (idx & mask) == 0
        zero_idx = idx[zero_mask]
        one_idx = zero_idx | mask
        ry_indices[q] = (zero_idx, one_idx)
    
    _PRECOMP[key] = (H, E0, cz_diag, ry_indices, dim)
    return _PRECOMP[key]


def _circuit_state_fast(theta, n, layers, precomp):
    H, E0, cz_diag, ry_indices, dim = precomp
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    
    for l in range(layers):
        for i in range(n):
            theta_val = theta[l, i]
            c = np.cos(theta_val * 0.5)
            s = np.sin(theta_val * 0.5)
            zero_idx, one_idx = ry_indices[i]
            a0 = psi[zero_idx]
            a1 = psi[one_idx]
            psi[zero_idx] = c * a0 - s * a1
            psi[one_idx] = s * a0 + c * a1
        
        if l < layers - 1:
            psi *= cz_diag
    
    return psi


def _heuristic_init(n, field, layers):
    """Initialize angles based on physics: |-> = Ry(pi/2)|0>."""
    theta = np.zeros((layers, n))
    base = np.pi / 2
    if field < 0.5:
        base *= field / 0.5
    for l in range(layers):
        for i in range(n):
            theta[l, i] = base
    return theta


def _optimize_case(n, coupling, field, layers, time_budget):
    precomp = _precompute(n, coupling, field)
    H, E0, cz_diag, ry_indices, dim = precomp
    total_params = layers * n

    def obj(theta_flat):
        theta = theta_flat.reshape(layers, n)
        psi = _circuit_state_fast(theta, n, layers, precomp)
        e = float(np.real(np.vdot(psi, H @ psi)))
        return (e - E0) / n

    bounds = [(-np.pi, np.pi)] * total_params
    t0 = time.time()

    # Phase 1: Nelder-Mead from heuristic init
    best_theta = _heuristic_init(n, field, layers)
    best_val = obj(best_theta.reshape(-1))

    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0:
        try:
            res = scipy.optimize.minimize(
                obj, best_theta.reshape(-1), method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-10, "fatol": 1e-12, "adaptive": True},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    # Phase 2: L-BFGS-B polish
    remaining = time_budget - (time.time() - t0)
    if remaining > 1.0:
        try:
            res = scipy.optimize.minimize(
                obj, best_theta.reshape(-1), method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-12},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    # Phase 3: Diverse restarts with L-BFGS-B
    remaining = time_budget - (time.time() - t0)
    if remaining > 3.0:
        rng = np.random.default_rng(123)
        n_extra = min(15, int(remaining / 1.5))
        for _ in range(n_extra):
            if time.time() - t0 > time_budget - 1.0:
                break
            if rng.random() < 0.5:
                s = rng.uniform(-np.pi, np.pi, size=(layers, n))
            else:
                s = _heuristic_init(n, field, layers) + rng.normal(0, 0.5, size=(layers, n))
                s = np.clip(s, -np.pi, np.pi)
            try:
                res = scipy.optimize.minimize(
                    obj, s.reshape(-1), method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 150, "ftol": 1e-13, "gtol": 1e-10},
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_theta = res.x.reshape(layers, n)
            except Exception:
                pass

    # Phase 4: Final high-precision polish
    remaining = time_budget - (time.time() - t0)
    if remaining > 1.0 and best_val < np.inf:
        try:
            res = scipy.optimize.minimize(
                obj, best_theta.reshape(-1), method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-13},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    # Phase 5: Nelder-Mead from best for potential escape
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0 and best_val < np.inf:
        try:
            res = scipy.optimize.minimize(
                obj, best_theta.reshape(-1), method="Nelder-Mead",
                options={"maxiter": 200, "xatol": 1e-12, "fatol": 1e-14, "adaptive": True},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((layers, n))
        best_theta[-1] = np.pi / 2
    return best_theta


def angles(n, coupling, field, layers):
    key = (int(n), float(coupling), float(field), int(layers))
    if key in _CACHE:
        return _CACHE[key].copy()

    remaining = _GLOBAL_DEADLINE - time.time()
    if remaining <= 3.0:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _CACHE[key] = theta
        return theta.copy()

    budget = min(remaining * 0.85, 25.0)

    try:
        theta = _optimize_case(n, coupling, field, layers, budget)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2

    if not np.all(np.isfinite(theta)):
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2

    _CACHE[key] = theta
    return theta.copy()
# EVOLVE-BLOCK-END
