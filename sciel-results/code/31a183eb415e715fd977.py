"""Model-and-optimize: parameterize the 3-layer Ry/CZ circuit and directly
minimize the exact mean energy gap per spin against the exact ground state.
Structural change: two-phase optimization (differential evolution + L-BFGS-B polish)
with vectorized gate application for speed."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize

_GLOBAL_DEADLINE = time.time() + 140.0
_CACHE = {}

# Pre-computed index arrays for fast gate application
_GATE_CACHE = {}


def _get_ry_indices(n, qubit):
    key = ('ry', n, qubit)
    if key not in _GATE_CACHE:
        dim = 2 ** n
        mask = 1 << qubit
        idx = np.arange(dim)
        zero_mask = (idx & mask) == 0
        zero_idx = idx[zero_mask]
        one_idx = zero_idx | mask
        _GATE_CACHE[key] = (zero_idx, one_idx)
    return _GATE_CACHE[key]


def _get_cz_indices(n, a, b):
    key = ('cz', n, a, b)
    if key not in _GATE_CACHE:
        dim = 2 ** n
        mask = (1 << a) | (1 << b)
        idx = np.arange(dim)
        both_one = (idx & mask) == mask
        _GATE_CACHE[key] = both_one
    return _GATE_CACHE[key]


def _apply_ry_fast(psi, n, qubit, theta):
    c = np.cos(theta * 0.5)
    s = np.sin(theta * 0.5)
    zero_idx, one_idx = _get_ry_indices(n, qubit)
    out = psi.copy()
    a0 = psi[zero_idx]
    a1 = psi[one_idx]
    out[zero_idx] = c * a0 - s * a1
    out[one_idx] = s * a0 + c * a1
    return out


def _apply_cz_fast(psi, n, a, b):
    both_one = _get_cz_indices(n, a, b)
    out = psi.copy()
    out[both_one] = -out[both_one]
    return out


def _circuit_state(theta, n, layers):
    dim = 2 ** n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    for l in range(layers):
        for i in range(n):
            psi = _apply_ry_fast(psi, n, i, theta[l, i])
        if l < layers - 1:
            for i in range(n - 1):
                psi = _apply_cz_fast(psi, n, i, i + 1)
    return psi


def _build_H(n, coupling, field):
    dim = 2 ** n
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
    return H


def _ground_energy(n, coupling, field):
    H = _build_H(n, coupling, field)
    return float(np.min(np.linalg.eigvalsh(H))), H


def _optimize_case(n, coupling, field, layers, time_budget):
    E0, H = _ground_energy(n, coupling, field)
    dim = 2 ** n
    total_params = layers * n

    def energy(theta):
        psi = _circuit_state(theta, n, layers)
        return float(np.real(np.vdot(psi, H @ psi)))

    def obj(theta_flat):
        theta = theta_flat.reshape(layers, n)
        return (energy(theta) - E0) / n

    bounds = [(-np.pi, np.pi)] * total_params
    t0 = time.time()

    # Phase 1: Differential Evolution for global exploration
    best_theta = np.zeros((layers, n))
    best_val = np.inf

    remaining = time_budget - (time.time() - t0)
    if remaining > 5.0:
        # Use a subset of time for DE
        de_time = min(remaining * 0.6, 12.0)
        try:
            # Differential evolution with moderate population
            pop_size = min(30, max(15, total_params * 2))
            de_res = scipy.optimize.differential_evolution(
                obj,
                bounds=bounds,
                maxiter=50,
                popsize=pop_size,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                polish=False,  # We'll do our own polish
                init='latinhypercube',
                updating='deferred',
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_theta = de_res.x.reshape(layers, n)
        except Exception:
            pass

    # Phase 2: L-BFGS-B polish from best DE result
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0 and best_val < np.inf:
        try:
            res = scipy.optimize.minimize(
                obj, best_theta.reshape(-1), method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-14, "gtol": 1e-10},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    # Phase 3: Additional L-BFGS-B restarts if time permits
    remaining = time_budget - (time.time() - t0)
    if remaining > 3.0:
        rng = np.random.default_rng(777)
        n_extra = min(8, int(remaining / 2.0))
        for _ in range(n_extra):
            if time.time() - t0 > time_budget - 1.5:
                break
            s = rng.uniform(-np.pi, np.pi, size=(layers, n))
            try:
                res = scipy.optimize.minimize(
                    obj, s.reshape(-1), method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 100, "ftol": 1e-13, "gtol": 1e-9},
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_theta = res.x.reshape(layers, n)
            except Exception:
                pass

    # Final high-precision polish
    remaining = time_budget - (time.time() - t0)
    if remaining > 1.5 and best_val < np.inf:
        try:
            res = scipy.optimize.minimize(
                obj, best_theta.reshape(-1), method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-15, "gtol": 1e-12},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((layers, n))
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

    budget = min(remaining * 0.85, 22.0)

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
