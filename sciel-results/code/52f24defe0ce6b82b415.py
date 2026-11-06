"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill
_cache = {}


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.float64)

    for idx in range(dim):
        # Diagonal part: ZZ terms
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag

        # Off-diagonal part: X terms
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field

    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return float(eigenvalues[0])


def _circuit_state(params, n, layers):
    """Compute the state vector produced by the circuit."""
    thetas = params.reshape(layers, n)
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    shape = [2] * n

    for layer in range(layers):
        # Apply all Ry gates for this layer
        for q in range(n):
            theta = thetas[layer, q]
            if abs(theta) < 1e-15:
                continue
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            state_r = state.reshape(shape)
            state_moved = np.moveaxis(state_r, q, 0)
            new_state = np.empty_like(state_moved)
            new_state[0] = c * state_moved[0] - s * state_moved[1]
            new_state[1] = s * state_moved[0] + c * state_moved[1]
            state = np.moveaxis(new_state, 0, q).reshape(-1)

        # Apply CZ gates between adjacent pairs (not after final layer)
        if layer < layers - 1:
            state_r = state.reshape(shape)
            for q in range(n - 1):
                idx = [slice(None)] * n
                idx[q] = 1
                idx[q + 1] = 1
                state_r[tuple(idx)] *= -1.0
            state = state_r.reshape(-1)

    return state


def _circuit_energy(params, n, coupling, field, layers, H=None):
    """Compute expectation value of H for given circuit parameters."""
    if H is None:
        H = _build_hamiltonian(n, coupling, field)
    state = _circuit_state(params, n, layers)
    energy = np.vdot(state, H @ state)
    return float(np.real(energy))


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]

    e0 = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)

    def objective(params):
        e_circ = _circuit_energy(params, n, coupling, field, layers, H)
        return (e_circ - e0) / n

    # Generate diverse initial guesses
    np.random.seed(12345)
    n_params = layers * n

    # Strategy: for the Ising model, the ground state has specific structure.
    # For low field: mostly |0000...> or |1111...> with some X mixing
    # For high field: mostly |+>^n
    # The circuit starts from |0>^n, so first layer creates superposition,
    # middle layers entangle, last layer fine-tunes.

    initial_guesses = []

    # Guess 1: All zeros (|0>^n, which is good for strong coupling, weak field)
    initial_guesses.append(np.zeros(n_params))

    # Guess 2: All pi/2 (creates |+>^n, good for strong field)
    g2 = np.full(n_params, np.pi / 2)
    initial_guesses.append(g2)

    # Guess 3: Layer 1 = pi/2, Layer 2 = 0, Layer 3 = pi/2
    g3 = np.zeros(n_params)
    g3[:n] = np.pi / 2
    g3[2*n:3*n] = np.pi / 2
    initial_guesses.append(g3)

    # Guess 4: Layer 1 = pi/4, Layer 2 = pi/4, Layer 3 = pi/4
    g4 = np.full(n_params, np.pi / 4)
    initial_guesses.append(g4)

    # Guess 5: Layer 1 = pi/6, Layer 2 = pi/3, Layer 3 = pi/6
    g5 = np.zeros(n_params)
    g5[:n] = np.pi / 6
    g5[n:2*n] = np.pi / 3
    g5[2*n:3*n] = np.pi / 6
    initial_guesses.append(g5)

    # Guess 6: Symmetric - all layers same angle, try pi/2 for last
    g6 = np.zeros(n_params)
    g6[:n] = np.pi / 3
    g6[n:2*n] = np.pi / 6
    g6[2*n:3*n] = np.pi / 2
    initial_guesses.append(g6)

    # Guess 7: Random perturbations around pi/2
    g7 = np.full(n_params, np.pi / 2) + np.random.randn(n_params) * 0.3
    g7 = np.clip(g7, -np.pi, np.pi)
    initial_guesses.append(g7)

    # Guess 8: Random perturbations around 0
    g8 = np.random.randn(n_params) * 0.5
    g8 = np.clip(g8, -np.pi, np.pi)
    initial_guesses.append(g8)

    # More random restarts
    for r in range(10):
        gr = np.random.randn(n_params) * 1.0
        gr = np.clip(gr, -np.pi, np.pi)
        initial_guesses.append(gr)

    best_result = None
    best_val = np.inf

    # Phase 1: Nelder-Mead from multiple starts
    for x_init in initial_guesses:
        if _time_left() < 5.0:
            break
        try:
            res = opt.minimize(objective, x_init, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                       'adaptive': True})
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
        except Exception:
            pass

    if best_result is None:
        best_result = initial_guesses[0].copy()
        best_val = objective(best_result)

    # Phase 2: L-BFGS-B polish
    if _time_left() > 3.0:
        try:
            res2 = opt.minimize(objective, best_result, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x.copy()
        except Exception:
            pass

    # Phase 3: Another Nelder-Mead polish with tighter tolerance
    if _time_left() > 3.0:
        try:
            res3 = opt.minimize(objective, best_result, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
            if res3.fun < best_val:
                best_val = res3.fun
                best_result = res3.x.copy()
        except Exception:
            pass

    # Phase 4: Final L-BFGS-B
    if _time_left() > 2.0:
        try:
            res4 = opt.minimize(objective, best_result, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res4.fun < best_val:
                best_val = res4.fun
                best_result = res4.x.copy()
        except Exception:
            pass

    theta = best_result.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    if _time_left() <= 0:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
