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


def _exact_ground_state(n, coupling, field):
    """Compute exact ground state vector and energy by diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_idx = np.argmin(eigenvalues)
    ground_energy = float(eigenvalues[ground_idx])
    ground_state = eigenvectors[:, ground_idx].copy()
    # Ensure deterministic phase: make the largest-magnitude component real and positive
    max_idx = np.argmax(np.abs(ground_state))
    phase = np.angle(ground_state[max_idx])
    ground_state = ground_state * np.exp(-1j * phase)
    if np.abs(ground_state[max_idx]) > 1e-15:
        ground_state = ground_state * (np.sign(np.real(ground_state[max_idx])) if np.real(ground_state[max_idx]) >= 0 else -1)
    return ground_state, ground_energy


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


def _fidelity_objective(params, target_state, n, layers):
    """Return 1 - |<target|circuit>|^2 (to minimize)."""
    state = _circuit_state(params, n, layers)
    overlap = np.vdot(target_state, state)
    fidelity = np.abs(overlap) ** 2
    return 1.0 - float(np.real(fidelity))


def _fidelity_and_grad(params, target_state, n, layers):
    """Return (1-fidelity, gradient) using finite differences."""
    val = _fidelity_objective(params, target_state, n, layers)
    grad = np.zeros_like(params, dtype=np.float64)
    h = 1e-7
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += h
        p_minus = params.copy()
        p_minus[i] -= h
        f_plus = _fidelity_objective(p_plus, target_state, n, layers)
        f_minus = _fidelity_objective(p_minus, target_state, n, layers)
        grad[i] = (f_plus - f_minus) / (2 * h)
    return val, grad


def _energy_and_grad(params, n, coupling, field, layers, H, e0):
    """Return ((e_circ - e0)/n, gradient) using finite differences."""
    e_circ = _circuit_energy(params, n, coupling, field, layers, H)
    val = (e_circ - e0) / n
    grad = np.zeros_like(params, dtype=np.float64)
    h = 1e-7
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += h
        p_minus = params.copy()
        p_minus[i] -= h
        e_plus = _circuit_energy(p_plus, n, coupling, field, layers, H)
        e_minus = _circuit_energy(p_minus, n, coupling, field, layers, H)
        grad[i] = (e_plus - e_minus) / (2 * h) / n
    return val, grad


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]

    # Get exact ground state and energy
    ground_state, e0 = _exact_ground_state(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    n_params = layers * n

    # Generate initial guesses
    np.random.seed(42 + n * 100 + int(field * 1000))
    initial_guesses = []

    # All zeros
    initial_guesses.append(np.zeros(n_params))

    # All pi/2
    initial_guesses.append(np.full(n_params, np.pi / 2))

    # Layer 1 = pi/2, Layer 2 = 0, Layer 3 = pi/2
    g3 = np.zeros(n_params)
    g3[:n] = np.pi / 2
    g3[2 * n:3 * n] = np.pi / 2
    initial_guesses.append(g3)

    # All pi/4
    initial_guesses.append(np.full(n_params, np.pi / 4))

    # Layer 1 = pi/6, Layer 2 = pi/3, Layer 3 = pi/6
    g5 = np.zeros(n_params)
    g5[:n] = np.pi / 6
    g5[n:2 * n] = np.pi / 3
    g5[2 * n:3 * n] = np.pi / 6
    initial_guesses.append(g5)

    # Layer 1 = pi/3, Layer 2 = pi/6, Layer 3 = pi/2
    g6 = np.zeros(n_params)
    g6[:n] = np.pi / 3
    g6[n:2 * n] = np.pi / 6
    g6[2 * n:3 * n] = np.pi / 2
    initial_guesses.append(g6)

    # Random perturbations
    for r in range(20):
        gr = np.random.randn(n_params) * 1.0
        gr = np.clip(gr, -np.pi, np.pi)
        initial_guesses.append(gr)

    # Additional structured guesses based on ground state
    # Try to match dominant amplitudes
    for r in range(10):
        gr = np.random.randn(n_params) * 0.5
        gr = np.clip(gr, -np.pi, np.pi)
        initial_guesses.append(gr)

    best_result = None
    best_val = np.inf

    # Phase 1: Nelder-Mead on energy objective from multiple starts
    def energy_obj(params):
        e_circ = _circuit_energy(params, n, coupling, field, layers, H)
        return (e_circ - e0) / n

    for x_init in initial_guesses:
        if _time_left() < 10.0:
            break
        try:
            res = opt.minimize(energy_obj, x_init, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10,
                                       'adaptive': True})
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
        except Exception:
            pass

    if best_result is None:
        best_result = initial_guesses[0].copy()
        best_val = energy_obj(best_result)

    # Phase 2: L-BFGS-B with finite-difference gradient on energy (polish)
    if _time_left() > 15.0:
        try:
            res2 = opt.minimize(_energy_and_grad, best_result, args=(n, coupling, field, layers, H, e0),
                               method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12,
                                        'maxls': 50})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x.copy()
        except Exception:
            pass

    # Phase 3: Fidelity-based optimization (least-squares style)
    if _time_left() > 15.0:
        def fid_obj(params):
            return _fidelity_objective(params, ground_state, n, layers)

        def fid_and_grad(params):
            return _fidelity_and_grad(params, ground_state, n, layers)

        # Start from best energy result
        try:
            res3 = opt.minimize(fid_obj, best_result, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
            if res3.fun < 1e-10:  # very high fidelity
                e_check = _circuit_energy(res3.x, n, coupling, field, layers, H)
                val_check = (e_check - e0) / n
                if val_check < best_val:
                    best_val = val_check
                    best_result = res3.x.copy()
        except Exception:
            pass

        # Also try L-BFGS-B on fidelity
        if _time_left() > 10.0:
            try:
                res3b = opt.minimize(fid_and_grad, best_result, args=(ground_state, n, layers),
                                    method='L-BFGS-B',
                                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12,
                                             'maxls': 50})
                e_check = _circuit_energy(res3b.x, n, coupling, field, layers, H)
                val_check = (e_check - e0) / n
                if val_check < best_val:
                    best_val = val_check
                    best_result = res3b.x.copy()
            except Exception:
                pass

    # Phase 4: Another Nelder-Mead polish on energy with tighter tolerance
    if _time_left() > 10.0:
        try:
            res4 = opt.minimize(energy_obj, best_result, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
            if res4.fun < best_val:
                best_val = res4.fun
                best_result = res4.x.copy()
        except Exception:
            pass

    # Phase 5: Final L-BFGS-B polish
    if _time_left() > 5.0:
        try:
            res5 = opt.minimize(_energy_and_grad, best_result, args=(n, coupling, field, layers, H, e0),
                               method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13,
                                        'maxls': 50})
            if res5.fun < best_val:
                best_val = res5.fun
                best_result = res5.x.copy()
        except Exception:
            pass

    # Phase 6: Very tight Nelder-Mead final polish
    if _time_left() > 3.0:
        try:
            res6 = opt.minimize(energy_obj, best_result, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13,
                                        'adaptive': True})
            if res6.fun < best_val:
                best_val = res6.fun
                best_result = res6.x.copy()
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
