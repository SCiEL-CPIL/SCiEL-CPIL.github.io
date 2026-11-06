import numpy as np
from scipy.optimize import minimize
import time


def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)

    I2 = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        ops = [I2] * n
        ops[i] = Z
        ops[i + 1] = Z
        ZZ = ops[0]
        for op in ops[1:]:
            ZZ = np.kron(ZZ, op)
        H -= coupling * ZZ

    # X_i terms
    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        Xi = ops[0]
        for op in ops[1:]:
            Xi = np.kron(Xi, op)
        H -= field * Xi

    return H


def _exact_ground_energy(H):
    eigvals = np.linalg.eigvalsh(H)
    return float(np.real(eigvals[0]))


def _circuit_state(theta, n, layers):
    """Build the circuit state from angles."""
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    for layer in range(layers):
        # Apply Ry(theta) to each qubit
        for q in range(n):
            th = theta[layer, q]
            c = np.cos(th / 2.0)
            s = np.sin(th / 2.0)
            new_state = np.zeros_like(state)
            for idx in range(dim):
                bit = (idx >> q) & 1
                if bit == 0:
                    idx1 = idx | (1 << q)
                    new_state[idx] = c * state[idx] - s * state[idx1]
                    new_state[idx1] = s * state[idx] + c * state[idx1]
            state = new_state

        # Apply CZ between adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                for idx in range(dim):
                    b1 = (idx >> q) & 1
                    b2 = (idx >> (q + 1)) & 1
                    if b1 and b2:
                        state[idx] = -state[idx]

    return state


def _energy_gap(theta, n, coupling, field, H, E_ground):
    state = _circuit_state(theta, n, 3)
    energy = np.real(state.conj().T @ H @ state)
    return (energy - E_ground) / n


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 135.0

    H = _build_hamiltonian(n, coupling, field)
    E_ground = _exact_ground_energy(H)

    num_params = layers * n

    def objective(params):
        theta = params.reshape(layers, n)
        return _energy_gap(theta, n, coupling, field, H, E_ground)

    best_params = None
    best_obj = np.inf

    rng = np.random.default_rng(42)

    # Structured initializations
    inits = [
        np.zeros(num_params),
        np.full(num_params, np.pi / 2),
        np.full(num_params, -np.pi / 2),
        np.full(num_params, np.pi / 4),
        np.full(num_params, -np.pi / 4),
        rng.uniform(-0.5, 0.5, size=num_params),
        rng.uniform(-np.pi, np.pi, size=num_params),
    ]

    for init in inits:
        if time.time() > deadline - 20.0:
            break
        try:
            result = minimize(objective, init, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            continue

    # Random restarts around best
    while time.time() < deadline - 20.0:
        if best_params is not None:
            init = best_params + rng.normal(0, 0.3, size=num_params)
        else:
            init = rng.uniform(-np.pi, np.pi, size=num_params)
        try:
            result = minimize(objective, init, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-11, 'gtol': 1e-7})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Final high-precision polish
    if best_params is not None and time.time() < deadline - 10.0:
        try:
            result = minimize(objective, best_params, method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    if best_params is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    return best_params.reshape(layers, n)
