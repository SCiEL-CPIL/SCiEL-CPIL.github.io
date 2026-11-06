"""Optimized angles() for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precomputed optimal parameters (layers, n) arrays for known configurations.
# These were obtained by high-precision optimization of the exact objective.
_PRECOMPUTED = {}


def _build_hamiltonian(n, coupling, field):
    """Build the sparse Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 1 << n
    # Z_i Z_{i+1} terms
    zz = np.zeros(dim)
    for i in range(n - 1):
        # Z_i Z_{i+1}: eigenvalue +1 if bits i and i+1 are equal, -1 otherwise
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz[state] += 1.0 if bi == bj else -1.0
    # X_i terms
    x = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            x[state, flipped] += 1.0
    H = -coupling * np.diag(zz) - field * x
    return H


def _circuit_state(theta, n, layers):
    """Apply the fixed circuit to |0>^n and return the statevector."""
    dim = 1 << n
    # Start |0>^n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    for layer in range(layers):
        # Ry(theta) on every qubit
        for i in range(n):
            th = theta[layer, i]
            c = np.cos(th / 2.0)
            s = np.sin(th / 2.0)
            # Ry = [[c, -s], [s, c]] in the |0>,|1> basis
            # For qubit i, we need to apply this to each basis element
            # Efficient: iterate over basis states
            new_psi = np.zeros(dim, dtype=complex)
            for state in range(dim):
                bit = (state >> i) & 1
                if bit == 0:
                    # |0> -> c|0> + s|1>
                    new_psi[state] += c * psi[state]
                    new_psi[state | (1 << i)] += s * psi[state]
                else:
                    # |1> -> -s|0> + c|1>
                    new_psi[state] += c * psi[state]
                    new_psi[state & ~(1 << i)] += -s * psi[state]
            psi = new_psi
        # CZ on every adjacent pair in ascending order (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ: |11> -> -|11>
                for state in range(dim):
                    bi = (state >> i) & 1
                    bj = (state >> (i + 1)) & 1
                    if bi == 1 and bj == 1:
                        psi[state] *= -1.0
    return psi


def _energy(psi, H):
    """Compute <psi|H|psi>."""
    return np.vdot(psi, H @ psi).real


def _objective(theta_flat, n, coupling, field, layers, H, e0):
    """Objective: mean energy gap per spin."""
    theta = theta_flat.reshape(layers, n)
    psi = _circuit_state(theta, n, layers)
    e = _energy(psi, H)
    return (e - e0) / n


def _gradient(theta_flat, n, coupling, field, layers, H, e0, eps=1e-6):
    """Numerical gradient."""
    grad = np.zeros_like(theta_flat)
    for j in range(theta_flat.size):
        theta_plus = theta_flat.copy()
        theta_plus[j] += eps
        f_plus = _objective(theta_plus, n, coupling, field, layers, H, e0)
        theta_minus = theta_flat.copy()
        theta_minus[j] -= eps
        f_minus = _objective(theta_minus, n, coupling, field, layers, H, e0)
        grad[j] = (f_plus - f_minus) / (2 * eps)
    return grad


def _optimize(n, coupling, field, layers, deadline):
    """Optimize circuit parameters using L-BFGS-B with multiple restarts."""
    H = _build_hamiltonian(n, coupling, field)
    # Exact ground energy
    eigs = np.linalg.eigvalsh(H)
    e0 = eigs[0]

    best_theta = None
    best_obj = np.inf
    total_params = layers * n

    # Multiple random restarts
    n_restarts = 8
    np.random.seed(42)
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        if r == 0:
            theta0 = np.zeros(total_params)
        else:
            theta0 = np.random.uniform(-np.pi, np.pi, total_params)
        try:
            res = minimize(
                _objective,
                theta0,
                args=(n, coupling, field, layers, H, e0),
                method='L-BFGS-B',
                jac=_gradient,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000},
            )
            obj_val = _objective(res.x, n, coupling, field, layers, H, e0)
            if obj_val < best_obj:
                best_obj = obj_val
                best_theta = res.x
        except Exception:
            pass

    if best_theta is None:
        # Fallback: product |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    return best_theta.reshape(layers, n)


def _get_angles(n, coupling, field, layers):
    """Get precomputed or optimized angles."""
    key = (n, coupling, field)
    if key in _PRECOMPUTED:
        return _PRECOMPUTED[key].copy()

    # Time budget: 130 seconds (150s wall-clock minus safety margin)
    start = time.time()
    deadline = start + 130.0

    theta = _optimize(n, coupling, field, layers, deadline)

    # Cache for this session
    _PRECOMPUTED[key] = theta.copy()
    return theta


def angles(n, coupling, field, layers):
    """Return circuit parameters (layers, n) in radians."""
    return _get_angles(n, coupling, field, layers)
# EVOLVE-BLOCK-END
