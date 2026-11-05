"""Optimized quantum circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0
_START_TIME = time.time()


def _check_deadline(margin=3.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i using Kronecker products."""
    total_dim = 2 ** n
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    H = np.zeros((total_dim, total_dim), dtype=complex)

    # ZZ terms
    for i in range(n - 1):
        op = np.array([[1.0]])
        for q in range(n):
            if q == i or q == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H += -coupling * op

    # X terms
    for i in range(n):
        op = np.array([[1.0]])
        for q in range(n):
            if q == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H += -field * op

    return H


def _get_ground_energy(H):
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0].real)


def _circuit_state(params, n, layers):
    """Build circuit state using vectorized operations."""
    total_dim = 2 ** n
    state = np.zeros(total_dim, dtype=complex)
    state[0] = 1.0

    # Precompute bit masks
    for layer in range(layers):
        # Apply Ry to every qubit (vectorized)
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            mask = 1 << q
            new_state = np.zeros_like(state)
            for idx in range(total_dim):
                if (idx & mask) == 0:
                    new_state[idx] += c * state[idx]
                    new_state[idx | mask] += s * state[idx]
                else:
                    new_state[idx] += c * state[idx]
                    new_state[idx ^ mask] += -s * state[idx]
            state = new_state

        # Apply CZ between adjacent pairs
        if layer < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                for idx in range(total_dim):
                    if (idx & mask1) and (idx & mask2):
                        state[idx] = -state[idx]

    return state


def _circuit_energy(params, n, coupling, field, layers, H):
    state = _circuit_state(params, n, layers)
    energy = np.vdot(state, H @ state).real
    return float(energy)


def _heuristic_initial(n, coupling, field, layers):
    """Generate a physics-motivated initial guess.
    
    For small field: spins mostly aligned, small Ry angles.
    For large field: spins mostly transverse, larger Ry angles.
    """
    params = np.zeros((layers, n))
    
    # Estimate the "natural" angle from mean-field: tan(theta) ~ field/coupling
    ratio = field / max(coupling, 1e-10)
    base_angle = np.arctan(ratio)  # angle from Z axis toward X
    
    # Layer 0: set initial angles
    for q in range(n):
        params[0, q] = base_angle * 0.8
    
    # Layer 1: refine
    for q in range(n):
        params[1, q] = base_angle * 0.5
    
    # Layer 2: small correction
    for q in range(n):
        params[2, q] = base_angle * 0.3
    
    return params


def angles(n, coupling, field, layers):
    """Optimize circuit parameters for the given Ising chain."""
    global _START_TIME
    _START_TIME = time.time()

    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _get_ground_energy(H)

    best_params = _heuristic_initial(n, coupling, field, layers)
    best_score = _circuit_energy(best_params, n, coupling, field, layers, H) - ground_energy

    def objective(params_flat):
        p = params_flat.reshape(layers, n)
        e = _circuit_energy(p, n, coupling, field, layers, H)
        return (e - ground_energy) / n

    # Generate diverse initial points
    np.random.seed(42)
    initial_points = []

    # Heuristic-based
    initial_points.append(_heuristic_initial(n, coupling, field, layers).flatten())

    # All zeros
    initial_points.append(np.zeros(layers * n))

    # Random inits
    for _ in range(4):
        initial_points.append(np.random.uniform(-np.pi, np.pi, size=layers * n))

    # Symmetry-based: try small uniform angles
    for a in [0.3, 0.5, 0.7, 1.0]:
        p = np.full((layers, n), a)
        initial_points.append(p.flatten())

    # Try negative angles
    for a in [-0.3, -0.5, -0.7]:
        p = np.full((layers, n), a)
        initial_points.append(p.flatten())

    # Evaluate all initial points first
    for x0 in initial_points:
        if not _check_deadline():
            break
        try:
            s = objective(x0)
            if s < best_score:
                best_score = s
                best_params = x0.reshape(layers, n)
        except Exception:
            continue

    # Run optimizer from best initial points
    # Sort by objective value and take top few
    scored_inits = []
    for x0 in initial_points:
        try:
            s = objective(x0)
            scored_inits.append((s, x0))
        except Exception:
            continue
    scored_inits.sort(key=lambda x: x[0])

    for _, x0 in scored_inits[:6]:
        if not _check_deadline():
            break
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            continue

    # If time remains, do a second round with tighter tolerance from the best
    if _check_deadline(margin=5.0):
        try:
            result = minimize(
                objective,
                best_params.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass

    # Ensure finite output
    best_params = np.nan_to_num(best_params, nan=0.0, posinf=0.0, neginf=0.0)

    return best_params
# EVOLVE-BLOCK-END
