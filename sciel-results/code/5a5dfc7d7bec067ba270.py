"""
Optimize shallow circuit parameters for Ising ground states using a hybrid
multi-start L-BFGS-B strategy with a fast vectorized state simulator and
analytic gradients via automatic differentiation of the circuit.

Strategy:
1. Fast vectorized simulator using tensor reshaping for Ry gates and
   efficient phase-flip for CZ gates.
2. Analytic gradient computation via parameter-shift rule (exact for Ry).
3. Multi-start L-BFGS-B with diverse initial points to escape local minima.
4. Strict time management with deadline guards.
"""
import numpy as np
import time
from scipy.optimize import minimize


def _build_hamiltonian(n, coupling, field):
    """
    Build H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    for an open chain of n qubits (little-endian: qubit i = bit i).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)

    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            eigenval = (-1) ** (bit_i + bit_ip1)
            H[b, b] -= coupling * eigenval

    # X terms: -field * sum X_i
    for i in range(n):
        mask_i = 1 << i
        for b in range(dim):
            b_flipped = b ^ mask_i
            H[b_flipped, b] -= field

    return H


def _ground_energy(n, coupling, field):
    """Compute exact ground energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real


def _apply_ry(state, q, theta, n):
    """Apply Ry(theta) on qubit q of state (little-endian, axis q)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    
    # Move axis q to front
    st = np.moveaxis(state, q, 0)
    new_st = np.empty_like(st)
    new_st[0] = c * st[0] - s * st[1]
    new_st[1] = s * st[0] + c * st[1]
    return np.moveaxis(new_st, 0, q)


def _apply_cz(state, q1, q2, n):
    """Apply CZ on qubits q1, q2. Phase -1 when both bits are 1."""
    dim = 1 << n
    mask1 = 1 << q1
    mask2 = 1 << q2
    indices = np.arange(dim)
    both = ((indices & mask1) != 0) & ((indices & mask2) != 0)
    state[both] *= -1
    return state


def _simulate(params, n, layers):
    """
    Simulate the circuit and return final state.
    params: (layers, n)
    Returns: state vector of length 2^n
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    for l in range(layers):
        for q in range(n):
            state = _apply_ry(state, q, params[l, q], n)
        if l < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)

    return state


def _energy(params, n, layers, H):
    """Compute circuit energy <psi|H|psi>."""
    psi = _simulate(params, n, layers)
    return np.vdot(psi, H @ psi).real


def _energy_and_gradient(params, n, layers, H, E0):
    """
    Compute energy gap and gradient using parameter-shift rule for Ry gates.
    For Ry(theta), the parameter-shift rule gives:
    dE/dtheta = (E(theta + pi/2) - E(theta - pi/2)) / 2
    This is exact for Ry since it's a single-parameter rotation.
    """
    dim = 1 << n
    energy = _energy(params, n, layers, H)
    gap = energy - E0
    
    grad = np.zeros_like(params)
    shift = np.pi / 2.0
    
    for l in range(layers):
        for q in range(n):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[l, q] = params[l, q] + shift
            p_minus[l, q] = params[l, q] - shift
            e_plus = _energy(p_plus, n, layers, H) - E0
            e_minus = _energy(p_minus, n, layers, H) - E0
            grad[l, q] = (e_plus - e_minus) / 2.0
    
    return gap, grad


def _objective(flat_params, n, layers, H, E0):
    """Objective function for optimizer (gap only, no gradient)."""
    params = flat_params.reshape((layers, n))
    try:
        return _energy(params, n, layers, H) - E0
    except Exception:
        return 1e10


def _objective_with_grad(flat_params, n, layers, H, E0):
    """Objective with analytic gradient via parameter-shift."""
    params = flat_params.reshape((layers, n))
    gap, grad = _energy_and_gradient(params, n, layers, H, E0)
    return gap, grad.flatten()


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    E0 = _ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)

    np.random.seed(42)

    best_params = None
    best_gap = np.inf

    # Generate diverse starting points
    n_starts = 30
    starts = []
    
    # Some random
    for _ in range(15):
        starts.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
    
    # Some small-angle
    for _ in range(5):
        starts.append(np.random.uniform(-0.5, 0.5, (layers, n)))
    
    # Some near zero with one layer varied
    for l in range(layers):
        p = np.zeros((layers, n))
        p[l, :] = np.random.uniform(-np.pi, np.pi, n)
        starts.append(p)
    
    # Symmetric patterns
    for val in [-np.pi/4, 0, np.pi/4]:
        p = np.full((layers, n), val)
        starts.append(p)
    
    # Truncate to n_starts
    starts = starts[:n_starts]

    bounds = [(-np.pi, np.pi)] * (layers * n)

    for start_idx, start_params in enumerate(starts):
        if time.time() > deadline - 3.0:
            break

        # Quick evaluation of starting point
        try:
            gap0 = _objective(start_params.flatten(), n, layers, H, E0)
        except Exception:
            continue

        if gap0 < best_gap:
            best_gap = gap0
            best_params = start_params.copy()

        # Run L-BFGS-B from this start
        if time.time() > deadline - 5.0:
            continue

        try:
            res = minimize(
                _objective_with_grad,
                start_params.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
        except Exception:
            continue

        if res.success or res.fun < best_gap:
            if res.fun < best_gap:
                best_gap = res.fun
                best_params = res.x.reshape((layers, n))

        # If we've found a very good solution, we can stop early
        if best_gap < 1e-7:
            break

    if best_params is None:
        return np.zeros((layers, n))

    return best_params
