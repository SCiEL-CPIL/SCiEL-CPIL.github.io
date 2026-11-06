"""Quantum information: shallow-circuit Ising ground states.
Exploits reflection symmetry of open transverse-field Ising chain to reduce
parameter space, then uses global grid search + local polish with deadline guard.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: must finish within 120s, use 150s hard limit with safety margin
_HARD_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0  # seconds to spare


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    Qubit i is bit i (little-endian: qubit 0 is least significant bit).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 - 2 * bit_i
            z_ip1 = 1 - 2 * bit_ip1
            H[state, state] += -coupling * z_i * z_ip1
    
    # X_i terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via eigendecomposition."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _apply_ry(theta, state):
    """Apply Ry(theta) to a single qubit (tensor product structure).
    Returns new state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    # Ry = [[c, -s], [s, c]]
    # For qubit i, reshape state, apply 2x2 to that axis, reshape back
    n = int(np.log2(len(state)))
    shape = [2] * n
    state_reshaped = state.reshape(shape)
    # Apply to qubit i (axis i)
    new_state = np.empty_like(state_reshaped)
    for idx in np.ndindex(*shape):
        idx_list = list(idx)
        if idx_list[i] == 0:
            idx1 = list(idx)
            idx1[i] = 1
            new_state[idx] = c * state_reshaped[idx] - s * state_reshaped[tuple(idx1)]
        else:
            idx0 = list(idx)
            idx0[i] = 0
            new_state[idx] = s * state_reshaped[tuple(idx0)] + c * state_reshaped[idx]
    return new_state.reshape(-1)


def _apply_ry_fast(theta, state, i, n):
    """Fast Ry application using reshape."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    shape = [2] * n
    sr = state.reshape(shape)
    # Extract the two slices for qubit i
    sl0 = [slice(None)] * n
    sl1 = [slice(None)] * n
    sl0[i] = 0
    sl1[i] = 1
    v0 = sr[tuple(sl0)]
    v1 = sr[tuple(sl1)]
    # New values
    new0 = c * v0 - s * v1
    new1 = s * v0 + c * v1
    # Build new state
    out = np.empty_like(sr)
    out[tuple(sl0)] = new0
    out[tuple(sl1)] = new1
    return out.reshape(-1)


def _apply_cz(state, i, j, n):
    """Apply CZ gate between qubits i and j.
    CZ = diag(1, 1, 1, -1) in basis |00>, |01>, |10>, |11>
    Only flips sign when both qubits are 1."""
    shape = [2] * n
    sr = state.reshape(shape)
    # Find indices where both qubit i and j are 1
    sl = [slice(None)] * n
    sl[i] = 1
    sl[j] = 1
    sr[tuple(sl)] *= -1
    return sr.reshape(-1)


def _circuit_energy(params, n, coupling, field, layers, ground_energy):
    """Compute mean energy gap per spin for given circuit parameters.
    params: array of shape (layers, n)
    Returns: (circuit_energy - ground_energy) / n
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for q in range(n):
            state = _apply_ry_fast(params[layer, q], state, q, n)
        
        # Apply CZ between adjacent pairs (ascending order), except after final layer
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
    
    # Compute energy: <state|H|state>
    # H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    energy = 0.0
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state_idx in range(dim):
            bit_i = (state_idx >> i) & 1
            bit_ip1 = (state_idx >> (i + 1)) & 1
            z_i = 1 - 2 * bit_i
            z_ip1 = 1 - 2 * bit_ip1
            amp = state[state_idx]
            energy += -coupling * z_i * z_ip1 * (amp.real ** 2 + amp.imag ** 2)
    
    # X_i terms: <psi|X_i|psi> = 2*Re(psi[i=0] * conj(psi[i=1]))
    for i in range(n):
        for state_idx in range(dim):
            if ((state_idx >> i) & 1) == 0:
                flipped = state_idx ^ (1 << i)
                a = state[state_idx]
                b = state[flipped]
                energy += -field * 2 * (a.real * b.real + a.imag * b.imag)
    
    return (energy - ground_energy) / n


def _circuit_energy_fast(params, n, coupling, field, layers, H, ground_energy):
    """Compute energy using precomputed Hamiltonian matrix."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry_fast(params[layer, q], state, q, n)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
    
    energy = np.real(np.vdot(state, H @ state))
    return (energy - ground_energy) / n


def _symmetric_params(a, b, c, d, n, layers):
    """Generate full parameter array from symmetric parameters.
    For open chain with reflection symmetry: theta_q = theta_{n-1-q}
    So we need n//2 unique angles per layer (if n even, n/2; if odd, (n+1)/2)
    
    For n=4: angles are [a, b, b, a]
    For n=6: angles are [a, b, c, c, b, a]
    
    params: (a, b, c, d) where a=layer0, b=layer1, c=layer2, d=unused or extra
    Actually let's use: layer0 symmetric angles, layer1 symmetric angles, layer2 symmetric angles
    """
    # For n=4: 2 unique per layer -> 6 params total
    # For n=6: 3 unique per layer -> 9 params total
    half = (n + 1) // 2  # number of unique angles per layer
    
    # We'll use a flat parameter vector of length layers * half
    # and expand to full (layers, n) array
    params = np.zeros((layers, n))
    for layer in range(layers):
        for q in range(half):
            val = params[layer, q]
            params[layer, q] = val
            params[layer, n - 1 - q] = val
    return params


def _symmetric_params_from_flat(flat_params, n, layers):
    """Convert flat symmetric parameter vector to full (layers, n) array."""
    half = (n + 1) // 2
    params = np.zeros((layers, n))
    for layer in range(layers):
        for q in range(half):
            val = flat_params[layer * half + q]
            params[layer, q] = val
            params[layer, n - 1 - q] = val
    return params


def _energy_from_flat(flat_params, n, coupling, field, layers, H, ground_energy):
    """Compute energy from flat symmetric parameter vector."""
    params = _symmetric_params_from_flat(flat_params, n, layers)
    return _circuit_energy_fast(params, n, coupling, field, layers, H, ground_energy)


def angles(n, coupling, field, layers):
    """Optimize circuit parameters using symmetry-reduced search."""
    start_time = time.time()
    deadline = start_time + _HARD_DEADLINE - _SAFETY_MARGIN
    
    # Precompute Hamiltonian and ground energy
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    ground_energy = eigvals[0]
    
    half = (n + 1) // 2
    n_params = layers * half
    
    # Check time
    if time.time() > deadline:
        # Return default: product |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    # Global grid search over symmetric parameter space
    # Each angle in [-pi/2, pi/2] (Ry range)
    # Start with coarse grid
    best_params = None
    best_energy = np.inf
    
    # Grid search with adaptive resolution
    grid_resolutions = [3, 5, 7]
    
    for res in grid_resolutions:
        if time.time() > deadline:
            break
        
        # Generate grid points
        grid_vals = np.linspace(-np.pi/2, np.pi/2, res)
        
        # For small grids, do full Cartesian product
        if n_params * res <= 10000:  # manageable
            # Full grid
            from itertools import product
            grid_points = list(product(*[grid_vals] * n_params))
        else:
            # Use random sampling
            np.random.seed(42)
            grid_points = [np.random.uniform(-np.pi/2, np.pi/2, n_params) for _ in range(5000)]
        
        # Evaluate each point
        for pt in grid_points:
            if time.time() > deadline:
                break
            e = _energy_from_flat(np.array(pt), n, coupling, field, layers, H, ground_energy)
            if e < best_energy:
                best_energy = e
                best_params = np.array(pt)
        
        # If we found a good solution, refine locally
        if best_params is not None:
            # Local polish using Nelder-Mead
            try:
                result = minimize(
                    _energy_from_flat,
                    best_params,
                    args=(n, coupling, field, layers, H, ground_energy),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final polish with tighter tolerance
    if best_params is not None and time.time() < deadline:
        try:
            result = minimize(
                _energy_from_flat,
                best_params,
                args=(n, coupling, field, layers, H, ground_energy),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Convert to full parameter array
    if best_params is not None:
        theta = _symmetric_params_from_flat(best_params, n, layers)
    else:
        # Fallback: product |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
    
    return theta
# EVOLVE-BLOCK-END
