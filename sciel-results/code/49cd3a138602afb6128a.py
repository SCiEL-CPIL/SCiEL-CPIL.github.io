"""
Rewritten program: Symmetry-reduced grid + local polish for Ising ground states.

Strategy:
1. Exploit left-right reflection symmetry of the open chain: theta[l, i] == theta[l, n-1-i].
   This reduces the parameter space from layers*n to layers*ceil(n/2) unique angles.
2. Use a coarse-to-fine grid search over the reduced parameter space to find a good basin.
3. Refine with L-BFGS-B (or Nelder-Mead if gradient fails) on the full parameter vector,
   initialized from the best grid point (symmetrically expanded).
4. Fast vectorized state simulation and energy evaluation.
5. Aggressive time management.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill
_cache = {}


def _circuit_state_vectorized(theta, n, layers):
    """
    Build the circuit state from angles using vectorized operations.
    theta: (layers, n) array of angles
    n: number of qubits
    layers: number of layers (fixed at 3)
    
    Circuit: Start |0>^n
    For each layer l:
        Apply Ry(theta[l, i]) to qubit i for all i
        If not the last layer, apply CZ on every adjacent pair in ascending order
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    indices = np.arange(dim)
    
    for l in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            mask = 1 << i
            mask_0 = indices[(indices & mask) == 0]
            mask_1 = mask_0 | mask
            
            c = np.cos(theta[l, i] / 2.0)
            s = -1j * np.sin(theta[l, i] / 2.0)
            
            a0 = state[mask_0].copy()
            a1 = state[mask_1].copy()
            
            state[mask_0] = c * a0 + s * a1
            state[mask_1] = s * a0 + c * a1
        
        # Apply CZ if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = 1 << i
                mask_j = 1 << (i + 1)
                mask_both = mask_i | mask_j
                mask_bits = (indices & mask_both) == mask_both
                state[indices[mask_bits]] *= -1.0
    
    return state


def _compute_energy_vectorized(state, n, coupling, field):
    """
    Compute <state|H|state> where H = -coupling*sum Z_i Z_{i+1} - field*sum X_i
    """
    dim = 1 << n
    indices = np.arange(dim)
    
    energy = 0.0
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    # Z_i = 1 if bit i is 0, -1 if bit i is 1
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        
        # Compute Z_i Z_{i+1} for each basis state
        zi = np.where((indices & mask_i) == 0, 1.0, -1.0)
        zj = np.where((indices & mask_j) == 0, 1.0, -1.0)
        zz = zi * zj
        
        # <psi| ZZ |psi> = sum |psi_k|^2 * ZZ_k (since ZZ is diagonal)
        prob = np.abs(state) ** 2
        energy += -coupling * np.sum(prob * zz)
    
    # X terms: -field * sum X_i
    # X_i flips bit i: <psi| X_i |psi> = 2 * Re(sum_k psi_k^* psi_{k^i})
    for i in range(n):
        mask = 1 << i
        # For each index k, k^mask is the flipped version
        # We need sum over k where (k & mask) == 0 of psi_k^* psi_{k|mask} + c.c.
        # Actually: <psi|X_i|psi> = sum_k psi_k^* (psi_{k^mask})
        # = sum_k psi_k^* psi_{k^mask}
        # Since X_i is real and symmetric, this is 2*Re(sum over k with bit i=0 of psi_k^* psi_{k|mask})
        
        mask_0 = indices[(indices & mask) == 0]
        mask_1 = mask_0 | mask
        
        # sum over k in mask_0 of psi_k^* psi_{k|mask}
        cross = np.sum(np.conj(state[mask_0]) * state[mask_1])
        energy += -field * 2.0 * np.real(cross)
    
    return np.real(energy)


def _build_hamiltonian(n, coupling, field):
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag
        
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    return H


def _exact_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _symmetric_expand(reduced, n, layers):
    """
    Expand reduced symmetric parameters to full (layers, n) array.
    reduced has shape (layers, n_half) where n_half = ceil(n/2).
    """
    n_half = (n + 1) // 2
    full = np.zeros((layers, n))
    for l in range(layers):
        for i in range(n_half):
            full[l, i] = reduced[l, i]
            if i != n - 1 - i:
                full[l, n - 1 - i] = reduced[l, i]
    return full


def _objective_full(params, n, coupling, field, layers, H, e0):
    theta = params.reshape(layers, n)
    psi = _circuit_state_vectorized(theta, n, layers)
    energy = _compute_energy_vectorized(psi, n, coupling, field)
    return (energy - e0) / n


def _objective_and_grad(params, n, coupling, field, layers, H, e0):
    theta = params.reshape(layers, n)
    psi = _circuit_state_vectorized(theta, n, layers)
    energy = _compute_energy_vectorized(psi, n, coupling, field)
    val = (energy - e0) / n
    
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros_like(params)
    for k in range(params.size):
        p_plus = params.copy()
        p_plus[k] += eps
        theta_plus = p_plus.reshape(layers, n)
        psi_plus = _circuit_state_vectorized(theta_plus, n, layers)
        energy_plus = _compute_energy_vectorized(psi_plus, n, coupling, field)
        grad[k] = (energy_plus - e0) / n - val
        # Actually: (E_plus - E)/eps / n
        grad[k] = (energy_plus - energy) / eps / n
    
    return val, grad


def _optimize(n, coupling, field, layers):
    H = _build_hamiltonian(n, coupling, field)
    e0 = _exact_ground_energy(n, coupling, field)
    
    n_half = (n + 1) // 2
    n_reduced = layers * n_half
    n_full = layers * n
    
    # Coarse grid search over reduced space
    # For each unique angle, search over a range of values
    # Use a coarse grid first, then refine
    
    # Heuristic: the optimal angle is roughly in [-pi/2, pi/2]
    # For pure Z coupling, theta=0 is optimal
    # For pure X field, theta=pi/2 is optimal
    
    best_val = np.inf
    best_reduced = None
    
    # Coarse grid: 5 points per parameter
    coarse_points = np.linspace(-np.pi/2, np.pi/2, 7)
    
    # Multidimensional grid is too large for 6 parameters (7^6 = 117649)
    # Instead, use a sequential coordinate descent approach with grid
    
    # Initialize with heuristic
    ratio = field / coupling if coupling != 0 else 1.0
    base_angle = np.arctan(ratio)
    reduced_init = np.full(n_reduced, base_angle)
    
    # Evaluate initial
    full_init = _symmetric_expand(reduced_init, n, layers)
    psi_init = _circuit_state_vectorized(full_init, n, layers)
    e_init = _compute_energy_vectorized(psi_init, n, coupling, field)
    best_val = (e_init - e0) / n
    best_reduced = reduced_init.copy()
    
    # Coordinate descent with grid refinement
    for coord in range(n_reduced):
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        
        # Try different values for this coordinate
        for val in coarse_points:
            test_reduced = best_reduced.copy()
            test_reduced[coord] = val
            full_test = _symmetric_expand(test_reduced, n, layers)
            psi_test = _circuit_state_vectorized(full_test, n, layers)
            e_test = _compute_energy_vectorized(psi_test, n, coupling, field)
            val_test = (e_test - e0) / n
            if val_test < best_val:
                best_val = val_test
                best_reduced = test_reduced.copy()
    
    # Now do a local optimization from the best point found
    full_best = _symmetric_expand(best_reduced, n, layers).flatten()
    
    # Try L-BFGS-B with numerical gradient
    try:
        res = minimize(
            _objective_and_grad,
            full_best,
            args=(n, coupling, field, layers, H, e0),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_reduced = res.x.reshape(layers, n).copy()
            # Symmetrize
            for l in range(layers):
                for i in range(n_half):
                    best_reduced[l, n-1-i] = best_reduced[l, i]
            best_params_full = best_reduced.flatten()
        else:
            best_params_full = full_best
    except Exception:
        best_params_full = full_best
    
    # Also try Nelder-Mead from a few other starting points
    # 1. All zeros
    # 2. All pi/4
    # 3. Random
    
    rng = np.random.default_rng(42)
    inits = [
        np.zeros(n_full),
        np.full(n_full, np.pi/4),
        np.full(n_full, base_angle),
        rng.uniform(-np.pi/2, np.pi/2, size=n_full)
    ]
    
    for x0 in inits:
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        try:
            def obj(x):
                return _objective_full(x, n, coupling, field, layers, H, e0)
            
            res = minimize(
                obj,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'fatol': 1e-10, 'xatol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params_full = res.x.copy()
        except Exception:
            pass
    
    return best_params_full.reshape(layers, n)


def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    try:
        theta = _optimize(n, coupling, field, layers)
        _cache[key] = theta
        return theta
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _cache[key] = theta
        return theta
# EVOLVE-BLOCK-END
