"""Optimized shallow-circuit Ising ground state approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _circuit_state(theta, n, layers, precomputed):
    """
    Build the circuit state from angles.
    theta: (layers, n) array of angles
    n: number of qubits
    layers: number of layers (fixed at 3)
    precomputed: dict with precomputed index arrays
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    indices = precomputed['indices']
    
    for l in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            mask = 1 << i
            mask_0 = precomputed['mask0'][i]
            mask_1 = precomputed['mask1'][i]
            
            c = np.cos(theta[l, i] / 2.0)
            s = -1j * np.sin(theta[l, i] / 2.0)
            
            a0 = state[mask_0].copy()
            a1 = state[mask_1].copy()
            
            state[mask_0] = c * a0 + s * a1
            state[mask_1] = s * a0 + c * a1
        
        # Apply CZ if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                mask_bits = precomputed['cz_mask'][i]
                state[mask_bits] *= -1.0
    
    return state

def _compute_energy(state, n, coupling, field, precomputed):
    """
    Compute <state|H|state> where H = -coupling*sum Z_i Z_{i+1} - field*sum X_i
    """
    prob = np.abs(state) ** 2
    energy = 0.0
    
    # ZZ terms
    for i in range(n - 1):
        zz_vals = precomputed['zz_vals'][i]
        energy += -coupling * np.sum(zz_vals * prob)
    
    # X terms
    for i in range(n):
        mask_0 = precomputed['mask0'][i]
        mask_1 = precomputed['mask1'][i]
        terms = np.conj(state[mask_0]) * state[mask_1]
        x_val = 2.0 * np.real(np.sum(terms))
        energy += -field * x_val
    
    return energy

def _exact_ground_energy(n, coupling, field, precomputed):
    """Compute exact ground state energy by diagonalizing H."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    indices = precomputed['indices']
    
    # ZZ terms
    for i in range(n - 1):
        zz_vals = precomputed['zz_vals'][i]
        H[indices, indices] += -coupling * zz_vals
    
    # X terms
    for i in range(n):
        mask_0 = precomputed['mask0'][i]
        mask_1 = precomputed['mask1'][i]
        H[mask_0, mask_1] += -field
        H[mask_1, mask_0] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _precompute(n):
    """Precompute index arrays and operator values for fast evaluation."""
    dim = 1 << n
    indices = np.arange(dim)
    
    mask0 = []
    mask1 = []
    for i in range(n):
        mask = 1 << i
        m0 = indices[indices & mask == 0]
        m1 = m0 | mask
        mask0.append(m0)
        mask1.append(m1)
    
    zz_vals = []
    for i in range(n - 1):
        bits_i = (indices >> i) & 1
        bits_j = (indices >> (i + 1)) & 1
        zz = 1.0 - 2.0 * (bits_i ^ bits_j)
        zz_vals.append(zz)
    
    cz_mask = []
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        mask_both = mask_i | mask_j
        mask_bits = indices[(indices & mask_both) == mask_both]
        cz_mask.append(mask_bits)
    
    return {
        'indices': indices,
        'mask0': mask0,
        'mask1': mask1,
        'zz_vals': zz_vals,
        'cz_mask': cz_mask
    }

def angles(n, coupling, field, layers):
    """
    Return optimal angles for the circuit.
    """
    start_time = time.time()
    deadline = 115.0  # seconds, with safety margin before 120s kill
    
    # Precompute
    precomputed = _precompute(n)
    
    # Get exact ground energy
    exact_e = _exact_ground_energy(n, coupling, field, precomputed)
    
    # Symmetry reduction: enforce left-right mirror symmetry
    # n=4: qubits 0,1,2,3 -> pairs (0,3), (1,2) -> unique: 0,1 (count 2)
    # n=6: qubits 0,1,2,3,4,5 -> pairs (0,5), (1,4), (2,3) -> unique: 0,1,2 (count 3)
    num_unique = (n + 1) // 2
    
    def expand_params(unique_params):
        """Expand unique (layers, num_unique) to full (layers, n) with mirror symmetry."""
        full_params = np.zeros((layers, n))
        for l in range(layers):
            for i in range(num_unique):
                val = unique_params[l, i]
                full_params[l, i] = val
                full_params[l, n - 1 - i] = val
        return full_params
    
    def objective_unique(x):
        unique_params = x.reshape(layers, num_unique)
        full_params = expand_params(unique_params)
        state = _circuit_state(full_params, n, layers, precomputed)
        e_circuit = _compute_energy(state, n, coupling, field, precomputed)
        return (e_circuit - exact_e) / n
    
    dim_params = layers * num_unique
    bounds = [(-np.pi, np.pi)] * dim_params
    
    # Multi-start Nelder-Mead optimization
    # Use deterministic seeds for reproducibility
    rng = np.random.RandomState(42)
    
    best_obj = np.inf
    best_params = None
    
    # Number of restarts: scale with available time
    # Each NM run takes roughly 5-15 seconds for small dimensions
    # Start with a heuristic initial guess, then add random restarts
    
    # Heuristic initial guess: for strong field, angles near -pi/2 help
    # For weak field, angles near 0
    field_factor = min(1.0, field / 2.0)
    x0_heuristic = np.zeros(dim_params)
    for l in range(layers):
        for i in range(num_unique):
            x0_heuristic[l * num_unique + i] = -np.pi / 2.0 * field_factor
    
    # First try the heuristic
    try:
        result = minimize(
            objective_unique,
            x0_heuristic,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        if best_params is None:
            best_params = x0_heuristic
        return expand_params(best_params.reshape(layers, num_unique))
    
    # Additional random restarts
    num_restarts = 10  # Will be limited by time
    for r in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        x0 = rng.uniform(-np.pi, np.pi, dim_params)
        
        try:
            result = minimize(
                objective_unique,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Final polish with L-BFGS-B if we have time
    elapsed = time.time() - start_time
    if best_params is not None and elapsed < deadline - 15:
        try:
            local_result = minimize(
                objective_unique,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-15}
            )
            if local_result.fun < best_obj:
                best_obj = local_result.fun
                best_params = local_result.x
        except Exception:
            pass
    
    if best_params is None:
        best_params = x0_heuristic
    
    return expand_params(best_params.reshape(layers, num_unique))

# EVOLVE-BLOCK-END
