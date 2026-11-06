"""Optimized shallow-circuit Ising ground state approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    # Precompute indices for Ry application
    indices = np.arange(dim)
    
    for l in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            mask = 1 << i
            mask_0 = indices[indices & mask == 0]
            mask_1 = mask_0 | mask
            
            c = np.cos(theta[l, i] / 2.0)
            s = -1j * np.sin(theta[l, i] / 2.0)
            
            a0 = state[mask_0]
            a1 = state[mask_1]
            
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
    energy = 0.0
    prob = np.abs(state) ** 2
    
    indices = np.arange(dim)
    
    # ZZ terms
    for i in range(n - 1):
        bits_i = (indices >> i) & 1
        bits_j = (indices >> (i + 1)) & 1
        zz_vals = 1.0 - 2.0 * (bits_i ^ bits_j)
        energy += -coupling * np.sum(zz_vals * prob)
    
    # X terms
    for i in range(n):
        mask = 1 << i
        mask_0 = indices[indices & mask == 0]
        mask_1 = mask_0 | mask
        terms = np.conj(state[mask_0]) * state[mask_1]
        x_val = 2.0 * np.real(np.sum(terms))
        energy += -field * x_val
    
    return energy

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    indices = np.arange(dim)
    
    # ZZ terms
    for i in range(n - 1):
        bits_i = (indices >> i) & 1
        bits_j = (indices >> (i + 1)) & 1
        zz_vals = 1.0 - 2.0 * (bits_i ^ bits_j)
        H[indices, indices] += -coupling * zz_vals
    
    # X terms
    for i in range(n):
        mask = 1 << i
        mask_0 = indices[indices & mask == 0]
        mask_1 = mask_0 | mask
        H[mask_0, mask_1] += -field
        H[mask_1, mask_0] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def angles(n, coupling, field, layers):
    """
    Return optimal angles for the circuit.
    """
    start_time = time.time()
    deadline = 120.0
    
    exact_e = _exact_ground_energy(n, coupling, field)
    
    # Symmetry reduction: reflection symmetry i <-> n-1-i
    num_unique = (n + 1) // 2
    
    def expand_params(unique_params):
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
        state = _circuit_state_vectorized(full_params, n, layers)
        e_circuit = _compute_energy_vectorized(state, n, coupling, field)
        return (e_circuit - exact_e) / n
    
    bounds = [(-np.pi, np.pi)] * (layers * num_unique)
    
    best_params = None
    best_obj = np.inf
    
    # Strategy: Global grid search + local polish
    # Grid search is robust for small parameter spaces
    # layers=3, num_unique is 2 (n=4) or 3 (n=6)
    # Total params: 6 or 9
    
    # For n=4: 6 params. Grid of 8 points per param -> 8^6 = 262144 points. Too many?
    # Let's estimate time.
    # Each eval: state vector dim 16. Very fast.
    # 262k evals * 1ms = 262s. Too slow.
    # Reduce grid.
    
    # For n=4: 6 params. Grid of 5 points per param -> 5^6 = 15625 points. ~15s.
    # For n=6: 9 params. Grid of 4 points per param -> 4^9 = 262144 points. Too slow.
    # For n=6: 9 params. Grid of 3 points per param -> 3^9 = 19683 points. ~20s.
    
    # Adaptive grid resolution
    if n == 4:
        grid_res = 5
    else:
        grid_res = 3
    
    # Generate grid
    grid_points = np.array(np.meshgrid(*([np.linspace(-np.pi, np.pi, grid_res)] * (layers * num_unique))))
    grid_points = grid_points.T.reshape(-1, layers * num_unique)
    
    # Evaluate all grid points
    # Vectorize? No, state vector construction is not trivially vectorizable across params.
    # But we can loop.
    
    for x in grid_points:
        if time.time() > deadline - 30:
            break
        val = objective_unique(x)
        if val < best_obj:
            best_obj = val
            best_params = x
    
    # If no grid points evaluated (shouldn't happen), fallback
    if best_params is None:
        best_params = np.zeros(layers * num_unique)
    
    # Local polish from best grid point
    # Try multiple local optimizations from nearby points or random restarts if time permits
    if time.time() < deadline - 10:
        try:
            local_result = minimize(
                objective_unique,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 2000}
            )
            if local_result.fun < best_obj:
                best_obj = local_result.fun
                best_params = local_result.x
        except Exception:
            pass
    
    # If time remains, try a few random restarts + local
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            try:
                x0 = np.random.uniform(-np.pi, np.pi, layers * num_unique)
                local_result = minimize(
                    objective_unique,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': 1000}
                )
                if local_result.fun < best_obj:
                    best_obj = local_result.fun
                    best_params = local_result.x
            except Exception:
                pass
    
    return expand_params(best_params.reshape(layers, num_unique))

# EVOLVE-BLOCK-END
