"""Optimized shallow-circuit Ising ground state approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
    # We need pairs of indices differing by 1<<i
    indices = np.arange(dim)
    
    for l in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            mask = 1 << i
            # Select indices where bit i is 0
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
                # Indices where both bits are 1
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
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        
        # Determine ZZ value for each state
        # bit_i = (state >> i) & 1
        # bit_j = (state >> j) & 1
        # ZZ = 1 if bit_i == bit_j else -1
        
        bits_i = (indices >> i) & 1
        bits_j = (indices >> (i + 1)) & 1
        
        # ZZ is 1 if equal, -1 if different
        # XOR is 0 if equal, 1 if different
        # So ZZ = 1 - 2*(bits_i XOR bits_j)
        zz_vals = 1.0 - 2.0 * (bits_i ^ bits_j)
        
        energy += -coupling * np.sum(zz_vals * prob)
    
    # X terms
    for i in range(n):
        mask = 1 << i
        # <state|X_i|state> = sum_{s} conj(state[s]) * state[s ^ mask]
        # We can sum over all s, but each pair is counted twice?
        # X_i flips bit i.
        # Let s have bit i = 0. s' = s ^ mask has bit i = 1.
        # Term: conj(state[s]) * state[s'] + conj(state[s']) * state[s]
        # = 2 * Re( conj(state[s]) * state[s'] )
        
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
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        
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
    deadline = 120.0  # seconds, with safety margin
    
    # Get exact ground energy
    exact_e = _exact_ground_energy(n, coupling, field)
    
    # Symmetry reduction
    # Unique qubits are 0 to floor(n/2 - 1) ? No.
    # n=4: 0,1,2,3. Pairs (0,3), (1,2). Unique: 0,1. Count 2.
    # n=6: 0,1,2,3,4,5. Pairs (0,5), (1,4), (2,3). Unique: 0,1,2. Count 3.
    # Count = ceil(n/2)
    num_unique = (n + 1) // 2
    
    # Mapping function: unique_params (layers, num_unique) -> full_params (layers, n)
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
    
    # Initial guess
    # Heuristic: -pi/2 is good for strong field
    # Try a few random starts or a grid?
    # Differential evolution will handle global search.
    
    bounds = [(-np.pi, np.pi)] * (layers * num_unique)
    
    best_params = None
    best_obj = np.inf
    
    # 1. Differential Evolution
    try:
        de_result = differential_evolution(
            objective_unique,
            bounds=bounds,
            maxiter=100,
            popsize=20,
            tol=0.01,
            seed=42,
            polish=False # We will polish manually
        )
        if de_result.fun < best_obj:
            best_obj = de_result.fun
            best_params = de_result.x
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        if best_params is None:
             # Fallback to zero
             best_params = np.zeros(layers * num_unique)
        return expand_params(best_params.reshape(layers, num_unique))
    
    # 2. Local Optimization (L-BFGS-B)
    if best_params is not None:
        try:
            local_result = minimize(
                objective_unique,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 5000}
            )
            if local_result.fun < best_obj:
                best_obj = local_result.fun
                best_params = local_result.x
        except Exception:
            pass
    else:
        # If DE failed, try a random start + local
        try:
            x0 = np.random.uniform(-np.pi, np.pi, layers * num_unique)
            local_result = minimize(
                objective_unique,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 5000}
            )
            if local_result.fun < best_obj:
                best_obj = local_result.fun
                best_params = local_result.x
        except Exception:
            pass

    # Final check
    elapsed = time.time() - start_time
    if best_params is None:
        best_params = np.zeros(layers * num_unique)
        
    return expand_params(best_params.reshape(layers, num_unique))

# EVOLVE-BLOCK-END
