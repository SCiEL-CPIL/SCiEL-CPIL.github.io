import numpy as np
from scipy.optimize import minimize
import itertools

def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit in state (state is 1D array of 2^n amplitudes)."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    new_state = np.zeros_like(state)
    for basis in range(2 ** n):
        bit = (basis >> qubit) & 1
        if bit == 0:
            new_state[basis] = c * state[basis]
            new_state[basis | (1 << qubit)] = s * state[basis | (1 << qubit)]
        else:
            new_state[basis] = -s * state[basis ^ (1 << qubit)]
            new_state[basis | (1 << qubit)] = c * state[basis]
    return new_state

def _apply_cz(state, q1, q2, n):
    """Apply CZ between q1 and q2."""
    new_state = np.copy(state)
    for basis in range(2 ** n):
        b1 = (basis >> q1) & 1
        b2 = (basis >> q2) & 1
        if b1 == 1 and b2 == 1:
            new_state[basis] = -state[basis]
    return new_state

def _simulate_circuit(n, angles, layers):
    """Simulate the circuit and return the statevector."""
    state = np.zeros(2 ** n)
    state[0] = 1.0  # |0...0>
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            state = _apply_ry(state, angles[layer, q], q, n)
        # Apply CZ between adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
    
    return state

def _compute_energy(state, n, coupling, field):
    """Compute expectation value of H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    energy = 0.0
    probs = np.abs(state) ** 2
    
    # ZZ terms
    for i in range(n - 1):
        zz_sum = 0.0
        for basis in range(2 ** n):
            b1 = (basis >> i) & 1
            b2 = (basis >> (i + 1)) & 1
            z1 = 1 if b1 == 0 else -1
            z2 = 1 if b2 == 0 else -1
            zz_sum += probs[basis] * z1 * z2
        energy += -coupling * zz_sum
    
    # X terms
    for i in range(n):
        x_sum = 0.0
        for basis in range(2 ** n):
            b = (basis >> i) & 1
            # X|0> = |1>, X|1> = |0>
            flipped = basis ^ (1 << i)
            x_sum += state[basis].conj() * state[flipped]
        energy += -field * 2 * np.real(x_sum)
    
    return energy

def _energy_objective(angles_flat, n, coupling, field, layers):
    """Flatten angles to 1D for optimizer."""
    angles = angles_flat.reshape(layers, n)
    state = _simulate_circuit(n, angles, layers)
    return _compute_energy(state, n, coupling, field)

def angles(n, coupling, field, layers):
    # Initial guess: all angles = pi/2 (creates |+> state)
    x0 = np.full((layers, n), np.pi / 2)
    x0_flat = x0.flatten()
    
    # Try multiple random starts and pick the best
    best_energy = np.inf
    best_angles = x0
    
    # Start with the default
    e = _energy_objective(x0_flat, n, coupling, field, layers)
    if e < best_energy:
        best_energy = e
        best_angles = x0
    
    # Try a few random starts
    np.random.seed(42)
    for trial in range(5):
        x0_rand = np.random.uniform(-np.pi, np.pi, size=(layers, n))
        x0_rand_flat = x0_rand.flatten()
        
        res = minimize(_energy_objective, x0_rand_flat, args=(n, coupling, field, layers),
                       method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
        
        if res.success and res.fun < best_energy:
            best_energy = res.fun
            best_angles = res.x.reshape(layers, n)
    
    # Also try starting from the default with optimization
    res = minimize(_energy_objective, x0_flat, args=(n, coupling, field, layers),
                   method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
    if res.success and res.fun < best_energy:
        best_energy = res.fun
        best_angles = res.x.reshape(layers, n)
    
    return best_angles
