"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute Hamiltonian matrices for n=4 and n=6
# H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
# Open chain, qubit i is bit i (little-endian)

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for open transverse-field Ising chain."""
    # Identity on single qubit
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Build Z_i Z_{i+1} terms and X_i terms
    # Use Kronecker products
    # Qubit 0 is least significant bit (rightmost in kron)
    
    def kron_all(ops):
        """ops is a list from qubit 0 to qubit n-1; result = ops[0] ⊗ ops[1] ⊗ ... ⊗ ops[n-1]"""
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * sum_{i=0}^{n-2} Z_i Z_{i+1}
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= coupling * kron_all(ops)
    
    # X terms: -field * sum X_i
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_all(ops)
    
    return H


def _apply_circuit(state, theta, n):
    """
    Apply the circuit: start from |0>^n, for each layer:
    - Apply Ry(theta[layer, i]) to qubit i for all i
    - If not the last layer, apply CZ on every adjacent pair (0,1), (1,2), ..., (n-2, n-1) in ascending order
    """
    layers = theta.shape[0]
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    # Precompute single-qubit gates
    # Ry(theta) = cos(theta/2) * I - i*sin(theta/2) * X
    # As a 2x2 matrix
    def ry_matrix(theta):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    
    # CZ matrix
    CZ = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]], dtype=complex)
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            gate = ry_matrix(theta[layer, i])
            # Apply gate to qubit i: need to build the full operator
            # For small n, we can do this by iterating over basis states
            # Or use the kron approach
            ops = [I2] * n
            ops[i] = gate
            full_op = kron_all(ops)
            state = full_op @ state
        
        # Apply CZ between adjacent pairs if not last layer
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ on qubits i and i+1
                ops = [I2] * n
                # Build CZ on qubits i, i+1
                # CZ acts on 2 qubits: |00>->|00>, |01>->|01>, |10>->|10>, |11>->-|11>
                # We need to embed this into the full space
                # For efficiency with small n, build the full operator
                full_cz = np.eye(dim, dtype=complex)
                for basis in range(dim):
                    bit_i = (basis >> i) & 1
                    bit_ip1 = (basis >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        full_cz[basis, basis] = -1
                state = full_cz @ state
    
    return state


# Precompute identity for n=2
I2 = np.eye(2, dtype=complex)


def _energy(state, H):
    """Compute <state|H|state>."""
    return np.real(np.conj(state) @ H @ state)


def _objective(params, n, coupling, field, H, exact_ground_energy):
    """Objective function: mean energy gap per spin."""
    theta = params.reshape(3, n)
    state = _apply_circuit(np.zeros(2**n, dtype=complex), theta, n)
    # Set state[0] = 1 for |0>^n
    # Actually _apply_circuit starts from |0>^n internally, let me fix
    energy = _energy(state, H)
    return (energy - exact_ground_energy) / n


def _objective_and_grad(params, n, coupling, field, H, exact_ground_energy):
    """Objective with gradient using parameter-shift rule for Ry gates."""
    theta = params.reshape(3, n)
    
    # Forward pass
    state = _apply_circuit(np.zeros(2**n, dtype=complex), theta, n)
    energy = _energy(state, H)
    obj = (energy - exact_ground_energy) / n
    
    # Gradient via parameter-shift rule
    # For Ry(theta) = cos(theta/2) I - i sin(theta/2) X
    # d/dtheta Ry = -1/2 sin(theta/2) I - i/2 cos(theta/2) X
    # Parameter shift: f'(theta) = (f(theta + pi/2) - f(theta - pi/2)) / 2
    # But we need the full circuit energy, so we shift each parameter individually
    
    grad = np.zeros_like(params)
    shift = np.pi / 2
    
    for idx in range(params.size):
        # Compute f(theta + shift)
        theta_plus = theta.copy()
        l, i = np.unravel_index(idx, (3, n))
        theta_plus[l, i] += shift
        state_plus = _apply_circuit(np.zeros(2**n, dtype=complex), theta_plus, n)
        energy_plus = _energy(state_plus, H)
        
        # Compute f(theta - shift)
        theta_minus = theta.copy()
        theta_minus[l, i] -= shift
        state_minus = _apply_circuit(np.zeros(2**n, dtype=complex), theta_minus, n)
        energy_minus = _energy(state_minus, H)
        
        grad[idx] = (energy_plus - energy_minus) / (2 * n)
    
    return obj, grad


# Cache for precomputed data
_cache = {}

def _get_precomputed(n, coupling, field):
    key = (n, coupling, field)
    if key not in _cache:
        H = _build_hamiltonian(n, coupling, field)
        # Compute exact ground state energy
        eigvals, eigvecs = np.linalg.eigh(H)
        exact_ground_energy = np.real(eigvals[0])
        _cache[key] = (H, exact_ground_energy)
    return _cache[key]


def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate
    ground states of open transverse-field Ising chains.
    
    Returns: finite real array (layers, n) in radians.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    H, exact_ground_energy = _get_precomputed(n, coupling, field)
    
    best_theta = np.zeros((layers, n))
    best_obj = np.inf
    
    def check_time():
        return time.time() < deadline
    
    # Multi-start L-BFGS-B optimization
    n_params = layers * n
    
    # Initial guesses: try several different starting points
    # 1. All zeros
    # 2. pi/2 on last layer (product |+> state)
    # 3. Random initializations
    # 4. Based on physical intuition
    
    initializations = []
    
    # Start 1: all zeros
    initializations.append(np.zeros(n_params))
    
    # Start 2: pi/2 on last layer
    init2 = np.zeros(n_params)
    init2[-n:] = np.pi / 2
    initializations.append(init2)
    
    # Start 3: pi/2 on all layers
    initializations.append(np.full(n_params, np.pi / 2))
    
    # Start 4: small values
    initializations.append(np.full(n_params, 0.1))
    
    # Start 5: alternating
    init5 = np.zeros(n_params)
    for l in range(layers):
        for i in range(n):
            init5[l * n + i] = 0.3 * ((l + i) % 2 * 2 - 1)
    initializations.append(init5)
    
    # Add random starts
    rng = np.random.default_rng(42)
    for _ in range(20):
        initializations.append(rng.uniform(-np.pi, np.pi, n_params))
    
    for init in initializations:
        if not check_time():
            break
        
        try:
            result = minimize(
                _objective,
                init,
                args=(n, coupling, field, H, exact_ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_theta = result.x.reshape(layers, n)
                
        except Exception:
            continue
    
    # If time remains, do a polish pass with tighter tolerances
    if check_time():
        try:
            result = minimize(
                _objective,
                best_theta.flatten(),
                args=(n, coupling, field, H, exact_ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_theta = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Final check: make sure we're within deadline
    if time.time() > deadline:
        # Return best found so far
        pass
    
    # Ensure finite values
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((layers, n))
    
    return best_theta


# EVOLVE-BLOCK-END
