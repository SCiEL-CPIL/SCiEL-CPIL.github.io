"""Variational ansatz for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# Precomputed Pauli matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Single-qubit gates
def ry(theta):
    """Ry rotation gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def rz(theta):
    """Rz rotation gate."""
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)

# Two-qubit gate
def cz_gate():
    """CZ gate."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=complex)

def kron(*mats):
    """Tensor product of matrices."""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result

def build_hamiltonian(n, coupling, field):
    """Build the Ising chain Hamiltonian as a dense matrix."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        # Build Z_i Z_{i+1}
        ops = [I2] * n
        ops[i] = Z
        ops[i+1] = Z
        H += -coupling * kron(*ops)
    
    # X terms: -field * sum X_i
    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        H += -field * kron(*ops)
    
    return H

def exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])

def apply_circuit(state, theta, n, layers):
    """Apply the fixed circuit to the state vector.
    
    theta: (layers, n) array of Ry angles
    Between layers, apply CZ on every adjacent pair in ascending order.
    No entangler after final layer.
    """
    state = state.copy()
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to every qubit i
        for i in range(n):
            # Apply Ry to qubit i
            R = ry(theta[layer, i])
            # Build the full operator
            ops = [I2] * n
            ops[i] = R
            full_op = kron(*ops)
            state = full_op @ state
        
        # Apply CZ between layers (not after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                # Apply CZ to qubits i and i+1
                ops = [I2] * n
                ops[i] = None  # placeholder
                ops[i+1] = None
                # Build CZ on qubits i and i+1
                cz_full = np.eye(4, dtype=complex)
                cz_full[3, 3] = -1  # CZ in computational basis
                # Need to embed this into the full Hilbert space
                # Simpler: build the full operator
                full_op = I2.copy()
                for j in range(n):
                    if j == i:
                        # This is part of the CZ pair
                        pass
                    elif j == i + 1:
                        pass
                    else:
                        full_op = np.kron(full_op, I2)
                # Actually, let's build it more carefully
                # The CZ acts on qubits i and i+1
                # We need to construct the full operator
                full_op = np.array([[1]], dtype=complex)
                for j in range(n):
                    if j < i:
                        full_op = np.kron(full_op, I2)
                    elif j == i:
                        # CZ acts on qubits i and i+1, so we need to handle them together
                        # Insert CZ as a 4x4 block, but we're going sequentially
                        # This approach is tricky; let's use a different method
                        pass
                    elif j == i + 1:
                        pass
                    else:
                        full_op = np.kron(full_op, I2)
                
                # Let me use a cleaner approach: build the full CZ operator
                cz_full = np.zeros((2**n, 2**n), dtype=complex)
                for basis_idx in range(2**n):
                    # Extract bits for qubits i and i+1
                    bit_i = (basis_idx >> (n - 1 - i)) & 1
                    bit_ip1 = (basis_idx >> (n - 1 - (i + 1))) & 1
                    sign = -1 if (bit_i == 1 and bit_ip1 == 1) else 1
                    cz_full[basis_idx, basis_idx] = sign
                
                state = cz_full @ state
    
    return state

def energy_expectation(theta, n, coupling, field, H, ground_energy):
    """Compute the mean energy gap per spin."""
    # Start from |0>^n
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    state = apply_circuit(state, theta, n, 3)
    
    # Compute <psi|H|psi>
    energy = np.vdot(state, H @ state).real
    
    # Mean energy gap per spin
    return (energy - ground_energy) / n

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    # Get exact ground energy
    ground_energy = exact_ground_energy(n, coupling, field)
    
    # Build Hamiltonian
    H = build_hamiltonian(n, coupling, field)
    
    # Initial guess: try a few different starting points
    best_theta = None
    best_energy = np.inf
    
    # Try several random initializations
    rng = np.random.RandomState(42)
    
    for trial in range(20):
        # Initialize with random angles
        if trial == 0:
            theta0 = np.zeros((layers, n))
        elif trial == 1:
            theta0 = np.full((layers, n), np.pi / 4)
        elif trial == 2:
            theta0 = np.full((layers, n), np.pi / 2)
        elif trial == 3:
            theta0 = np.full((layers, n), -np.pi / 4)
        elif trial == 4:
            theta0 = rng.uniform(-np.pi / 2, np.pi / 2, (layers, n))
        else:
            theta0 = rng.uniform(-np.pi, np.pi, (layers, n))
        
        def objective(x):
            theta_flat = x.reshape(layers, n)
            return energy_expectation(theta_flat, n, coupling, field, H, ground_energy)
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(objective, theta0.flatten(), method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.success or result.fun < best_energy:
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_theta = result.x.reshape(layers, n)
        except Exception:
            continue
    
    if best_theta is None:
        # Fallback: return product |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_theta
# EVOLVE-BLOCK-END
