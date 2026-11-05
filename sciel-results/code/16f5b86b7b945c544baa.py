"""Quantum information: shallow-circuit Ising ground states
Uses sparse Hamiltonian and direct energy minimization via scipy optimizer.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import lil_matrix, diags, kron
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
import itertools


def _build_hamiltonian(n, coupling, field):
    """Build sparse Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i for open chain."""
    # Single-qubit operators
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Build full operators using kron
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= coupling * kron_all(ops)
    
    # X terms: -field * X_i
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_all(ops)
    
    return H


def _precompute_circuit_ops(n, layers):
    """Precompute all Ry and CZ operators for efficient state evolution."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    
    def ry(theta_val):
        c = np.cos(theta_val / 2)
        s = np.sin(theta_val / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    def build_single_qubit_op(op, qubit, n_qubits):
        """Build full operator for single-qubit op at position qubit."""
        ops = [I] * n_qubits
        ops[qubit] = op
        result = ops[0]
        for k in range(1, n_qubits):
            result = np.kron(result, ops[k])
        return result
    
    def build_cz_op(q1, q2, n_qubits):
        """Build CZ operator between qubits q1 and q2."""
        dim = 2 ** n_qubits
        cz_full = np.zeros((dim, dim), dtype=complex)
        for basis_idx in range(dim):
            bits = [(basis_idx >> k) & 1 for k in range(n_qubits)]
            if bits[q1] == 1 and bits[q2] == 1:
                cz_full[basis_idx, basis_idx] = -1
            else:
                cz_full[basis_idx, basis_idx] = 1
        return cz_full
    
    # Precompute CZ operators (they don't depend on parameters)
    cz_ops = {}
    for i in range(n - 1):
        cz_ops[i] = build_cz_op(i, i + 1, n)
    
    return cz_ops


def _apply_circuit_fast(state, theta, n, cz_ops):
    """Apply the 3-layer Ry/CZ circuit to initial state |0>^n.
    theta has shape (layers, n). Returns final state vector.
    """
    layers = theta.shape[0]
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    
    def ry(theta_val):
        c = np.cos(theta_val / 2)
        s = np.sin(theta_val / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    def build_single_qubit_op(op, qubit, n_qubits):
        """Build full operator for single-qubit op at position qubit."""
        ops = [I] * n_qubits
        ops[qubit] = op
        result = ops[0]
        for k in range(1, n_qubits):
            result = np.kron(result, ops[k])
        return result
    
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to each qubit i
        for i in range(n):
            op = ry(theta[layer, i])
            full_op = build_single_qubit_op(op, i, n)
            state = full_op @ state
        
        # Apply CZ on adjacent pairs in ascending order, except after final layer
        if layer < layers - 1:
            for i in range(n - 1):
                state = cz_ops[i] @ state
    
    return state


def _energy(state, H):
    """Compute expectation value <state|H|state>."""
    return np.real(np.vdot(state, H @ state))


def _objective(params, n, coupling, field, H, cz_ops):
    """Objective function: mean energy per spin."""
    theta = params.reshape(3, n)
    state = _apply_circuit_fast(theta, n, cz_ops)
    e = _energy(state, H)
    return e / n


def _mean_field_angles(n, coupling, field):
    """Get approximate angles from mean-field theory for transverse field Ising model."""
    # For open chain with transverse field, approximate solution
    # In the thermodynamic limit, the ground state is a product state with
    # <Z> = sqrt(1 - (h/J)^2) for h/J < 1, and 0 otherwise
    # For our case, let's use a simple heuristic
    angles = np.zeros((3, n))
    
    # The last layer should rotate towards the ground state
    # For small fields, ground state is close to |0>^n (all Z = +1)
    # For large fields, ground state has more X component
    
    ratio = field / coupling if coupling > 0 else 1.0
    
    if ratio < 1.0:
        # Below critical point, ground state has Z polarization
        # Approximate angle to rotate from |0> to the ground state
        # <Z> ~ sqrt(1 - ratio^2), so angle ~ arccos(sqrt(1-ratio^2))
        target_z = np.sqrt(max(0, 1 - ratio**2))
        angle = np.arccos(target_z)
        angles[-1] = angle
    else:
        # Above critical point, ground state is more like |+>^n
        angles[-1] = np.pi / 2
    
    return angles


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy per spin.
    
    Returns array of shape (layers, n) with optimized angles.
    """
    # Build Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    
    # Precompute CZ operators
    cz_ops = _precompute_circuit_ops(n, layers)
    
    # Get exact ground state energy for reference
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0].real
    
    # Initial guesses
    candidates = []
    
    # Initial guess 1: mean-field based
    candidates.append(_mean_field_angles(n, coupling, field))
    
    # Initial guess 2: product |+> state
    theta_plus = np.zeros((layers, n))
    theta_plus[-1] = np.pi / 2
    candidates.append(theta_plus)
    
    # Initial guess 3: all zeros
    candidates.append(np.zeros((layers, n)))
    
    # Initial guess 4: small random perturbations
    rng = np.random.RandomState(42)
    for _ in range(10):
        cand = rng.uniform(-np.pi/3, np.pi/3, (layers, n))
        candidates.append(cand)
    
    # Initial guess 5: varying last layer
    for trial in range(5):
        cand = np.zeros((layers, n))
        base = 0.3 + 0.1 * trial
        cand[-1] = base + rng.uniform(-0.1, 0.1, n)
        candidates.append(cand)
    
    best_theta = candidates[0].copy()
    best_energy = np.inf
    
    for theta_init in candidates:
        params_init = theta_init.flatten()
        
        # Use L-BFGS-B optimizer
        try:
            result = minimize(
                _objective,
                params_init,
                args=(n, coupling, field, H, cz_ops),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7, 'maxls': 50}
            )
            e = _objective(result.x, n, coupling, field, H, cz_ops)
            if e < best_energy:
                best_energy = e
                best_theta = result.x.reshape(layers, n)
        except Exception:
            continue
    
    # Clamp angles to reasonable range
    best_theta = np.clip(best_theta, -np.pi, np.pi)
    
    return best_theta
# EVOLVE-BLOCK-END
