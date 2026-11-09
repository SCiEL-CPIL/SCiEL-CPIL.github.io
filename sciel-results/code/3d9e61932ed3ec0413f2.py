"""Optimized quantum circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    H = np.zeros((2**n, 2**n), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        op = np.array([[1]], dtype=complex)
        for j in range(n):
            if j == i or j == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H += -coupling * op
    
    # X terms: -field * X_i
    for i in range(n):
        op = np.array([[1]], dtype=complex)
        for j in range(n):
            if j == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H += -field * op
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def _circuit_state_vector(n, theta, layers):
    """
    Build the state vector from the circuit.
    theta: (layers, n) array of Ry angles
    Circuit: |0>^n, then for each layer l:
      - Apply Ry(theta[l, i]) to qubit i for all i
      - If not last layer, apply CZ on adjacent pairs (i, i+1) for i in ascending order
    """
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    for l in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            theta_i = theta[l, i]
            Ry = np.array([
                [np.cos(theta_i / 2), -np.sin(theta_i / 2)],
                [np.sin(theta_i / 2), np.cos(theta_i / 2)]
            ], dtype=complex)
            op = np.array([[1]], dtype=complex)
            for j in range(n):
                if j == i:
                    op = np.kron(op, Ry)
                else:
                    op = np.kron(op, I)
            state = op @ state
        
        # Apply CZ on adjacent pairs if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                # CZ on qubits i and i+1
                CZ = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, -1]], dtype=complex)
                op = np.array([[1]], dtype=complex)
                for j in range(n):
                    if j == i or j == i + 1:
                        op = np.kron(op, np.eye(2, dtype=complex))  # placeholder
                    else:
                        op = np.kron(op, np.eye(2, dtype=complex))
                
                # Build CZ operator properly
                # CZ acts on qubits i and i+1
                # We need to construct the full operator
                CZ_full = np.array([[1]], dtype=complex)
                for j in range(n):
                    if j == i:
                        CZ_full = np.kron(CZ_full, np.eye(2, dtype=complex))
                    elif j == i + 1:
                        CZ_full = np.kron(CZ_full, np.eye(2, dtype=complex))
                    else:
                        CZ_full = np.kron(CZ_full, np.eye(2, dtype=complex))
                
                # This approach is inefficient. Let me use a different method.
                # Build CZ as a tensor product with identity on other qubits
                CZ_op = np.array([[1]], dtype=complex)
                for j in range(n):
                    if j == i:
                        CZ_op = np.kron(CZ_op, np.array([[1, 0], [0, 0]], dtype=complex))
                    elif j == i + 1:
                        CZ_op = np.kron(CZ_op, np.array([[1, 0], [0, 0]], dtype=complex))
                    else:
                        CZ_op = np.kron(CZ_op, np.eye(2, dtype=complex))
                
                # Actually, CZ = |0><0|⊗I + |1><1|⊗Z
                # Let me build it properly
                CZ_op = np.zeros((2**n, 2**n), dtype=complex)
                # For each basis state, apply CZ
                for idx in range(2**n):
                    bits = [(idx >> (n-1-k)) & 1 for k in range(n)]
                    # CZ flips phase if both qubits i and i+1 are 1
                    if bits[i] == 1 and bits[i+1] == 1:
                        CZ_op[idx, idx] = -1
                    else:
                        CZ_op[idx, idx] = 1
                
                state = CZ_op @ state
    
    return state


def _circuit_energy(n, coupling, field, theta, layers):
    """Compute the energy of the circuit state."""
    H = _build_hamiltonian(n, coupling, field)
    state = _circuit_state_vector(n, theta, layers)
    energy = np.vdot(state, H @ state).real
    return float(energy)


def _objective(params, n, coupling, field, layers, ground_energy):
    """Objective function for optimization."""
    theta = params.reshape((layers, n))
    energy = _circuit_energy(n, coupling, field, theta, layers)
    return (energy - ground_energy) / n


def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate the ground state."""
    start_time = time.time()
    deadline = start_time + 110  # Leave safety margin
    
    # Precompute ground energy
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    # Initial guess: uniform angles
    best_params = np.zeros(layers * n)
    best_score = float('inf')
    
    # Try multiple initializations
    initializations = []
    
    # 1. All zeros
    initializations.append(np.zeros(layers * n))
    
    # 2. All pi/4
    initializations.append(np.full(layers * n, np.pi / 4))
    
    # 3. Last layer pi/2 (like seed but all layers)
    init3 = np.zeros(layers * n)
    init3[-n:] = np.pi / 2
    initializations.append(init3)
    
    # 4. Physics-based: for small field, angles near pi/2; for large field, smaller
    # The ground state of transverse field Ising: 
    # - At field=0: all Z aligned, so |0> or |1> states, theta=0
    # - At large field: |+> states, theta=pi/2
    # Interpolate based on field
    init4 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            # Simple heuristic
            init4[l * n + i] = np.pi / 2 * min(1.0, field)
    initializations.append(init4)
    
    # 5. Random
    np.random.seed(42)
    initializations.append(np.random.uniform(-np.pi/2, np.pi/2, layers * n))
    
    # 6. Another random
    np.random.seed(123)
    initializations.append(np.random.uniform(-np.pi/2, np.pi/2, layers * n))
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective,
                init,
                args=(n, coupling, field, layers, ground_energy),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'maxfev': 1000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Check if we should try L-BFGS-B from the best Nelder-Mead result
    if time.time() < deadline - 30:
        try:
            result2 = minimize(
                best_params,
                args=(n, coupling, field, layers, ground_energy),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={
                    'maxiter': 200,
                    'maxfun': 500,
                    'ftol': 1e-10
                }
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Check time and return best result
    if time.time() > deadline - 5:
        # Return immediately
        pass
    
    theta = best_params.reshape((layers, n))
    return theta
# EVOLVE-BLOCK-END
