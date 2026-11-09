"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

# Cache for exact ground energies
_ground_energy_cache = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix for an open chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    # Identity
    I = np.eye(dim)
    # Pauli Z
    Z = np.diag([1, -1])
    # Pauli X
    X = np.array([[0, 1], [1, 0]])
    
    # Build Z_i Z_{i+1} terms
    for i in range(n - 1):
        term = I
        # Apply Z to qubit i
        Zi = np.kron(np.eye(1 << i), Z)
        Zi = np.kron(Zi, np.eye(1 << (n - 1 - i)))
        # Apply Z to qubit i+1
        Zij = np.kron(np.eye(1 << i), Z)
        Zij = np.kron(Zij, Z)
        Zij = np.kron(Zij, np.eye(1 << (n - 2 - i)))
        H -= coupling * np.kron(Zi, np.eye(1)) if False else 0  # placeholder
        
    # Better approach: use bit manipulation for efficiency
    H = np.zeros((dim, dim), dtype=float)
    for state in range(dim):
        # Transverse field term: -field * sum X_i
        # X_i flips bit i
        for i in range(n):
            neighbor = state ^ (1 << i)
            H[state, neighbor] -= field
        
        # Coupling term: -coupling * sum Z_i Z_{i+1}
        # Z_i = +1 if bit i is 0, -1 if bit i is 1
        # Z_i Z_{i+1} = (+1)(+1)=1 if both 0 or both 1, (-1)(-1)=1 if both 1... 
        # Wait: Z|0>=+|0>, Z|1>=-|1>
        # So Z_i eigenvalue is +1 if bit i=0, -1 if bit i=1
        # Z_i Z_{i+1} eigenvalue is product
        zz_product = 1.0
        for i in range(n - 1):
            zi = 1.0 if (state >> i) & 1 == 0 else -1.0
            zij = 1.0 if (state >> (i + 1)) & 1 == 0 else -1.0
            zz_product = zi * zij
            H[state, state] -= coupling * zz_product
            break  # This is wrong, need sum
    
    # Redo properly
    H = np.zeros((dim, dim), dtype=float)
    for state in range(dim):
        # Diagonal: coupling terms
        diag_val = 0.0
        for i in range(n - 1):
            zi = 1.0 if (state >> i) & 1 == 0 else -1.0
            zij = 1.0 if (state >> (i + 1)) & 1 == 0 else -1.0
            diag_val -= coupling * zi * zij
        H[state, state] += diag_val
        
        # Off-diagonal: field terms
        for i in range(n):
            neighbor = state ^ (1 << i)
            H[state, neighbor] -= field
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy."""
    key = (n, coupling, field)
    if key in _ground_energy_cache:
        return _ground_energy_cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    ground_energy = eigenvalues[0]
    _ground_energy_cache[key] = ground_energy
    return ground_energy


def _circuit_state_vector(n, params, layers):
    """
    Apply the circuit to |0>^n.
    params shape: (layers, n)
    Each layer: Ry(theta) on all qubits, then CZ on adjacent pairs (except after last layer).
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            theta = params[layer, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # Ry = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
            # Apply to qubit i
            new_state = np.zeros_like(state)
            for idx in range(dim):
                if (idx >> i) & 1 == 0:
                    # bit i is 0
                    idx1 = idx | (1 << i)  # bit i becomes 1
                    new_state[idx] = c * state[idx] + s * state[idx1]
                    new_state[idx1] = s * state[idx] + c * state[idx1]
                # else handled when we hit the 0 case
                
            # Wait, this double-counts. Let me be more careful.
            # For each pair (idx0, idx1) where idx1 = idx0 | (1<<i) and bit i of idx0 is 0:
            # new[idx0] = c*old[idx0] + s*old[idx1]
            # new[idx1] = s*old[idx0] + c*old[idx1]
            
            new_state = np.zeros_like(state)
            for idx0 in range(dim):
                if (idx0 >> i) & 1 == 0:
                    idx1 = idx0 | (1 << i)
                    new_state[idx0] = c * state[idx0] + s * state[idx1]
                    new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state
        
        # Apply CZ on adjacent pairs (if not last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ: |11> -> -|11>
                for idx in range(dim):
                    if ((idx >> i) & 1) == 1 and ((idx >> (i + 1)) & 1) == 1:
                        state[idx] *= -1
    
    return state


def _circuit_energy(n, params, coupling, field, layers):
    """Compute the energy of the circuit state."""
    state = _circuit_state_vector(n, params, layers)
    H = _build_hamiltonian(n, coupling, field)
    energy = np.real(np.vdot(state, H @ state))
    return energy


def _objective(params, n, coupling, field, layers, ground_energy):
    """Objective function: mean energy gap per spin."""
    energy = _circuit_energy(n, params, coupling, field, layers)
    gap = (energy - ground_energy) / n
    return gap


def angles(n, coupling, field, layers):
    # Start time
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    # Get exact ground energy
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    # Initial guess: try a few starting points
    best_params = None
    best_energy = float('inf')
    
    # Try several initializations
    inits = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 4),
        np.full((layers, n), np.pi / 2),
        np.random.uniform(-np.pi, np.pi, (layers, n)),
        np.random.uniform(-np.pi, np.pi, (layers, n)),
    ]
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        # Flatten for scipy
        x0 = init.flatten()
        
        # Use L-BFGS-B with analytic gradient via finite differences (scipy handles it)
        # But let's try to be efficient
        try:
            result = opt.minimize(
                lambda x: _objective(x.reshape(layers, n), n, coupling, field, layers, ground_energy),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 5000}
            )
            
            if result.success or result.fun < best_energy:
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_params = result.x.reshape(layers, n)
                    
        except Exception:
            pass
    
    # If we still have time, do more random restarts
    while time.time() < deadline - 5:
        x0 = np.random.uniform(-np.pi, np.pi, (layers, n)).flatten()
        try:
            result = opt.minimize(
                lambda x: _objective(x.reshape(layers, n), n, coupling, field, layers, ground_energy),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Return best params
    if best_params is None:
        # Fallback: simple |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_params
# EVOLVE-BLOCK-END
