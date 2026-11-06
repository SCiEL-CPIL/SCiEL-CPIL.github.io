"""Optimize 3-layer Ry/CZ circuit for transverse-field Ising ground states."""
import numpy as np
from scipy.optimize import minimize
import time

def _simulate_circuit(n, theta, coupling=1.0):
    """Simulate the Ry/CZ circuit and return the state vector."""
    # Start with |0>^n
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    for layer in range(theta.shape[0]):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            t = theta[layer, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            # Ry = [[cos, -sin], [sin, cos]]
            # Apply to qubit i (bit i)
            new_state = np.zeros_like(state)
            for idx in range(2**n):
                bit = (idx >> i) & 1
                if bit == 0:
                    idx1 = idx | (1 << i)
                    new_state[idx] += c * state[idx]
                    new_state[idx1] += s * state[idx]
                else:
                    idx0 = idx & ~(1 << i)
                    new_state[idx0] += -s * state[idx]
                    new_state[idx] += c * state[idx]
            state = new_state
        
        # Apply CZ on every adjacent pair in ascending order (except after final layer)
        if layer < theta.shape[0] - 1:
            for i in range(n - 1):
                # CZ on qubits i and i+1
                for idx in range(2**n):
                    bit_i = (idx >> i) & 1
                    bit_ip1 = (idx >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        state[idx] = -state[idx]
    
    return state

def _compute_energy(state, n, coupling, field):
    """Compute <H> for the state. H = -coupling*sum Z_i Z_{i+1} - field*sum X_i"""
    energy = 0.0
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for idx in range(2**n):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            prob = np.abs(state[idx])**2
            energy += -coupling * z_i * z_ip1 * prob
    # X_i terms
    for i in range(n):
        for idx in range(2**n):
            bit = (idx >> i) & 1
            idx_flip = idx ^ (1 << i)
            # <X> = 2*Re(<idx|X|idx_flip>) = 2*Re(conj(state[idx]) * state[idx_flip])
            # But X|idx> = |idx_flip>, so <state|X|state> = sum conj(s[idx])*s[idx_flip]
            # Actually X = |0><1| + |1><0|, so <X_i> = 2*Re(conj(s[idx0])*s[idx1]) for each pair
            if bit == 0:
                idx1 = idx | (1 << i)
                energy += -field * 2 * np.real(np.conj(state[idx]) * state[idx1])
    return energy

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            H[idx, idx] += -coupling * z_i * z_ip1
    
    # X_i terms
    for i in range(n):
        for idx in range(dim):
            bit = (idx >> i) & 1
            if bit == 0:
                idx1 = idx | (1 << i)
                H[idx, idx1] += -field
                H[idx1, idx] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _physics_informed_init(n, field):
    """Initialize angles based on physics intuition."""
    theta = np.zeros((3, n))
    # At high field, qubits want to be in |+> (X eigenstate), so Ry ~ pi/2
    # At low field, qubits want to be in |0> or correlated, so Ry ~ 0
    # Smooth transition
    alpha = field / 2.0  # normalize
    theta[0, :] = np.pi / 4 * alpha
    theta[1, :] = np.pi / 4 * alpha
    theta[2, :] = np.pi / 2 * alpha
    return theta

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Compute exact ground energy
    e_ground = _exact_ground_energy(n, coupling, field)
    
    # Objective function
    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        state = _simulate_circuit(n, theta)
        e_circuit = _compute_energy(state, n, coupling, field)
        return (e_circuit - e_ground) / n
    
    # Try physics-informed initialization
    best_theta = None
    best_val = np.inf
    
    # Initial guess
    theta0 = _physics_informed_init(n, field)
    
    # Optimize with L-BFGS-B
    try:
        result = minimize(
            objective,
            theta0.flatten(),
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi)] * (layers * n),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success or result.fun < best_val:
            best_val = result.fun
            best_theta = result.x.reshape(layers, n)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 20:
        if best_theta is None:
            best_theta = theta0
        return best_theta
    
    # Try a few random restarts
    np.random.seed(42)
    for restart in range(3):
        if time.time() > deadline - 10:
            break
        theta_rand = np.random.uniform(-np.pi, np.pi, (layers, n))
        try:
            result = minimize(
                objective,
                theta_rand.flatten(),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_theta = result.x.reshape(layers, n)
        except Exception:
            pass
    
    if best_theta is None:
        best_theta = theta0
    
    return best_theta
