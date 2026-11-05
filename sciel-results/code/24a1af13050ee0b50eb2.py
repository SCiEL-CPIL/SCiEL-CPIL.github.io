"""Tensor-network / exact-diagonalization hybrid optimizer for 3-layer Ry/CZ Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

# Precompute Pauli matrices and single-qubit ops
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)

def _ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def _cz():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=np.complex128)

def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix for an open transverse-field Ising chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        # Build Z_i Z_{i+1}
        for bit in range(dim):
            # Get bits
            bi = (bit >> i) & 1
            bi1 = (bit >> (i + 1)) & 1
            # Z eigenvalue: +1 if bit=0, -1 if bit=1
            zi = 1 if bi == 0 else -1
            zi1 = 1 if bi1 == 0 else -1
            H[bit, bit] -= coupling * zi * zi1
    
    # X terms: -field * X_i
    for i in range(n):
        for bit in range(dim):
            bi = (bit >> i) & 1
            # X flips bit i
            bit_flip = bit ^ (1 << i)
            H[bit_flip, bit] -= field
    
    return H

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def _circuit_state_vector(theta, n, layers):
    """
    Apply the circuit to |0>^n and return the state vector.
    theta has shape (layers, n).
    Layer k: apply Ry(theta[k, i]) to each qubit i.
    Between layers k and k+1: apply CZ on every adjacent pair in ascending order.
    No entangler after final layer.
    """
    dim = 1 << n
    # Start with |0>^n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            U = _ry(theta[layer, i])
            # Apply U to qubit i
            new_state = np.zeros(dim, dtype=np.complex128)
            for bit in range(dim):
                bi = (bit >> i) & 1
                if bi == 0:
                    # |0> -> U[0,0]|0> + U[1,0]|1>
                    bit0 = bit
                    bit1 = bit | (1 << i)
                    new_state[bit0] += U[0, 0] * state[bit0] + U[0, 1] * state[bit1]
                    new_state[bit1] += U[1, 0] * state[bit0] + U[1, 1] * state[bit1]
                # We handle both cases via the pair iteration; but this double-counts.
                # Let's use a cleaner approach: iterate over pairs.
            # Reset and do it properly
            new_state = np.zeros(dim, dtype=np.complex128)
            for bit in range(dim):
                bi = (bit >> i) & 1
                if bi == 0:
                    bit0 = bit
                    bit1 = bit | (1 << i)
                    new_state[bit0] = U[0, 0] * state[bit0] + U[0, 1] * state[bit1]
                    new_state[bit1] = U[1, 0] * state[bit0] + U[1, 1] * state[bit1]
            state = new_state
        
        # Apply CZ between layers (not after last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                # Apply CZ on qubits i and i+1
                CZ = _cz()
                new_state = np.zeros(dim, dtype=np.complex128)
                for bit in range(dim):
                    bi = (bit >> i) & 1
                    bi1 = (bit >> (i + 1)) & 1
                    # CZ only affects |11> -> -|11>
                    if bi == 1 and bi1 == 1:
                        new_state[bit] = -state[bit]
                    else:
                        new_state[bit] = state[bit]
                state = new_state
    
    return state

def _energy_from_state(state, n, coupling, field):
    """Compute <psi|H|psi> from state vector."""
    H = _build_hamiltonian(n, coupling, field)
    return np.real(np.vdot(state, H @ state))

def _objective(theta_flat, n, coupling, field, layers, ground_energy):
    """Objective function: mean energy gap per spin."""
    theta = theta_flat.reshape(layers, n)
    state = _circuit_state_vector(theta, n, layers)
    energy = _energy_from_state(state, n, coupling, field)
    return (energy - ground_energy) / n

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    Returns theta with shape (layers, n) in radians.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Get exact ground energy
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    # Initial guess: the seed program's approach
    theta0 = np.zeros((layers, n))
    theta0[-1] = np.pi / 2
    theta0_flat = theta0.flatten()
    
    # Try multiple random restarts with gradient-based optimization
    best_theta_flat = theta0_flat.copy()
    best_val = _objective(theta0_flat, n, coupling, field, layers, ground_energy)
    
    np.random.seed(42)
    
    # Number of restarts we can afford
    max_restarts = 20
    attempt = 0
    
    while attempt < max_restarts:
        if time.time() - start_time > deadline:
            break
        
        # Random initialization
        theta_rand = np.random.uniform(-np.pi, np.pi, size=(layers, n))
        # Keep last layer biased toward pi/2
        theta_rand[-1] = np.pi / 2 + np.random.uniform(-0.5, 0.5, size=n)
        theta_rand_flat = theta_rand.flatten()
        
        val = _objective(theta_rand_flat, n, coupling, field, layers, ground_energy)
        if val < best_val:
            best_theta_flat = theta_rand_flat.copy()
            best_val = val
        
        # Try L-BFGS-B from this starting point
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            res = minimize(
                _objective,
                theta_rand_flat,
                args=(n, coupling, field, layers, ground_energy),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success and res.fun < best_val:
                best_theta_flat = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
        
        attempt += 1
    
    # If we still have time, do a final polish from the best point
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                _objective,
                best_theta_flat,
                args=(n, coupling, field, layers, ground_energy),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.success and res.fun < best_val:
                best_theta_flat = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    return best_theta_flat.reshape(layers, n)
# EVOLVE-BLOCK-END
