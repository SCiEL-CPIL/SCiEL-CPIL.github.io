"""Tensor-network-inspired parameter initialization for shallow Ry/CZ Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _build_circuit_state(theta, n, layers):
    """
    Build the statevector for the fixed circuit.
    theta: (layers, n) array of Ry angles.
    Circuit: start |0>^n, then for each layer:
      - apply Ry(theta[layer, i]) to qubit i
      - if not the last layer, apply CZ on every adjacent pair (i, i+1) in ascending order.
    """
    # Statevector as complex array of size 2^n
    # Qubit i is bit i (little-endian: qubit 0 is LSB)
    # Start |0>^n
    dim = 1 << n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    def apply_ry(psi, qubit, theta):
        """Apply Ry(theta) to a specific qubit."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        # Ry = [[c, -s], [s, c]]
        new_psi = np.zeros_like(psi)
        for idx in range(dim):
            bit = (idx >> qubit) & 1
            if bit == 0:
                # |0> -> c|0> + s|1>
                idx1 = idx | (1 << qubit)
                new_psi[idx] += c * psi[idx]
                new_psi[idx1] += s * psi[idx]
            else:
                # |1> -> -s|0> + c|1>
                idx0 = idx & ~(1 << qubit)
                new_psi[idx0] += -s * psi[idx]
                new_psi[idx] += c * psi[idx]
        return new_psi
    
    def apply_cz(psi, q1, q2):
        """Apply CZ to qubits q1 and q2."""
        # CZ = diag(1, 1, 1, -1) in basis |q1 q2>
        new_psi = psi.copy()
        mask = ((np.arange(dim) >> q1) & 1) & ((np.arange(dim) >> q2) & 1)
        new_psi[mask] *= -1
        return new_psi
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for i in range(n):
            psi = apply_ry(psi, i, theta[layer, i])
        # Apply CZ between layers (not after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                psi = apply_cz(psi, i, i + 1)
    
    return psi

def _exact_ground_state(n, coupling, field):
    """
    Compute the exact ground state of H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    Open chain, qubit i is bit i.
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Build Hamiltonian
    for idx in range(dim):
        # Diagonal part: -coupling * sum Z_i Z_{i+1}
        diag_val = 0.0
        for i in range(n - 1):
            zi = 1 - 2 * ((idx >> i) & 1)
            zi1 = 1 - 2 * ((idx >> (i + 1)) & 1)
            diag_val += -coupling * zi * zi1
        H[idx, idx] += diag_val
        
        # Off-diagonal part: -field * sum X_i
        for i in range(n):
            idx_flip = idx ^ (1 << i)
            H[idx, idx_flip] += -field
    
    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_idx = np.argmin(eigvals)
    ground_state = eigvecs[:, ground_idx]
    ground_energy = eigvals[ground_idx]
    
    return ground_state, ground_energy

def _energy(psi, n, coupling, field):
    """Compute expectation value of H for state psi."""
    # H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    dim = 1 << n
    energy = 0.0
    
    # Diagonal part
    for i in range(n - 1):
        for idx in range(dim):
            zi = 1 - 2 * ((idx >> i) & 1)
            zi1 = 1 - 2 * ((idx >> (i + 1)) & 1)
            energy += -coupling * zi * zi1 * np.abs(psi[idx])**2
    
    # Off-diagonal part: -field * sum X_i
    # <psi| X_i |psi> = psi[idx]* conj(psi[idx^2^i]) + psi[idx^2^i]* conj(psi[idx])
    # Actually: <psi| X_i |psi> = sum_idx psi[idx]* conj(psi[idx^2^i]) * 1 (since X flips)
    # More carefully: X_i |idx> = |idx ^ 2^i>
    # So <psi| X_i |psi> = sum_idx conj(psi[idx]) * psi[idx ^ 2^i]
    for i in range(n):
        for idx in range(dim):
            idx_flip = idx ^ (1 << i)
            energy += -field * (np.conj(psi[idx]) * psi[idx_flip])
    
    return np.real(energy)

def _objective(theta_flat, n, coupling, field, layers, ground_energy):
    """Objective: mean energy gap per spin = (circuit_energy - ground_energy) / n"""
    theta = theta_flat.reshape((layers, n))
    psi = _build_circuit_state(theta, n, layers)
    e_circuit = _energy(psi, n, coupling, field)
    return (e_circuit - ground_energy) / n

def angles(n, coupling, field, layers):
    """
    Return optimal angles for the fixed circuit.
    """
    # First, get the exact ground state and energy
    ground_state, ground_energy = _exact_ground_state(n, coupling, field)
    
    # Initialize with a reasonable guess: start from |+> state (theta = pi/2 on last layer)
    theta_init = np.zeros((layers, n))
    theta_init[-1] = np.pi / 2
    
    # Use differential evolution for global optimization
    bounds = [(-np.pi, np.pi)] * (layers * n)
    result = differential_evolution(
        _objective,
        bounds=bounds,
        args=(n, coupling, field, layers, ground_energy),
        seed=42,
        maxiter=100,
        tol=1e-10,
        polish=True
    )
    
    theta_opt = result.x.reshape((layers, n))
    
    # Additional local refinement with L-BFGS-B
    res2 = minimize(
        _objective,
        theta_opt.flatten(),
        args=(n, coupling, field, layers, ground_energy),
        method='L-BFGS-B',
        bounds=[(-np.pi, np.pi)] * (layers * n),
        options={'maxiter': 500, 'ftol': 1e-12}
    )
    theta_final = res2.x.reshape((layers, n))
    
    # Ensure finite values
    if not np.all(np.isfinite(theta_final)):
        theta_final = theta_init
    
    return theta_final
# EVOLVE-BLOCK-END
