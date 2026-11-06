"""Variational ansatz: fit circuit angles to exact ground-state amplitudes via least-squares."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares


def _exact_ground_state(n, coupling, field):
    """Compute the exact ground state of the open transverse-field Ising chain.
    H = -coupling * sum_i Z_i Z_{i+1} - field * sum_i X_i
    Qubit i is bit i (little-endian convention for state indexing).
    Returns the ground state vector as a complex array of length 2**n.
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Build Hamiltonian
    for basis in range(dim):
        # Z_i Z_{i+1} terms
        zz_energy = 0.0
        for i in range(n - 1):
            bit_i = (basis >> i) & 1
            bit_ip1 = (basis >> (i + 1)) & 1
            # Z eigenvalue: |0> -> +1, |1> -> -1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz_energy += z_i * z_ip1
        H[basis, basis] += -coupling * zz_energy
        
        # X_i terms: flip bit i
        for i in range(n):
            flipped = basis ^ (1 << i)
            H[flipped, basis] += -field
    
    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_idx = np.argmin(eigvals)
    return eigvecs[:, ground_idx], eigvals[ground_idx]


def _apply_circuit(state, angles, n):
    """Apply the 3-layer Ry/CZ circuit to the initial |0>^n state.
    angles: shape (layers, n)
    Each layer: apply Ry(theta[i]) to all qubits, then CZ on adjacent pairs (except after last layer).
    """
    layers = angles.shape[0]
    state = state.copy()
    
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            theta = angles[layer, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            # Ry matrix: [[c, -s], [s, c]]
            # Apply to qubit i
            new_state = np.zeros_like(state)
            for basis in range(2 ** n):
                bit_i = (basis >> i) & 1
                if bit_i == 0:
                    # |0> -> c|0> + s|1>
                    new_state[basis] += c * state[basis]
                    new_state[basis | (1 << i)] += s * state[basis]
                else:
                    # |1> -> -s|0> + c|1>
                    new_state[basis & ~(1 << i)] += -s * state[basis]
                    new_state[basis] += c * state[basis]
            state = new_state
        
        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ: |11> -> -|11>
                for basis in range(2 ** n):
                    bit_i = (basis >> i) & 1
                    bit_ip1 = (basis >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        state[basis] *= -1
    
    return state


def _circuit_state(angles, n):
    """Compute the state produced by the circuit starting from |0>^n."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    return _apply_circuit(state, angles, n)


def _energy_of_state(state, n, coupling, field):
    """Compute the energy <psi|H|psi> for a given state."""
    dim = 2 ** n
    energy = 0.0
    
    for basis in range(dim):
        # Z_i Z_{i+1} terms
        zz_energy = 0.0
        for i in range(n - 1):
            bit_i = (basis >> i) & 1
            bit_ip1 = (basis >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz_energy += z_i * z_ip1
        energy += -coupling * zz_energy * abs(state[basis]) ** 2
        
        # X_i terms: off-diagonal
        for i in range(n):
            flipped = basis ^ (1 << i)
            if flipped > basis:  # count each pair once
                energy += -field * 2 * np.real(state[flipped].conj() * state[basis])
    
    return energy


def _residuals(params, n, coupling, field, target_state):
    """Compute residuals for least-squares: difference between circuit state and target state."""
    angles = params.reshape((3, n))
    circuit_state = _circuit_state(angles, n)
    
    # Residuals: real and imaginary parts of (circuit - target)
    diff = circuit_state - target_state
    return np.concatenate([diff.real, diff.imag])


def _fourier_init(n, target_state, num_guesses=5):
    """Generate initial guesses based on dominant Fourier components of the ground state."""
    dim = 2 ** n
    # Take the real part of the state (assuming real ground state for Ising model)
    real_state = target_state.real.copy()
    
    # FFT to find dominant frequencies
    fft_coeffs = np.fft.fft(real_state)
    magnitudes = np.abs(fft_coeffs)
    
    # Get top frequency indices
    top_indices = np.argsort(magnitudes)[::-1][:num_guesses]
    
    inits = []
    for idx in top_indices:
        # Derive initial angles from the phase/frequency structure
        freq = idx / dim
        # Map frequency to a simple angle pattern
        base_angle = 2 * np.pi * freq
        init = np.zeros((3, n))
        for layer in range(3):
            for i in range(n):
                init[layer, i] = base_angle * (i + 1) / (n + 1) + layer * 0.1
        inits.append(init.flatten())
    
    # Also add a uniform initialization
    inits.append(np.zeros(3 * n))
    # And a pi/2 initialization
    init_pi2 = np.zeros((3, n))
    init_pi2[-1] = np.pi / 2
    inits.append(init_pi2.flatten())
    
    return inits


def angles(n, coupling, field, layers):
    """Return circuit angles (layers, n) that approximate the ground state."""
    # Compute exact ground state
    target_state, ground_energy = _exact_ground_state(n, coupling, field)
    
    # Generate initial guesses
    inits = _fourier_init(n, target_state, num_guesses=5)
    
    best_params = None
    best_cost = np.inf
    
    for init in inits:
        try:
            result = least_squares(
                _residuals,
                init,
                args=(n, coupling, field, target_state),
                method='trf',
                max_nfev=1000,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12
            )
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_params = result.x
        except Exception:
            continue
    
    if best_params is None:
        # Fallback: return zeros
        return np.zeros((layers, n))
    
    return best_params.reshape((layers, n))
# EVOLVE-BLOCK-END
