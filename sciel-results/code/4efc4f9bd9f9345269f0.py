"""Optimized program for shallow-circuit Ising ground state approximation."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J*sum Z_i Z_{i+1} - h*sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        Zi = np.zeros((dim, dim), dtype=complex)
        Zj = np.zeros((dim, dim), dtype=complex)
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_j = (state >> (i + 1)) & 1
            zi = 1 - 2 * bit_i
            zj = 1 - 2 * bit_j
            Zi[state, state] = zi
            Zj[state, state] = zj
        H -= coupling * (Zi @ Zj)
    # X_i terms
    for i in range(n):
        Xi = np.zeros((dim, dim), dtype=complex)
        for state in range(dim):
            flipped = state ^ (1 << i)
            Xi[flipped, state] = 1.0
        H -= field * Xi
    return H

def _circuit_state(theta, n, layers):
    """Prepare the circuit state |psi> from angles theta (shape: layers x n)."""
    dim = 1 << n
    # Start with |0>^n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    
    # Ry rotation matrix
    def ry_matrix(angle):
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to each qubit i
        for i in range(n):
            angle = theta[layer, i]
            if abs(angle) < 1e-15:
                continue
            # Build full Ry operator for qubit i
            Ry_full = np.zeros((dim, dim), dtype=complex)
            for state in range(dim):
                bit = (state >> i) & 1
                if bit == 0:
                    Ry_full[state, state] = np.cos(angle / 2)
                    Ry_full[state ^ (1 << i), state] = -np.sin(angle / 2)
                else:
                    Ry_full[state, state] = np.cos(angle / 2)
                    Ry_full[state ^ (1 << i), state] = np.sin(angle / 2)
            psi = Ry_full @ psi
        
        # Apply CZ on adjacent pairs (ascending order), except after last layer
        if layer < layers - 1:
            for i in range(n - 1):
                CZ_full = np.zeros((dim, dim), dtype=complex)
                for state in range(dim):
                    bit_i = (state >> i) & 1
                    bit_j = (state >> (i + 1)) & 1
                    if bit_i == 1 and bit_j == 1:
                        CZ_full[state, state] = -1.0
                    else:
                        CZ_full[state, state] = 1.0
                psi = CZ_full @ psi
    
    return psi

def _circuit_state_fast(theta, n, layers):
    """Faster circuit state preparation using tensor operations."""
    dim = 1 << n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            angle = theta[layer, i]
            c = np.cos(angle / 2)
            s = np.sin(angle / 2)
            # Build 2x2 block structure
            for state in range(dim):
                bit = (state >> i) & 1
                partner = state ^ (1 << i)
                if bit == 0:
                    new_val = c * psi[state] - s * psi[partner]
                    psi[state] = new_val
                elif bit == 1:
                    # Don't update partner here; it will be handled when we process it
                    pass
            # Now fix the partner entries
            for state in range(dim):
                bit = (state >> i) & 1
                if bit == 1:
                    partner = state ^ (1 << i)
                    psi[state] = s * psi[partner] + c * psi[state]
        
        # Apply CZ gates
        if layer < layers - 1:
            for i in range(n - 1):
                for state in range(dim):
                    bit_i = (state >> i) & 1
                    bit_j = (state >> (i + 1)) & 1
                    if bit_i == 1 and bit_j == 1:
                        psi[state] *= -1.0
    
    return psi

def _energy(psi, H):
    """Compute <psi|H|psi>."""
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    return energy

def _objective(theta_flat, n, coupling, field, H, E0, layers):
    """Objective function: (circuit_energy - E0) / n."""
    theta = theta_flat.reshape(layers, n)
    psi = _circuit_state_fast(theta, n, layers)
    e = _energy(psi, H)
    return (e - E0) / n

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate Ising ground state."""
    t_start = time.time()
    
    # Build Hamiltonian and get exact ground energy
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    E0 = eigvals[0].real
    
    dim = 1 << n
    n_params = layers * n
    
    # Initial guesses: try several starting points
    # Good initial guess: all angles = pi/2 (|+> state) as baseline
    best_theta = np.full((layers, n), np.pi / 2)
    best_obj = _objective(best_theta.ravel(), n, coupling, field, H, E0, layers)
    
    # Generate multiple random initializations
    rng = np.random.RandomState(42)
    initializations = [np.full((layers, n), np.pi / 2)]
    for _ in range(15):
        init = rng.uniform(-np.pi, np.pi, size=(layers, n))
        initializations.append(init)
    # Also try symmetric patterns
    for base in [0.3, 0.5, 0.7, 1.0]:
        init = np.full((layers, n), base)
        initializations.append(init)
    
    # Optimize each initialization
    for init in initializations:
        if time.time() - t_start > _TIME_LIMIT - 10:
            break
        
        x0 = init.ravel()
        
        # L-BFGS-B optimization
        try:
            res = minimize(
                _objective, x0,
                args=(n, coupling, field, H, E0, layers),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * n_params,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass
        
        # Nelder-Mead polish
        if time.time() - t_start > _TIME_LIMIT - 5:
            break
        
        try:
            res_nm = minimize(
                _objective, best_theta.ravel(),
                args=(n, coupling, field, H, E0, layers),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < best_obj:
                best_obj = res_nm.fun
                best_theta = res_nm.x.reshape(layers, n)
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() - t_start < _TIME_LIMIT - 3:
        try:
            res_final = minimize(
                _objective, best_theta.ravel(),
                args=(n, coupling, field, H, E0, layers),
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_final.fun < best_obj:
                best_obj = res_final.fun
                best_theta = res_final.x.reshape(layers, n)
        except Exception:
            pass
    
    # Ensure output is finite
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.full((layers, n), np.pi / 2)
    
    return best_theta
# EVOLVE-BLOCK-END
