"""Optimized 3-layer Ry/CZ circuit for transverse-field Ising ground states."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 115.0  # safety margin before 120s kill

def _build_hamiltonian(n, coupling, field):
    """Build the open transverse-field Ising Hamiltonian as a dense matrix."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bi1 = (state >> (i + 1)) & 1
            zz = (1 - 2 * bi) * (1 - 2 * bi1)
            H[state, state] += -coupling * zz
    # X terms: -field * X_i
    for i in range(n):
        mask = 1 << i
        for state in range(dim):
            partner = state ^ mask
            H[state, partner] += -field
    return H

def _circuit_energy(params, n, coupling, field, layers):
    """Compute <psi|H|psi> for the given circuit parameters.
    params: flat array of shape (layers*n,). In-place gates."""
    dim = 1 << n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0  # |0>^n
    
    # Precompute single-qubit Ry rotation matrices for each theta
    # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
    # We apply Ry to each qubit in-place.
    
    for layer in range(layers):
        # Apply Ry(theta) to each qubit
        for q in range(n):
            theta = params[layer * n + q]
            c = np.cos(theta * 0.5)
            s = np.sin(theta * 0.5)
            # Ry = [[c, -s], [s, c]]
            # For qubit q, we iterate over all basis states
            # Split by bit q
            mask = 1 << q
            # For each pair of states differing only in bit q
            for base in range(0, dim, 2 * mask):
                for offset in range(mask):
                    idx0 = base + offset       # bit q = 0
                    idx1 = idx0 + mask         # bit q = 1
                    a0 = psi[idx0]
                    a1 = psi[idx1]
                    psi[idx0] = c * a0 - s * a1
                    psi[idx1] = s * a0 + c * a1
        
        # Apply CZ on adjacent pairs (ascending order), except after last layer
        if layer < layers - 1:
            for q in range(n - 1):
                mask0 = 1 << q
                mask1 = 1 << (q + 1)
                both_mask = mask0 | mask1
                # CZ: |11> -> -|11>, others unchanged
                for base in range(0, dim, 4 * mask0):
                    for offset in range(mask0):
                        idx = base + offset
                        # Only flip sign when both bits are 1
                        if (idx & mask0) and (idx & mask1):
                            psi[idx] *= -1.0
    
    # Compute energy <psi|H|psi>
    # H is real symmetric, so E = real(psi.conj() @ H @ psi)
    Hp = _H_cache @ psi
    energy = np.real(np.vdot(psi, Hp))
    return energy

# We'll cache H per (n, coupling, field) to avoid rebuilding
_H_cache = None
_H_cache_key = None

def _get_H(n, coupling, field):
    global _H_cache, _H_cache_key
    key = (n, coupling, field)
    if _H_cache_key != key:
        _H_cache = _build_hamiltonian(n, coupling, field)
        _H_cache_key = key
    return _H_cache

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via full diagonalization."""
    H = _get_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _objective(params, n, coupling, field, layers):
    return _circuit_energy(params, n, coupling, field, layers)

def angles(n, coupling, field, layers):
    """Return optimal circuit parameters as (layers, n) array of angles in radians."""
    t_start = time.time()
    
    # Get Hamiltonian and exact ground energy
    H = _get_H(n, coupling, field)
    exact_e = _exact_ground_energy(n, coupling, field)
    
    dim = 1 << n
    n_params = layers * n
    
    # Multi-restart L-BFGS-B optimization
    best_params = None
    best_energy = np.inf
    
    # Initial guesses: several random starts + physics-motivated starts
    rng = np.random.RandomState(42)
    
    # Physics-motivated: for strong coupling, product of |0> or |1> states
    # For strong field, |+> state
    # Blend: try a few structured starts
    structured_starts = []
    
    # All zeros (|0>^n)
    structured_starts.append(np.zeros(n_params))
    # All pi/2 (|+>^n) - the seed
    structured_starts.append(np.full(n_params, np.pi / 2))
    # Layer-dependent: first layers rotate toward |+>, last layer fine-tune
    for frac in [0.3, 0.5, 0.7, 1.0]:
        p = np.zeros(n_params)
        for l in range(layers):
            p[l * n:(l + 1) * n] = frac * np.pi / 2
        structured_starts.append(p)
    # Alternating
    for l in range(layers):
        p = np.zeros(n_params)
        for q in range(n):
            p[l * n + q] = np.pi / 2 if q % 2 == 0 else -np.pi / 2
        structured_starts.append(p)
    
    # Random starts
    n_random = 8
    for _ in range(n_random):
        structured_starts.append(rng.uniform(-np.pi, np.pi, size=n_params))
    
    all_starts = structured_starts
    
    for start_idx, x0 in enumerate(all_starts):
        if time.time() - t_start > _DEADLINE - 10:
            break
        
        # Stage 1: coarse optimization
        res = minimize(
            _objective, x0,
            args=(n, coupling, field, layers),
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi)] * n_params,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success and res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x.copy()
        
        # Stage 2: fine polish from best of stage 1
        if time.time() - t_start > _DEADLINE - 5:
            break
        
        res2 = minimize(
            _objective, res.x,
            args=(n, coupling, field, layers),
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi)] * n_params,
            options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        
        if res2.success and res2.fun < best_energy:
            best_energy = res2.fun
            best_params = res2.x.copy()
    
    # Final ultra-tight polish
    if best_params is not None and time.time() - t_start < _DEADLINE - 2:
        res_final = minimize(
            _objective, best_params,
            args=(n, coupling, field, layers),
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi)] * n_params,
            options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-12}
        )
        if res_final.success and res_final.fun < best_energy:
            best_params = res_final.x
            best_energy = res_final.fun
    
    if best_params is None:
        # Fallback: seed
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    # Reshape to (layers, n)
    result = best_params.reshape(layers, n)
    return result
# EVOLVE-BLOCK-END
