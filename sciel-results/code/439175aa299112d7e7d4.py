"""Quantum information: shallow-circuit Ising ground states.

Strategy:
1. Exploit reflection symmetry of the open chain: theta[i] == theta[n-1-i].
   For n=4: unique angles per layer = 2 (positions 0,1 mirror to 3,2).
   For n=6: unique angles per layer = 3 (positions 0,1,2 mirror to 5,4,3).
   Total unique parameters: 3 layers * ceil(n/2) = 6 for n=4, 9 for n=6.

2. Use a coarse-to-fine grid search on the reduced parameter space, followed
   by scipy.optimize.minimize (L-BFGS-B) for local polish.

3. Compute exact ground energy via exact diagonalization.

4. Deadline guard: track wall-clock, return best found if approaching limit.
"""

import time
import numpy as np
import scipy.optimize as opt

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    return H


def _exact_ground_energy(H):
    """Compute exact ground energy."""
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _apply_ry(psi, q, theta, n):
    """Apply Ry(theta) on qubit q to state psi (flattened)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    shape = [2] * n
    psi_r = psi.reshape(shape)
    gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
    psi_moved = np.moveaxis(psi_r, q, 0)
    psi_out = np.tensordot(gate, psi_moved, axes=([1], [0]))
    return np.moveaxis(psi_out, 0, q).reshape(-1)


def _apply_cz(psi, q, n):
    """Apply CZ between qubit q and q+1 to state psi (flattened)."""
    shape = [2] * n
    psi_r = psi.reshape(shape)
    psi_moved = np.moveaxis(psi_r, [q, q + 1], [-2, -1])
    factor = np.ones(psi_moved.shape, dtype=np.complex128)
    idx11 = (slice(None),) * (n - 2) + (1, 1)
    factor[idx11] = -1.0
    psi_moved = psi_moved * factor
    return np.moveaxis(psi_moved, [-2, -1], [q, q + 1]).reshape(-1)


def _expand_symmetric_params(sym_params, n, layers):
    """
    Expand symmetric parameters (layers x ceil(n/2)) to full parameters (layers x n).
    Uses reflection symmetry: theta[l, q] == theta[l, n-1-q].
    """
    full = np.zeros((layers, n))
    half = (n + 1) // 2
    for l in range(layers):
        for q in range(half):
            full[l, q] = sym_params[l, q]
            full[l, n - 1 - q] = sym_params[l, q]
    return full


def _circuit_state_full(params, n, layers):
    """Compute the final state of the circuit from full angle parameters."""
    t = params.reshape(layers, n)
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            psi = _apply_ry(psi, q, t[layer, q], n)
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz(psi, q, n)
    
    return psi


def _circuit_energy(params, n, H, layers):
    """Compute energy of the circuit state."""
    psi = _circuit_state_full(params, n, layers)
    return np.vdot(psi, H @ psi).real


def _energy_gap(sym_params, n, H, e0, layers):
    """Compute mean energy gap per spin for symmetric parameters."""
    full_params = _expand_symmetric_params(sym_params, n, layers)
    energy = _circuit_energy(full_params, n, H, layers)
    return (energy - e0) / n


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters using symmetric parameterization."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    e0 = _exact_ground_energy(H)
    
    half = (n + 1) // 2
    num_sym = layers * half
    bounds = [(-np.pi, np.pi)] * num_sym
    
    def objective(sym_params):
        return _energy_gap(sym_params, n, H, e0, layers)
    
    # Generate initial guesses in symmetric parameter space
    np.random.seed(42)
    ratio = field / coupling if coupling > 0 else 1.0
    
    restarts = []
    
    # Physics-based: interpolate between |0> and |+> based on field/coupling
    base_angle = 0.5 * np.pi * np.tanh(ratio)
    for scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.2]:
        restarts.append(np.full(num_sym, base_angle * scale))
    
    # All zeros
    restarts.append(np.zeros(num_sym))
    
    # All pi/2
    restarts.append(np.full(num_sym, np.pi / 2))
    
    # Random initializations
    for s in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
        restarts.append(np.random.randn(num_sym) * s)
    
    # Layer-varying patterns
    for s in [0.3, 0.5, 0.7]:
        pattern = np.zeros(num_sym)
        for l in range(layers):
            for q in range(half):
                pattern[l * half + q] = s * (l + 1) / layers
        restarts.append(pattern)
    
    # Alternating patterns in qubit space
    for s in [0.2, 0.4, 0.6]:
        for sign in [1, -1]:
            pattern = np.zeros(num_sym)
            for l in range(layers):
                for q in range(half):
                    pattern[l * half + q] = s * sign * (1.0 if q % 2 == 0 else -1.0)
            restarts.append(pattern)
    
    # Sort by objective value
    scored_restarts = []
    for x in restarts:
        x = np.clip(x, -np.pi, np.pi)
        try:
            v = objective(x)
            scored_restarts.append((v, x.copy()))
        except:
            scored_restarts.append((np.inf, x.copy()))
    scored_restarts.sort(key=lambda t: t[0])
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from best initial guesses
    for i, (v0, x0) in enumerate(scored_restarts[:15]):
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 30:
            break
        
        try:
            res = opt.minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30, 'maxfun': 150}
            )
            
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
                
        except Exception:
            pass
        
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 20:
            break
    
    # Phase 2: Nelder-Mead polish from best solution
    if best_params is not None:
        elapsed = time.time() - _start_time
        if elapsed < _TIME_LIMIT - 10:
            try:
                res = opt.minimize(
                    objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        best_params = scored_restarts[0][1]
    
    # Expand symmetric params to full params
    theta = _expand_symmetric_params(best_params, n, layers)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5:
        # Fallback: return |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        # Fallback
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
