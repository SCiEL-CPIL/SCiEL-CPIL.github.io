"""Adjoint-gradient optimization of 3-layer Ry/CZ circuit for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


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
    """Compute exact ground state energy."""
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _circuit_state(params, n, layers):
    """Compute the final state of the circuit from angle parameters."""
    t = params.reshape(layers, n)
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = t[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            shape = [2] * n
            psi_r = psi.reshape(shape)
            gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
            psi_moved = np.moveaxis(psi_r, q, 0)
            psi_out = np.tensordot(gate, psi_moved, axes=([1], [0]))
            psi = np.moveaxis(psi_out, 0, q).reshape(-1)
        if layer < layers - 1:
            for q in range(n - 1):
                shape = [2] * n
                psi_r = psi.reshape(shape)
                psi_moved = np.moveaxis(psi_r, [q, q + 1], [-2, -1])
                factor = np.ones(psi_moved.shape, dtype=np.complex128)
                idx11 = (slice(None),) * (n - 2) + (1, 1)
                factor[idx11] = -1.0
                psi_moved = psi_moved * factor
                psi = np.moveaxis(psi_moved, [-2, -1], [q, q + 1]).reshape(-1)
    
    return psi


def _energy_and_gradient(params, n, H, e0, layers):
    """
    Compute energy and exact gradient using the adjoint method.
    
    Circuit structure:
    |0> -> Ry_0 -> CZ_0 -> Ry_1 -> CZ_1 -> Ry_2 -> |psi>
    
    For each Ry gate, the gradient is computed via backward propagation of H|psi>.
    """
    t = params.reshape(layers, n)
    dim = 2 ** n
    num_params = layers * n
    
    # Forward pass: store states before each Ry gate
    ry_states_before = []  # state before each Ry gate
    ry_thetas = []  # theta for each Ry gate
    cz_positions = []  # list of (layer, q) for CZ gates in order
    
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    # Track gate sequence
    gate_sequence = []  # list of ('ry', layer, q, theta) or ('cz', layer, q)
    
    for layer in range(layers):
        for q in range(n):
            theta = t[layer, q]
            ry_states_before.append(psi.copy())
            ry_thetas.append(theta)
            gate_sequence.append(('ry', layer, q, theta))
            
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            shape = [2] * n
            psi_r = psi.reshape(shape)
            gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
            psi_moved = np.moveaxis(psi_r, q, 0)
            psi_out = np.tensordot(gate, psi_moved, axes=([1], [0]))
            psi = np.moveaxis(psi_out, 0, q).reshape(-1)
        
        if layer < layers - 1:
            for q in range(n - 1):
                gate_sequence.append(('cz', layer, q))
                shape = [2] * n
                psi_r = psi.reshape(shape)
                psi_moved = np.moveaxis(psi_r, [q, q + 1], [-2, -1])
                factor = np.ones(psi_moved.shape, dtype=np.complex128)
                idx11 = (slice(None),) * (n - 2) + (1, 1)
                factor[idx11] = -1.0
                psi_moved = psi_moved * factor
                psi = np.moveaxis(psi_moved, [-2, -1], [q, q + 1]).reshape(-1)
    
    psi_final = psi.copy()
    energy = np.vdot(psi_final, H @ psi_final).real
    obj_val = (energy - e0) / n
    
    # Backward pass
    chi = H @ psi_final  # adjoint state at the end
    
    grad = np.zeros(num_params)
    ry_idx = num_params - 1  # index of last Ry gate in parameter array
    
    # Process gates in reverse
    for g in range(len(gate_sequence) - 1, -1, -1):
        gate = gate_sequence[g]
        if gate[0] == 'ry':
            layer, q, theta = gate[1], gate[2], gate[3]
            psi_before = ry_states_before[ry_idx]
            
            # Compute X_q |psi_before>
            shape = [2] * n
            psi_r = psi_before.reshape(shape)
            x_psi = np.moveaxis(psi_r, q, 0).copy()
            x_new = np.zeros_like(x_psi)
            x_new[0] = x_psi[1]
            x_new[1] = x_psi[0]
            x_flat = np.moveaxis(x_new, 0, q).reshape(-1)
            
            # Gradient: 2 Re(<chi | (-i/2) X_q | psi_before>) / n
            inner = np.vdot(chi, (-0.5j) * x_flat)
            grad[ry_idx] = 2.0 * inner.real / n
            
            # Propagate chi backwards through Ry
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            shape = [2] * n
            chi_r = chi.reshape(shape)
            gate_dag = np.array([[c, s], [-s, c]], dtype=np.complex128)
            chi_moved = np.moveaxis(chi_r, q, 0)
            chi_out = np.tensordot(gate_dag, chi_moved, axes=([1], [0]))
            chi = np.moveaxis(chi_out, 0, q).reshape(-1)
            
            ry_idx -= 1
        else:  # cz
            layer, q = gate[1], gate[2]
            # CZ is self-adjoint
            shape = [2] * n
            chi_r = chi.reshape(shape)
            chi_moved = np.moveaxis(chi_r, [q, q + 1], [-2, -1])
            factor = np.ones(chi_moved.shape, dtype=np.complex128)
            idx11 = (slice(None),) * (n - 2) + (1, 1)
            factor[idx11] = -1.0
            chi_moved = chi_moved * factor
            chi = np.moveaxis(chi_moved, [-2, -1], [q, q + 1]).reshape(-1)
    
    return obj_val, grad


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters using adjoint-gradient L-BFGS-B."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    e0 = _exact_ground_energy(H)
    num_params = layers * n
    
    def objective(params):
        psi = _circuit_state(params, n, layers)
        energy = np.vdot(psi, H @ psi).real
        return (energy - e0) / n
    
    def objective_grad(params):
        return _energy_and_gradient(params, n, H, e0, layers)
    
    # Generate diverse initial guesses
    np.random.seed(42)
    ratio = field / coupling if coupling > 0 else 1.0
    
    restarts = []
    
    # Physics-based: angle scales with field/coupling ratio
    base_angle = 0.5 * np.pi * np.tanh(ratio)
    for scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.2, 1.5]:
        restarts.append(np.full(num_params, base_angle * scale))
    
    # All zeros
    restarts.append(np.zeros(num_params))
    
    # All pi/2
    restarts.append(np.full(num_params, np.pi / 2))
    
    # Random initializations
    for s in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        restarts.append(np.random.randn(num_params) * s)
    
    # Alternating patterns (checkerboard)
    for s in [0.2, 0.4, 0.6, 0.8]:
        for sign in [1, -1]:
            pattern = np.zeros(num_params)
            for i in range(num_params):
                l = i // n
                q = i % n
                pattern[i] = s * sign * (1.0 if q % 2 == 0 else -1.0)
            restarts.append(pattern)
    
    # Layer-varying patterns
    for s in [0.3, 0.5, 0.7, 1.0]:
        pattern = np.zeros(num_params)
        for l in range(layers):
            for q in range(n):
                pattern[l * n + q] = s * (l + 1) / layers
        restarts.append(pattern)
    
    # Uniform with layer offset
    for base in [0.3, 0.5, 0.8, 1.2]:
        for offset in [0.1, 0.3, 0.5]:
            pattern = np.zeros(num_params)
            for l in range(layers):
                for q in range(n):
                    pattern[l * n + q] = base + offset * l
            restarts.append(pattern)
    
    # Sort by objective value and keep top candidates
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
    
    # Phase 1: L-BFGS-B from top initial guesses
    for i, (v0, x0) in enumerate(scored_restarts[:12]):
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 30:
            break
        
        try:
            res = opt.minimize(
                objective_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * num_params,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50, 'maxfun': 200}
            )
            
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
                
        except Exception:
            pass
        
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 25:
            break
    
    # Phase 2: Refine with more iterations
    candidates = []
    if best_params is not None:
        candidates.append(best_params)
    candidates.extend([x for _, x in scored_restarts[:5]])
    
    for x0 in candidates[:4]:
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 15:
            break
        
        try:
            res = opt.minimize(
                objective_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * num_params,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50, 'maxfun': 400}
            )
            
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
                
        except Exception:
            pass
        
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 10:
            break
    
    # Phase 3: Nelder-Mead polish
    if best_params is not None:
        elapsed = time.time() - _start_time
        if elapsed < _TIME_LIMIT - 5:
            try:
                res = opt.minimize(
                    objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        best_params = scored_restarts[0][1]
    
    theta = best_params.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
