"""Model-and-optimize: Hybrid state-fitting and energy-polish for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_start_time = time.time()
_TIME_LIMIT = 148.0  # Safety margin before 150s kill
_cache = {}


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bi = (idx >> i) & 1
            bp = (idx >> (i + 1)) & 1
            zi = 1.0 if bi == 0 else -1.0
            zp = 1.0 if bp == 0 else -1.0
            h_diag += -coupling * zi * zp
        H[idx, idx] = h_diag
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    return H


def _get_ground_state(n, coupling, field):
    """Compute the exact ground state vector and energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    idx_min = np.argmin(eigvals)
    e0 = float(eigvals[idx_min])
    psi_gs = eigvecs[:, idx_min]
    # Fix phase convention for stability
    max_idx = np.argmax(np.abs(psi_gs))
    phase = np.angle(psi_gs[max_idx])
    psi_gs = psi_gs * np.exp(-1j * phase)
    if psi_gs[max_idx].real < 0:
        psi_gs = -psi_gs
    return psi_gs, e0, H


def _circuit_forward(params, n, layers):
    """Forward pass returning the final state vector."""
    thetas = params.reshape(layers, n)
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            if abs(theta) > 1e-15:
                c = np.cos(theta / 2.0)
                s = np.sin(theta / 2.0)
                shape = [2] * n
                sr = state.reshape(shape)
                moved = np.moveaxis(sr, q, 0)
                out = np.empty_like(moved)
                out[0] = c * moved[0] - s * moved[1]
                out[1] = s * moved[0] + c * moved[1]
                state = np.moveaxis(out, 0, q).reshape(-1)
        if layer < layers - 1:
            for q in range(n - 1):
                shape = [2] * n
                sr = state.reshape(shape)
                moved = np.moveaxis(sr, [q, q + 1], [-2, -1])
                idx_11 = (slice(None),) * (n - 2) + (1, 1)
                moved[idx_11] *= -1.0
                state = np.moveaxis(moved, [-2, -1], [q, q + 1]).reshape(-1)
    return state


def _energy_and_grad(params, n, coupling, field, layers, H, e0):
    """Compute energy and gradient using adjoint method."""
    thetas = params.reshape(layers, n)
    dim = 1 << n
    states_before_ry = []
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            states_before_ry.append(state.copy())
            theta = thetas[layer, q]
            if abs(theta) > 1e-15:
                c = np.cos(theta / 2.0)
                s = np.sin(theta / 2.0)
                shape = [2] * n
                sr = state.reshape(shape)
                moved = np.moveaxis(sr, q, 0)
                out = np.empty_like(moved)
                out[0] = c * moved[0] - s * moved[1]
                out[1] = s * moved[0] + c * moved[1]
                state = np.moveaxis(out, 0, q).reshape(-1)
        if layer < layers - 1:
            for q in range(n - 1):
                shape = [2] * n
                sr = state.reshape(shape)
                moved = np.moveaxis(sr, [q, q + 1], [-2, -1])
                idx_11 = (slice(None),) * (n - 2) + (1, 1)
                moved[idx_11] *= -1.0
                state = np.moveaxis(moved, [-2, -1], [q, q + 1]).reshape(-1)
    
    psi_c = state
    Hpsi = H @ psi_c
    energy = np.vdot(psi_c, Hpsi).real
    obj_val = (energy - e0) / n
    
    # Backward pass for energy gradient
    grad = np.zeros((layers, n), dtype=np.float64)
    adjoint = Hpsi.copy()
    gate_idx = len(states_before_ry) - 1

    for layer in range(layers - 1, -1, -1):
        if layer < layers - 1:
            for q in range(n - 2, -1, -1):
                shape = [2] * n
                sr = adjoint.reshape(shape)
                moved = np.moveaxis(sr, [q, q + 1], [-2, -1])
                idx_11 = (slice(None),) * (n - 2) + (1, 1)
                moved[idx_11] *= -1.0
                adjoint = np.moveaxis(moved, [-2, -1], [q, q + 1]).reshape(-1)

        for q in range(n - 1, -1, -1):
            theta = thetas[layer, q]
            psi_b = states_before_ry[gate_idx]
            
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            v_moved = np.moveaxis(adjoint.reshape([2]*n), q, 0)
            psi_moved = np.moveaxis(psi_b.reshape([2]*n), q, 0)
            
            dp0 = 0.5 * (-s * psi_moved[0] - c * psi_moved[1])
            dp1 = 0.5 * (c * psi_moved[0] - s * psi_moved[1])
            
            inner = np.vdot(v_moved, np.stack([dp0, dp1], axis=0))
            grad[layer, q] = 2.0 * inner.real

            v_new = np.empty_like(v_moved)
            v_new[0] = c * v_moved[0] + s * v_moved[1]
            v_new[1] = -s * v_moved[0] + c * v_moved[1]
            adjoint = np.moveaxis(v_new, 0, q).reshape(-1)
            gate_idx -= 1

    grad_flat = (grad / n).flatten()
    return obj_val, grad_flat


def _physical_init(n, coupling, field, layers):
    """Generate physically-motivated initial parameters."""
    # The TFIM ground state transitions from ferromagnetic to paramagnetic
    # as field increases. For the circuit starting from |0>^n:
    # - Low field: small rotations (stay near |0>)
    # - High field: larger rotations toward |->
    
    h_ratio = field / coupling if coupling != 0 else field
    
    # Estimate target rotation angle based on field strength
    # At h=0, ground state is |0...0> (all Z=+1), so theta=0
    # At h>>J, ground state approaches |->^n, so theta=pi
    # Use a smooth interpolation
    target_theta = np.arctan(h_ratio) * 2.0  # maps [0,inf) to [0,pi]
    
    inits = []
    
    # Primary: uniform rotation scaled by field
    x = np.full(layers * n, target_theta / layers)
    inits.append(x)
    
    # Secondary: increasing rotation per layer
    x = np.zeros(layers * n)
    for l in range(layers):
        x[l*n:(l+1)*n] = target_theta * (l + 1) / layers
    inits.append(x)
    
    # Tertiary: decreasing rotation per layer
    x = np.zeros(layers * n)
    for l in range(layers):
        x[l*n:(l+1)*n] = target_theta * (layers - l) / layers
    inits.append(x)
    
    # Quartic: only last layer rotates
    x = np.zeros(layers * n)
    x[(layers-1)*n:layers*n] = target_theta
    inits.append(x)
    
    # Quintic: only first layer rotates
    x = np.zeros(layers * n)
    x[0:n] = target_theta
    inits.append(x)
    
    return inits


def _optimize_for_params(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]

    psi_gs, e0, H = _get_ground_state(n, coupling, field)
    np_params = layers * n

    def obj_only(x):
        return _energy_and_grad(x, n, coupling, field, layers, H, e0)[0]

    def jac_only(x):
        return _energy_and_grad(x, n, coupling, field, layers, H, e0)[1]

    # Generate diverse restarts combining physical and random inits
    restarts = []
    
    # Physical inits
    restarts.extend(_physical_init(n, coupling, field, layers))
    
    # Random inits with time-aware count
    elapsed = time.time() - _start_time
    remaining = _TIME_LIMIT - elapsed
    n_random = max(3, min(15, int(remaining / 3.0)))
    
    rng = np.random.default_rng(42 + hash((n, field)) % 1000)
    for _ in range(n_random):
        restarts.append(rng.standard_normal(np_params) * np.pi * 0.4)
    
    # A few structured inits
    for val in [0.0, np.pi/6, np.pi/3, -np.pi/3, np.pi/2]:
        restarts.append(np.full(np_params, val))

    best_val = np.inf
    best_x = np.zeros(np_params)

    # Sort restarts by initial objective to try good ones first
    def _eval_init(x0):
        x0 = np.clip(x0, -np.pi, np.pi)
        try:
            v = obj_only(x0)
            return v, x0
        except:
            return np.inf, x0
    
    evaluated = [_eval_init(r) for r in restarts]
    evaluated.sort(key=lambda t: t[0])
    
    # Run L-BFGS-B on each restart with time budget
    per_restart_time = max(1.0, remaining / len(evaluated))
    
    for val_init, x0 in evaluated:
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 8:
            break
        x0 = np.clip(x0, -np.pi, np.pi)
        try:
            t0 = time.time()
            res = opt.minimize(
                obj_only, x0, jac=jac_only,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final high-precision polish
    elapsed = time.time() - _start_time
    if elapsed < _TIME_LIMIT - 5 and best_val < np.inf:
        try:
            res2 = opt.minimize(
                obj_only, best_x, jac=jac_only,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 50}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x.copy()
        except Exception:
            pass

    theta = best_x.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    if time.time() - _start_time > _TIME_LIMIT:
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
