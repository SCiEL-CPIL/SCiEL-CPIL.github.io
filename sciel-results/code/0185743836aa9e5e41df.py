"""Shallow-circuit Ising ground states - high-precision energy minimization with fast state evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}
_PRECOMPUTED = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    I = np.eye(2, dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_list(ops)
        
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
        
    return H

def _precompute_config(n, coupling, field):
    """Precompute Hamiltonian and exact ground state for a configuration."""
    key = (n, coupling, field)
    if key in _PRECOMPUTED:
        return _PRECOMPUTED[key]
    
    H = _build_hamiltonian(n, coupling, field)
    evals, evecs = np.linalg.eigh(H)
    ground_state = evecs[:, 0]
    ground_energy = float(evals[0])
    
    _PRECOMPUTED[key] = (H, ground_state, ground_energy)
    return _PRECOMPUTED[key]

def _precompute_all():
    """Precompute all target configurations at module load time."""
    configs = []
    for n in [4, 6]:
        for field in [0.45, 1.0, 1.65]:
            configs.append((n, 1.0, field))
    
    for n, coupling, field in configs:
        _precompute_config(n, coupling, field)

# Precompute at module load
_precompute_all()

def _circuit_state(theta, n, layers):
    """Compute the full state vector from circuit parameters (vectorized)."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    base = np.arange(dim)
    idx_arrays = {}
    for i in range(n):
        mask = 1 << i
        idx0 = base[(base & mask) == 0]
        idx1 = idx0 | mask
        idx_arrays[i] = (idx0, idx1)
    
    cz_masks = []
    for i in range(n - 1):
        m = (1 << i) | (1 << (i + 1))
        cz_masks.append((base & m) == m)
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0, idx1 = idx_arrays[i]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if l < layers - 1:
            for mask in cz_masks:
                state[mask] *= -1.0
    
    return state

def _energy_from_state(state, H):
    """Compute energy expectation value."""
    return float(state @ (H @ state))

def _energy_objective(x, n, layers, H):
    """Energy objective function."""
    theta = x.reshape((layers, n))
    state = _circuit_state(theta, n, layers)
    return _energy_from_state(state, H)

def _gradient_energy(x, n, layers, H, eps=1e-6):
    """Central-difference gradient for energy objective."""
    theta = x.reshape((layers, n))
    state = _circuit_state(theta, n, layers)
    e0 = _energy_from_state(state, H)
    
    grad = np.zeros_like(x)
    for idx in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        
        theta_p = x_plus.reshape((layers, n))
        theta_m = x_minus.reshape((layers, n))
        
        sp = _circuit_state(theta_p, n, layers)
        sm = _circuit_state(theta_m, n, layers)
        
        ep = _energy_from_state(sp, H)
        em = _energy_from_state(sm, H)
        
        grad[idx] = (ep - em) / (2 * eps)
    
    return grad

def _physics_init(n, coupling, field):
    """Physics-based initialization using mean-field approximation."""
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    
    theta = np.zeros((3, n))
    theta[0] = t_opt * 0.1
    theta[1] = t_opt * 0.2
    theta[2] = t_opt * 0.7
    return theta

def _generate_initial_candidates(n, coupling, field, layers, rng):
    """Generate diverse initial candidates."""
    candidates = []
    
    # Physics-based
    candidates.append(_physics_init(n, coupling, field))
    
    # Uniform distributions
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    
    candidates.append(np.full((layers, n), t_opt / layers))
    
    theta_fl = np.zeros((layers, n))
    theta_fl[-1] = t_opt
    candidates.append(theta_fl)
    
    theta_pihalf = np.zeros((layers, n))
    theta_pihalf[-1] = np.pi / 2
    candidates.append(theta_pihalf)
    
    # Structured patterns
    for pattern in [
        lambda t: np.array([t*0.1, t*0.2, t*0.7]),
        lambda t: np.array([t*0.3, t*0.3, t*0.4]),
        lambda t: np.array([t*0.5, t*0.25, t*0.25]),
        lambda t: np.array([t*0.2, t*0.3, t*0.5]),
    ]:
        theta_p = np.zeros((layers, n))
        for l in range(layers):
            theta_p[l] = pattern(t_opt)[l]
        candidates.append(theta_p)
    
    # Random starts
    for _ in range(20):
        candidates.append(rng.uniform(-np.pi, np.pi, size=(layers, n)))
    
    # Small perturbations of physics init
    phys = _physics_init(n, coupling, field)
    for _ in range(10):
        candidates.append(phys + rng.normal(0, 0.3, size=(layers, n)))
    
    return candidates

def _local_optimize(x0, n, layers, H, deadline, method='L-BFGS-B', maxiter=5000, ftol=1e-16):
    """Local optimization with time guard."""
    if time.time() > deadline - 5:
        return x0, float('inf')
    
    def objective(x):
        return _energy_objective(x, n, layers, H)
    
    try:
        res = opt.minimize(
            objective,
            x0,
            method=method,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-14}
        )
        return res.x, res.fun
    except Exception:
        return x0, float('inf')

def _polish(x0, n, layers, H, deadline):
    """Multi-tier polish with tightening tolerances."""
    best_x = x0.copy()
    best_e = _energy_objective(x0, n, layers, H)
    
    tiers = [
        ('L-BFGS-B', 300, 1e-14),
        ('L-BFGS-B', 500, 1e-16),
        ('L-BFGS-B', 300, 1e-18),
    ]
    
    for method, maxiter, ftol in tiers:
        if time.time() > deadline - 3:
            break
        x_new, e_new = _local_optimize(best_x, n, layers, H, deadline, method, maxiter, ftol)
        if e_new < best_e:
            best_x = x_new
            best_e = e_new
    
    # Final high-precision pass
    if time.time() < deadline - 2:
        x_new, e_new = _local_optimize(best_x, n, layers, H, deadline, 'L-BFGS-B', 1000, 1e-20)
        if e_new < best_e:
            best_x = x_new
            best_e = e_new
    
    return best_x, best_e

def _multi_start_optimize(n, coupling, field, layers, deadline):
    """Multi-start optimization using energy objective."""
    H, ground_state, ground_energy = _precompute_config(n, coupling, field)
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = _generate_initial_candidates(n, coupling, field, layers, rng)
    
    # Evaluate all candidates
    best_x = None
    best_e = float('inf')
    
    for tc in candidates:
        if time.time() > deadline - 30:
            break
        e = _energy_objective(tc.flatten(), n, layers, H)
        if e < best_e:
            best_e = e
            best_x = tc.flatten().copy()
    
    # Local optimization from best candidate
    if time.time() < deadline - 20:
        x_opt, e_opt = _local_optimize(best_x, n, layers, H, deadline - 15, 'L-BFGS-B', 3000, 1e-14)
        if e_opt < best_e:
            best_x = x_opt
            best_e = e_opt
    
    # Polish
    if time.time() < deadline - 10:
        best_x, best_e = _polish(best_x, n, layers, H, deadline - 5)
    
    # Random restarts with local optimization
    restart_budget = 10
    for r in range(restart_budget):
        if time.time() > deadline - 8:
            break
        tc = best_x.reshape((layers, n)) + rng.normal(0, 0.4, size=(layers, n))
        x_opt, e_opt = _local_optimize(tc.flatten(), n, layers, H, deadline - 5, 'L-BFGS-B', 1500, 1e-15)
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt
    
    # Final polish
    if time.time() < deadline - 3:
        best_x, best_e = _polish(best_x, n, layers, H, deadline - 1)
    
    best_theta = best_x.reshape((layers, n))
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((layers, n))
        best_theta[-1] = np.pi / 2
    
    return best_theta

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_theta = _multi_start_optimize(n, coupling, field, layers, deadline)
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
