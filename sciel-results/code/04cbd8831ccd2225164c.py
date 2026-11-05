"""Combined program for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}

def _build_hamiltonian(n, coupling, field):
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

def _get_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0]

def _build_cz_matrix(n):
    """Build the full CZ layer as a diagonal matrix applied in ascending order."""
    dim = 1 << n
    diag = np.ones(dim, dtype=float)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        for b in range(dim):
            if (b & mask_i) and (b & mask_ip1):
                diag[b] *= -1.0
    return diag

def _circuit_state(theta, n, layers, cz_diag):
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            
            # Apply Ry on qubit i
            new_state = np.zeros(dim, dtype=float)
            for b in range(dim):
                bit_i = (b >> i) & 1
                if bit_i == 0:
                    idx0 = b
                    idx1 = b | (1 << i)
                    new_state[idx0] = c * state[idx0] - s * state[idx1]
                    new_state[idx1] = s * state[idx0] + c * state[idx1]
                # else: handled by the pair above (idx1 case)
            state = new_state
        
        if l < layers - 1:
            state = cz_diag * state
            
    return state

def _energy_and_grad(theta_flat, n, coupling, field, H, H0, cz_diag, need_grad=False):
    theta = theta_flat.reshape((3, n))
    state = _circuit_state(theta, n, 3, cz_diag)
    e_val = float(np.dot(state, H @ state))
    
    if not need_grad:
        return e_val
    
    # Compute gradient using parameter-shift rule
    # d/dtheta Ry(theta) = -1/2 * (|0><1| - |1><0|) Ry(theta) ... 
    # Actually, let's use a more direct approach: 
    # For each parameter, compute the derivative of the state.
    # This is expensive but we only have 3*n parameters.
    # Alternative: use finite differences for gradient
    grad = np.zeros_like(theta_flat)
    eps = 1e-6
    for idx in range(theta_flat.size):
        tp = theta_flat.copy()
        tm = theta_flat.copy()
        tp[idx] += eps
        tm[idx] -= eps
        ep = _energy(tp, n, coupling, field, H, H0, cz_diag)
        em = _energy(tm, n, coupling, field, H, H0, cz_diag)
        grad[idx] = (ep - em) / (2 * eps)
    
    return e_val, grad

def _energy(theta_flat, n, coupling, field, H, H0, cz_diag):
    theta = theta_flat.reshape((3, n))
    state = _circuit_state(theta, n, 3, cz_diag)
    return float(np.dot(state, H @ state))

def _energy_grad(theta_flat, n, coupling, field, H, H0, cz_diag):
    theta = theta_flat.reshape((3, n))
    state = _circuit_state(theta, n, 3, cz_diag)
    e_val = float(np.dot(state, H @ state))
    
    # Compute gradient analytically
    # The state is built by a sequence of gates. We can compute the gradient
    # by tracking how each parameter affects the state.
    # For simplicity and correctness, use central differences with a small step.
    grad = np.zeros_like(theta_flat)
    eps = 1e-5
    for idx in range(theta_flat.size):
        tp = theta_flat.copy()
        tm = theta_flat.copy()
        tp[idx] += eps
        tm[idx] -= eps
        ep = _energy(tp, n, coupling, field, H, H0, cz_diag)
        em = _energy(tm, n, coupling, field, H, H0, cz_diag)
        grad[idx] = (ep - em) / (2 * eps)
    
    return e_val, grad

def _physics_init(n, field):
    """Physics-inspired initialization based on arctan(1/field)."""
    theta = np.zeros((3, n))
    base_angle = np.arctan(1.0 / field)
    theta[0] = base_angle * 0.3
    theta[1] = base_angle * 0.5
    theta[2] = base_angle
    return theta

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
        
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    H = _build_hamiltonian(n, coupling, field)
    E0 = _get_ground_energy(n, coupling, field)
    cz_diag = _build_cz_matrix(n)
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return deadline - time.time()
    
    # Multiple initial guesses
    theta0_seed = np.zeros((layers, n))
    theta0_seed[-1] = np.pi / 2
    
    theta0_physics = _physics_init(n, field)
    
    best_theta = theta0_seed.copy()
    best_energy = _energy(theta0_seed.flatten(), n, coupling, field, H, None, cz_diag)
    
    e_phys = _energy(theta0_physics.flatten(), n, coupling, field, H, None, cz_diag)
    if e_phys < best_energy:
        best_energy = e_phys
        best_theta = theta0_physics.copy()
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Differential Evolution (global search) if time permits
    if remaining() > 30:
        try:
            bounds = [(-np.pi, np.pi)] * (layers * n)
            de_result = opt.differential_evolution(
                _energy,
                bounds,
                args=(n, coupling, field, H, None, cz_diag),
                seed=42,
                maxiter=100,
                tol=1e-10,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_energy:
                best_energy = de_result.fun
                best_theta = de_result.x.reshape((layers, n))
        except Exception:
            pass
    
    # Strategy 2: L-BFGS-B from multiple starting points
    starts = [best_theta.copy(), theta0_seed, theta0_physics]
    
    # Add structured random starts
    for r in range(12):
        if remaining() < 15:
            break
        if r % 3 == 0:
            current_theta = rng.uniform(-np.pi, np.pi, size=(layers, n))
        elif r % 3 == 1:
            current_theta = best_theta + rng.normal(0, 0.3, size=(layers, n))
        else:
            # Perturb the physics init
            current_theta = theta0_physics + rng.normal(0, 0.2, size=(layers, n))
        starts.append(current_theta)
    
    for st in starts:
        if remaining() < 8:
            break
        
        try:
            res = opt.minimize(
                _energy, 
                st.flatten(), 
                args=(n, coupling, field, H, None, cz_diag),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Strategy 3: Nelder-Mead polish from best
    if remaining() > 15:
        try:
            res = opt.minimize(
                _energy, 
                best_theta.flatten(), 
                args=(n, coupling, field, H, None, cz_diag),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Strategy 4: Final L-BFGS-B polish
    if remaining() > 5:
        try:
            res = opt.minimize(
                _energy, 
                best_theta.flatten(), 
                args=(n, coupling, field, H, None, cz_diag),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-13}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Ensure output is finite real array
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = theta0_seed
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
