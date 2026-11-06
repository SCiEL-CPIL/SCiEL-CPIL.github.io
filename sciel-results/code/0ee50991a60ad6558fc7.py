"""State-fitting approach: directly fit circuit output to exact ground state via least squares."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 138.0  # safety margin before 150s kill


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


def _exact_ground_state(H):
    """Compute exact ground state vector and energy."""
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argmin(eigvals)
    return np.real(eigvals[idx]), eigvecs[:, idx]


def _apply_ry_inplace(psi, q, theta, n):
    """Apply Ry(theta) on qubit q to state psi in-place (returns new array)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    shape = [2] * n
    psi_r = psi.reshape(shape)
    psi_moved = np.moveaxis(psi_r, q, 0)
    new0 = c * psi_moved[0] - s * psi_moved[1]
    new1 = s * psi_moved[0] + c * psi_moved[1]
    psi_out = np.stack([new0, new1], axis=0)
    return np.moveaxis(psi_out, 0, q).reshape(-1)


def _apply_cz_inplace(psi, q, n):
    """Apply CZ on qubits q, q+1 to state psi. Returns new array."""
    shape = [2] * n
    psi_r = psi.reshape(shape)
    psi_moved = np.moveaxis(psi_r, [q, q + 1], [-2, -1])
    psi_moved = psi_moved.copy()
    psi_moved[..., 1, 1] *= -1.0
    return np.moveaxis(psi_moved, [-2, -1], [q, q + 1]).reshape(-1)


def _circuit_state(params, n, layers):
    """Compute the final state from flattened (layers*n) parameter vector."""
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    t = params.reshape(layers, n)
    for layer in range(layers):
        for q in range(n):
            psi = _apply_ry_inplace(psi, q, t[layer, q], n)
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz_inplace(psi, q, n)
    return psi


def _circuit_state_and_jacobian(params, n, layers):
    """Compute state and Jacobian (dim x num_params) of the circuit output."""
    dim = 2 ** n
    num_params = layers * n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    t = params.reshape(layers, n)
    
    # Store intermediate states for Jacobian computation
    # For each Ry gate, store the state before it and the angle
    states_before_ry = []
    angles_list = []
    
    for layer in range(layers):
        for q in range(n):
            states_before_ry.append(psi.copy())
            angles_list.append(t[layer, q])
            psi = _apply_ry_inplace(psi, q, t[layer, q], n)
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz_inplace(psi, q, n)
    
    # Compute Jacobian using backpropagation
    # J[:, k] = <final_psi_conj | d/dtheta_k | psi>  -- actually we need d(psi_final)/d(theta_k)
    # Using adjoint method:
    # Let |psi_final> = U |psi_0>
    # d|psi_final>/dtheta_k = U_after_k * (dU_k/dtheta_k) * U_before_k * |psi_0>
    # where U_before_k applies all gates before k, U_after_k applies all gates after k
    
    # More efficient: propagate adjoint backward
    # grad_state[k] = d|psi_final>/dtheta_k
    # = (product of gates after k) * (dRy_k/dtheta) * (product of gates before k) * |psi_0>
    
    # We'll compute this by forward-propagating dpsi for each parameter
    # But that's O(num_params * circuit_cost). For small n this is fine.
    
    J = np.zeros((dim, num_params), dtype=np.complex128)
    
    # For each Ry parameter, compute the derivative of the final state
    param_idx = 0
    for layer in range(layers):
        for q in range(n):
            # Forward propagate to get state before this Ry
            psi_temp = np.zeros(dim, dtype=np.complex128)
            psi_temp[0] = 1.0
            t_local = params.reshape(layers, n)
            
            # Apply all gates before this one
            for l2 in range(layers):
                for q2 in range(n):
                    if l2 < layer or (l2 == layer and q2 < q):
                        psi_temp = _apply_ry_inplace(psi_temp, q2, t_local[l2, q2], n)
                    elif l2 < layer:
                        pass
                if l2 < layers - 1 and l2 < layer:
                    for q2 in range(n - 1):
                        psi_temp = _apply_cz_inplace(psi_temp, q2, n)
                elif l2 == layer - 1 and l2 < layers - 1:
                    for q2 in range(n - 1):
                        psi_temp = _apply_cz_inplace(psi_temp, q2, n)
            
            # This is getting complex. Let's use a simpler finite-difference-free approach:
            # Just compute the state for each parameter perturbation
            pass
            param_idx += 1
    
    # Simpler approach: use the stored states and compute derivatives analytically
    # d|psi>/dtheta for Ry(theta) on qubit q:
    # dRy/dtheta = 0.5 * [[-sin(theta/2), -cos(theta/2)], [cos(theta/2), -sin(theta/2)]]
    
    # We need to propagate dpsi through the remaining gates.
    # Let's do it properly with a forward sweep storing all needed states.
    
    # Actually, for small systems (n<=6, dim<=64), let's just use central differences
    # for the Jacobian. It's expensive but correct and simple.
    
    eps = 1e-7
    J = np.zeros((dim, num_params), dtype=np.complex128)
    for k in range(num_params):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[k] += eps
        params_minus[k] -= eps
        psi_plus = _circuit_state(params_plus, n, layers)
        psi_minus = _circuit_state(params_minus, n, layers)
        J[:, k] = (psi_plus - psi_minus) / (2 * eps)
    
    return psi, J


def _energy_and_gradient(params, n, layers, H, e0):
    """Compute energy and gradient of (E_circuit - e0)/n."""
    dim = 2 ** n
    psi = _circuit_state(params, n, layers)
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    obj_val = (energy - e0) / n
    
    # Gradient: dE/dtheta_k = 2 * Re[<H psi | dpsi/dtheta_k>]
    # = 2 * Re[<psi | H | dpsi/dtheta_k>]  (H is Hermitian)
    # But we can compute it as: grad_k = 2 * Re[<Hpsi | dpsi/dtheta_k>]
    
    # Using finite differences for gradient too (simpler, still fast for small n)
    num_params = len(params)
    eps = 1e-7
    grad = np.zeros(num_params)
    for k in range(num_params):
        params_plus = params.copy()
        params_plus[k] += eps
        psi_plus = _circuit_state(params_plus, n, layers)
        Hpsi_plus = H @ psi_plus
        energy_plus = np.vdot(psi_plus, Hpsi_plus).real
        grad[k] = (energy_plus - energy) / eps
    
    return obj_val, grad


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    e0, psi_gs = _exact_ground_state(H)
    
    num_params = layers * n
    
    def objective(params):
        val, _ = _energy_and_gradient(params, n, layers, H, e0)
        return val
    
    def gradient(params):
        _, grad = _energy_and_gradient(params, n, layers, H, e0)
        return grad
    
    # Physics-based initial guesses
    ratio = field / coupling if coupling > 0 else 1.0
    # For transverse field Ising, the ground state has <X> ~ tanh(field/coupling)
    # Ry(theta) on |0> gives cos(theta/2)|0> + sin(theta/2)|1>
    # To get <X> = sin(theta), we want theta = arcsin(tanh(ratio))
    base_angle = np.arcsin(np.tanh(ratio)) if abs(np.tanh(ratio)) <= 1.0 else np.pi / 2
    
    np.random.seed(42)
    
    restarts = []
    # Symmetric initial guesses
    restarts.append(np.full(num_params, base_angle))
    restarts.append(np.zeros(num_params))
    restarts.append(np.full(num_params, np.pi / 2))
    restarts.append(np.full(num_params, np.pi / 4))
    restarts.append(np.full(num_params, -np.pi / 4))
    
    # Random restarts
    for scale in [0.3, 0.5, 0.8, 1.0]:
        restarts.append(np.random.randn(num_params) * scale)
    
    # Alternating patterns
    for offset in [0, np.pi/2]:
        pat = np.zeros(num_params)
        for l in range(layers):
            for q in range(n):
                pat[l * n + q] = base_angle + offset * ((l + q) % 2)
        restarts.append(pat)
    
    best_result = None
    best_val = np.inf
    
    for r, x_init in enumerate(restarts):
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 15:
            break
        
        x_init = np.clip(x_init, -np.pi, np.pi)
        
        try:
            res = opt.minimize(
                objective,
                x_init,
                jac=gradient,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 50,
                    'maxfun': 500
                }
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
                
        except Exception:
            pass
    
    if best_result is None:
        best_result = np.full(num_params, base_angle)
    
    theta = best_result.reshape(layers, n)
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
