"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _ry_matrix(theta):
    """Return 2x2 Ry rotation matrix."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def _ry_derivative_matrix(theta):
    """Return 2x2 derivative of Ry rotation matrix w.r.t theta."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return 0.5 * np.array([[-s, -c], [c, -s]], dtype=np.complex128)


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit of state (2^n vector). Qubit i is bit i (LSB)."""
    if abs(theta) < 1e-15:
        return state
    shape = [2] * n
    state_r = state.reshape(shape)
    gate = _ry_matrix(theta)
    state_moved = np.moveaxis(state_r, qubit, 0)
    state_out = np.tensordot(gate, state_moved, axes=([1], [0]))
    state_moved_back = np.moveaxis(state_out, 0, qubit)
    return state_moved_back.reshape(-1)

def _apply_ry_derivative(state, theta, qubit, n):
    """Apply dRy/dtheta to qubit of state (2^n vector)."""
    shape = [2] * n
    state_r = state.reshape(shape)
    gate = _ry_derivative_matrix(theta)
    state_moved = np.moveaxis(state_r, qubit, 0)
    state_out = np.tensordot(gate, state_moved, axes=([1], [0]))
    state_moved_back = np.moveaxis(state_out, 0, qubit)
    return state_moved_back.reshape(-1)


def _apply_cz(state, q1, q2, n):
    """Apply CZ gate between qubits q1 and q2. Qubit i is bit i (LSB)."""
    if q1 == q2:
        return state
    shape = [2] * n
    state_r = state.reshape(shape)
    state_moved = np.moveaxis(state_r, [q1, q2], [-2, -1])
    factor = np.ones(state_moved.shape, dtype=np.complex128)
    idx11 = (slice(None),) * (n - 2) + (1, 1)
    factor[idx11] = -1.0
    state_moved = state_moved * factor
    state_moved_back = np.moveaxis(state_moved, [-2, -1], [q1, q2])
    return state_moved_back.reshape(-1)


def _circuit_state(params, n, layers, derivative_index=-1):
    """
    Compute the state vector produced by the circuit.
    If derivative_index >= 0, the gate at that index (flattened) is replaced by its derivative.
    """
    thetas = params.reshape(layers, n)
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    idx = 0
    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            if idx == derivative_index:
                state = _apply_ry_derivative(state, theta, q, n)
            else:
                state = _apply_ry(state, theta, q, n)
            idx += 1
        
        # Apply CZ between adjacent pairs (ascending order), if not last layer
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
    
    return state


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for idx in range(dim):
        # Diagonal: Z_i Z_{i+1} terms
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag
        
        # Off-diagonal: X_i terms
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _energy_and_grad(params, n, coupling, field, layers, H, e0):
    """
    Compute energy and gradient of (E_circuit - E0)/n.
    params: flattened array of size layers*n
    """
    # 1. Compute main state
    psi = _circuit_state(params, n, layers)
    
    # 2. Compute H |psi>
    Hpsi = H @ psi
    
    # 3. Energy
    energy = np.vdot(psi, Hpsi).real
    obj_val = (energy - e0) / n
    
    # 4. Gradient
    # dE/dtheta_k = 2 Re < psi_k' | H | psi >
    # where psi_k' is the state where the k-th Ry is replaced by dRy/dtheta
    grad = np.zeros_like(params)
    num_params = params.size
    
    for k in range(num_params):
        # Compute derivative state
        psi_k = _circuit_state(params, n, layers, derivative_index=k)
        # Inner product <psi_k | Hpsi>
        # Note: psi_k is the state with the derivative gate.
        # The gradient of the expectation value <psi|H|psi> is 2 Re <dpsi/dtheta | H | psi>
        val = np.vdot(psi_k, Hpsi)
        grad[k] = 2.0 * val.real / n  # Divide by n for the objective function
        
    return obj_val, grad


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    # Get exact ground energy and Hamiltonian
    e0 = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    def objective_and_grad(params):
        return _energy_and_grad(params, n, coupling, field, layers, H, e0)
    
    def objective_only(params):
        val, _ = _energy_and_grad(params, n, coupling, field, layers, H, e0)
        return val
    
    # Heuristic initial guess
    # For small field, |0> is good (thetas ~ 0)
    # For large field, |+> is good (thetas ~ pi/2)
    # Interpolate
    x0 = np.zeros(layers * n)
    # Simple heuristic: start with pi/4 for all, or interpolate
    # Let's use a few restarts with L-BFGS-B
    
    best_result = None
    best_val = np.inf
    
    # Define restarts
    # 1. All zeros
    # 2. All pi/2
    # 3. All pi/4
    # 4. Random
    
    np.random.seed(42)
    restarts = [
        np.zeros(layers * n),
        np.full(layers * n, np.pi / 2),
        np.full(layers * n, np.pi / 4),
        np.random.randn(layers * n) * 0.5
    ]
    
    # Add more random restarts if n is small
    if n <= 4:
        for _ in range(4):
            restarts.append(np.random.randn(layers * n) * 0.8)
    else:
        for _ in range(2):
            restarts.append(np.random.randn(layers * n) * 0.8)

    for r, x_init in enumerate(restarts):
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
            
        x_init = np.clip(x_init, -np.pi, np.pi)
        
        try:
            # L-BFGS-B with analytical gradient
            res = opt.minimize(
                objective_only, 
                x_init, 
                jac=lambda x: _energy_and_grad(x, n, coupling, field, layers, H, e0)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
                
        except Exception as e:
            pass

    if best_result is None:
        best_result = x0
    
    theta = best_result.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    if time.time() - _start_time > _TIME_LIMIT:
        # Return fallback: |+> state
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
