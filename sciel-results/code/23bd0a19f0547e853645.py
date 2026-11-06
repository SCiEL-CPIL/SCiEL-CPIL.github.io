"""Rewritten program: Fast variational optimization for Ising ground states with symmetry exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, least_squares

_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill
_cache = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for idx in range(dim):
        # Diagonal: ZZ terms
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag
        
        # Off-diagonal: X terms
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    
    return H

def _exact_ground_state(n, coupling, field):
    """Compute the exact ground state vector and energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    e0 = np.real(eigenvalues[0])
    psi0 = eigenvectors[:, 0]
    # Ensure the phase is consistent (make the largest component real and positive)
    max_idx = np.argmax(np.abs(psi0))
    phase = np.angle(psi0[max_idx])
    psi0 = psi0 * np.exp(-1j * phase)
    if psi0[max_idx] < 0:
        psi0 = -psi0
    return psi0, e0

def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) gate to qubit."""
    if abs(theta) < 1e-15:
        return state
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    
    # Split state into two halves based on the qubit value
    # We need to reshape to apply the gate
    shape = [2] * n
    state_r = state.reshape(shape)
    state_moved = np.moveaxis(state_r, qubit, 0)
    
    # state_moved has shape (2, 2^(n-1), ...)
    # Ry = [[c, -s], [s, c]]
    new_state_moved = np.empty_like(state_moved)
    new_state_moved[0] = c * state_moved[0] - s * state_moved[1]
    new_state_moved[1] = s * state_moved[0] + c * state_moved[1]
    
    state_moved_back = np.moveaxis(new_state_moved, 0, qubit)
    return state_moved_back.reshape(-1)

def _apply_cz(state, q1, q2, n):
    """Apply CZ gate between qubits q1 and q2."""
    if q1 == q2:
        return state
    shape = [2] * n
    state_r = state.reshape(shape)
    state_moved = np.moveaxis(state_r, [q1, q2], [-2, -1])
    
    # Flip sign where both qubits are 1
    # The last two axes are q1, q2
    # Index [1, 1] in the last two dimensions
    idx = [slice(None)] * (n - 2) + [1, 1]
    state_moved[idx] *= -1.0
    
    state_moved_back = np.moveaxis(state_moved, [-2, -1], [q1, q2])
    return state_moved_back.reshape(-1)

def _circuit_state(params, n, layers):
    """Compute the state produced by the circuit."""
    thetas = params.reshape(layers, n)
    state = np.zeros(2 ** n, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry(state, thetas[layer, q], q, n)
        
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
    
    return state

def _objective_fidelity(params, n, layers, psi_target):
    """Objective: minimize 1 - |<psi_target|psi_circuit>|^2."""
    psi_circuit = _circuit_state(params, n, layers)
    overlap = np.vdot(psi_target, psi_circuit)
    fidelity = np.abs(overlap) ** 2
    return 1.0 - fidelity

def _optimize_symmetry(n, coupling, field, layers):
    """
    Optimize using symmetry.
    For open chain with uniform fields/couplings, the ground state is symmetric under reversal.
    The circuit is also symmetric if thetas are symmetric.
    We can restrict thetas[layer, q] = thetas[layer, n-1-q].
    This reduces parameters from layers*n to layers*(n//2 + n%2).
    """
    H = _build_hamiltonian(n, coupling, field)
    psi_target, e0 = _exact_ground_state(n, coupling, field)
    
    # Determine number of unique parameters due to symmetry
    # Qubits 0..n-1. Symmetry: q <-> n-1-q.
    # Unique qubits: 0, 1, ..., (n-1)//2
    n_unique = (n + 1) // 2
    total_params = layers * n_unique
    
    def _symmetry_objective(unique_params):
        # Expand to full params
        full_params = np.zeros((layers, n))
        for layer in range(layers):
            for q in range(n_unique):
                theta = unique_params[layer * n_unique + q]
                full_params[layer, q] = theta
                full_params[layer, n - 1 - q] = theta
        
        psi_circuit = _circuit_state(full_params, n, layers)
        overlap = np.vdot(psi_target, psi_circuit)
        fidelity = np.abs(overlap) ** 2
        return 1.0 - fidelity

    def _symmetry_gradient(unique_params):
        # Numerical gradient? Or analytic?
        # Let's stick to Nelder-Mead or L-BFGS-B without explicit gradient first for robustness.
        # Actually, L-BFGS-B is faster if we provide gradient.
        # Let's just use Nelder-Mead on the reduced space for simplicity and robustness.
        pass

    # Initial guesses
    inits = []
    
    # 1. Zero
    inits.append(np.zeros(total_params))
    
    # 2. Pi/2
    inits.append(np.full(total_params, np.pi / 2))
    
    # 3. Heuristic
    ratio = field / coupling
    base_theta = np.arctan(ratio)
    inits.append(np.full(total_params, base_theta))
    
    # 4. Random restarts
    rng = np.random.default_rng(42)
    for _ in range(30):
        inits.append(rng.uniform(-np.pi, np.pi, size=total_params))
    
    best_val = np.inf
    best_params = None
    
    nm_options = {
        'maxiter': 300,
        'fatol': 1e-10,
        'xatol': 1e-8,
        'adaptive': True
    }

    for x0 in inits:
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        
        try:
            res = minimize(
                _symmetry_objective,
                x0,
                method='Nelder-Mead',
                options=nm_options
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.zeros(total_params)
    
    # Expand best_params to full shape
    full_params = np.zeros((layers, n))
    for layer in range(layers):
        for q in range(n_unique):
            theta = best_params[layer * n_unique + q]
            full_params[layer, q] = theta
            full_params[layer, n - 1 - q] = theta
            
    return full_params

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    try:
        theta = _optimize_symmetry(n, coupling, field, layers)
        _cache[key] = theta
        return theta
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _cache[key] = theta
        return theta
# EVOLVE-BLOCK-END
