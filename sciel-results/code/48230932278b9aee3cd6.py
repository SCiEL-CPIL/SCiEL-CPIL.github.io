import numpy as np
from scipy.optimize import minimize
import time

_H_cache = {}
_E_cache = {}

def _build_hamiltonian(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    H = np.zeros((2**n, 2**n), dtype=complex)
    
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_all(ops)
    
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_all(ops)
    
    _H_cache[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _E_cache:
        return _E_cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0].real
    _E_cache[key] = e0
    return e0

def _apply_ry(state, n, qubit, theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    shape = [2] * n
    state = state.reshape(shape)
    state = np.moveaxis(state, qubit, 0)
    new_state = np.empty_like(state)
    new_state[0] = c * state[0] - s * state[1]
    new_state[1] = s * state[0] + c * state[1]
    new_state = np.moveaxis(new_state, 0, qubit)
    return new_state.reshape(-1)

def _apply_cz(state, n, qubit1, qubit2):
    shape = [2] * n
    state = state.reshape(shape)
    state = np.moveaxis(state, qubit1, 0)
    state = np.moveaxis(state, 1, 0)
    state[1, 1] = -state[1, 1]
    state = np.moveaxis(state, 0, qubit2)
    state = np.moveaxis(state, 0, qubit1)
    return state.reshape(-1)

def _apply_circuit(state, n, thetas, layers):
    for l in range(layers):
        for i in range(n):
            state = _apply_ry(state, n, i, thetas[l, i])
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, n, i, i + 1)
    return state

def _circuit_energy(n, coupling, field, thetas, layers=3):
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    state = _apply_circuit(state, n, thetas, layers)
    H = _build_hamiltonian(n, coupling, field)
    energy = np.vdot(state, H @ state).real
    return energy

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0
    
    e_exact = _exact_ground_energy(n, coupling, field)
    
    def objective(theta_flat):
        thetas = theta_flat.reshape(layers, n)
        return _circuit_energy(n, coupling, field, thetas, layers)
    
    best_theta = None
    best_energy = np.inf
    
    np.random.seed(42)
    
    inits = []
    
    # 1. |+> state (pi/2 on last layer)
    t = np.zeros((layers, n))
    t[-1] = np.pi / 2
    inits.append(t.flatten())
    
    # 2. All pi/2
    t = np.full((layers, n), np.pi / 2)
    inits.append(t.flatten())
    
    # 3. Small random
    inits.append(np.random.uniform(-0.1, 0.1, (layers, n)).flatten())
    
    # 4-13. Random
    for _ in range(10):
        inits.append(np.random.uniform(-np.pi, np.pi, (layers, n)).flatten())
    
    for theta0 in inits:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                theta0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_theta = result.x.copy()
        except Exception:
            pass
    
    if best_theta is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_theta.reshape(layers, n)
