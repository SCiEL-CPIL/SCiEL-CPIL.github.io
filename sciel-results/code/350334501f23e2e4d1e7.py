"""Optimize 3-layer Ry/CZ circuit for transverse-field Ising ground states."""
import numpy as np
from scipy.optimize import minimize
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for open transverse-field Ising chain using kron products."""
    dim = 1 << n
    I = np.eye(2, dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    
    H = np.zeros((dim, dim), dtype=float)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        term = np.array([[1.0]])
        for q in range(n):
            if q == i or q == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
    
    # X_i terms
    for i in range(n):
        term = np.array([[1.0]])
        for q in range(n):
            if q == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
    
    return H

def _apply_ry_inplace(state, qubit, theta):
    """Apply Ry(theta) to qubit in-place using vectorized operations."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    dim = len(state)
    mask = (np.arange(dim) >> qubit) & 1
    zeros = ~mask
    ones = mask
    idx0 = np.where(zeros)[0]
    idx1 = idx0 | (1 << qubit)
    old0 = state[idx0].copy()
    old1 = state[idx1].copy()
    state[idx0] = c * old0 - s * old1
    state[idx1] = s * old0 + c * old1

def _apply_cz_inplace(state, q1, q2):
    """Apply CZ gate in-place using vectorized operations."""
    dim = len(state)
    mask = ((np.arange(dim) >> q1) & 1) & ((np.arange(dim) >> q2) & 1)
    state[mask] *= -1.0

def _circuit_state(n, params, layers):
    """Apply the circuit and return the final state vector."""
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer * n + q]
            _apply_ry_inplace(state, q, theta)
        if layer < layers - 1:
            for q in range(n - 1):
                _apply_cz_inplace(state, q, q + 1)
    
    return state

def _energy(state, H):
    """Compute expectation value of H in state."""
    return np.real(np.vdot(state, H @ state))

def _physics_init(n, coupling, field, layers):
    """Generate physics-inspired initial angles based on mean-field approximation."""
    # For transverse-field Ising model, the ground state has a preferred angle
    # where tan(theta/2) ~ field/coupling gives a reasonable starting point
    base_angle = 2 * np.arctan(field / (coupling + 1e-10))
    params = np.zeros(layers * n)
    for layer in range(layers):
        for q in range(n):
            params[layer * n + q] = base_angle
    return params

def _objective_and_grad(params, n, coupling, field, layers, H, ground_energy):
    """Objective and gradient computation."""
    state = _circuit_state(n, params, layers)
    e_circuit = _energy(state, H)
    obj = (e_circuit - ground_energy) / n
    
    # Compute gradient via finite differences (more robust than analytical for complex states)
    grad = np.zeros_like(params)
    eps = 1e-7
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        state_plus = _circuit_state(n, params_plus, layers)
        e_plus = _energy(state_plus, H)
        grad[i] = (e_plus - e_circuit) / (eps * n)
    
    return obj, grad

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate ground state."""
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Build Hamiltonian and find ground state
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    ground_energy = eigenvalues[0]
    
    n_params = layers * n
    
    # Generate diverse initializations
    inits = []
    inits.append(_physics_init(n, coupling, field, layers))
    inits.append(np.zeros(n_params))
    inits.append(np.full(n_params, np.pi / 2))
    inits.append(np.full(n_params, np.pi / 4))
    inits.append(np.full(n_params, -np.pi / 4))
    inits.append(np.full(n_params, 2 * np.arctan(field / (coupling + 1e-10))))
    inits.append(np.random.uniform(-np.pi, np.pi, n_params))
    inits.append(np.random.uniform(-np.pi, np.pi, n_params))
    inits.append(np.random.uniform(-np.pi, np.pi, n_params))
    
    best_params = None
    best_obj = np.inf
    
    # Stage 1: L-BFGS-B with finite-difference gradients
    for init in inits:
        if time_left() < 15.0:
            break
        try:
            result = minimize(
                lambda p: _objective_and_grad(p, n, coupling, field, layers, H, ground_energy)[0],
                init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            obj_val = result.fun
            if obj_val < best_obj:
                best_obj = obj_val
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Stage 2: Powell refinement on best result
    if best_params is not None and time_left() > 10.0:
        try:
            result2 = minimize(
                lambda p: _objective_and_grad(p, n, coupling, field, layers, H, ground_energy)[0],
                best_params,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-15}
            )
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Stage 3: Additional random restarts if time permits
    if time_left() > 20.0:
        for _ in range(8):
            if time_left() < 10.0:
                break
            init = np.random.uniform(-np.pi, np.pi, n_params)
            try:
                result = minimize(
                    lambda p: _objective_and_grad(p, n, coupling, field, layers, H, ground_energy)[0],
                    init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                obj_val = result.fun
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_params = result.x.copy()
            except Exception:
                continue
    
    # Stage 4: Final Powell polish
    if best_params is not None and time_left() > 5.0:
        try:
            result3 = minimize(
                lambda p: _objective_and_grad(p, n, coupling, field, layers, H, ground_energy)[0],
                best_params,
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-13, 'ftol': 1e-16}
            )
            if result3.fun < best_obj:
                best_obj = result3.fun
                best_params = result3.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.zeros(n_params)
    
    # Reshape to (layers, n)
    theta = best_params.reshape(layers, n)
    return theta
