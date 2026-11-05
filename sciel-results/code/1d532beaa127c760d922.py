"""Optimized quantum circuit parameters using parameter-shift gradient descent with multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0
_START_TIME = time.time()

def _check_deadline(margin=3.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)

def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    total_dim = 2 ** n
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    H = np.zeros((total_dim, total_dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        term = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i or q == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term

    # X terms
    for i in range(n):
        term = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term

    return H

def _circuit_state(params, n, layers):
    """
    Compute circuit state vector.
    params: (layers, n) array of angles.
    Qubit i is bit i. Start |0>^n.
    Each layer: Ry(theta) on every qubit, then CZ on adjacent pairs (except after final layer).
    """
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # Qubit q is bit q. In tensor reshape [2]*n, axis 0 is MSB (qubit n-1), axis n-1 is LSB (qubit 0).
            # So qubit q corresponds to axis n-1-q.
            axis = n - 1 - q
            state_tensor = state.reshape([2] * n)
            
            # Move axis to front
            state_moved = np.moveaxis(state_tensor, axis, 0)
            
            # Ry(theta) = [[c, -s], [s, c]]
            # |0> -> c|0> + s|1>
            # |1> -> -s|0> + c|1>
            # new_0 = c * old_0 - s * old_1
            # new_1 = s * old_0 + c * old_1
            
            new_moved = np.zeros_like(state_moved)
            new_moved[0] = c * state_moved[0] - s * state_moved[1]
            new_moved[1] = s * state_moved[0] + c * state_moved[1]
            
            state = np.moveaxis(new_moved, 0, axis).reshape(-1)
        
        if layer < layers - 1:
            # Apply CZ on adjacent pairs
            # CZ flips sign if both bits are 1.
            for q in range(n - 1):
                # Mask for indices where bit q is 1
                mask_q = (np.arange(2 ** n) >> q) & 1
                mask_q1 = (np.arange(2 ** n) >> (q + 1)) & 1
                both_1 = (mask_q == 1) & (mask_q1 == 1)
                state[both_1] *= -1.0

    return state

def _energy(params, n, coupling, field, H, ground_energy):
    """Compute (circuit_energy - ground_energy) / n."""
    state = _circuit_state(params, n, 3)
    energy = np.vdot(state, H @ state).real
    return (energy - ground_energy) / n

def _energy_and_grad(params, n, coupling, field, H, ground_energy):
    """Compute energy and gradient using parameter-shift rule for Ry gates."""
    params = np.array(params, dtype=float)
    state = _circuit_state(params, n, 3)
    energy = np.vdot(state, H @ state).real
    base_energy = (energy - ground_energy) / n
    
    grad = np.zeros_like(params)
    
    for layer in range(3):
        for q in range(n):
            # Parameter shift rule: dE/dtheta = (E(theta + pi/2) - E(theta - pi/2)) / 2
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[layer, q] += np.pi / 2
            params_minus[layer, q] -= np.pi / 2
            
            e_plus = _energy(params_plus, n, coupling, field, H, ground_energy)
            e_minus = _energy(params_minus, n, coupling, field, H, ground_energy)
            
            grad[layer, q] = (e_plus - e_minus) / 2.0
    
    return base_energy, grad

def _heuristic_init(n, field):
    """
    Heuristic initialization based on product state approximation.
    For transverse field Ising, ground state has X-like character if field dominates, Z-like if coupling dominates.
    Here coupling=1, fields are 0.45, 1, 1.65.
    If field is large, we want states closer to |+>^n (theta = pi/2).
    If field is small, we want states closer to |0>^n or |1>^n (theta = 0 or pi).
    Let's try a few heuristic starting points.
    """
    # Try theta = pi/2 (maximally mixed X state)
    p1 = np.full((3, n), np.pi / 2)
    # Try theta = 0
    p2 = np.zeros((3, n))
    # Try theta = pi
    p3 = np.full((3, n), np.pi)
    # Try random
    p4 = np.random.uniform(-np.pi, np.pi, (3, n))
    # Try staggered
    p5 = np.full((3, n), np.pi / 4)
    p6 = np.full((3, n), -np.pi / 4)
    
    return [p1, p2, p3, p4, p5, p6]

def angles(n, coupling, field, layers):
    global _START_TIME
    _START_TIME = time.time()
    
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = np.min(np.linalg.eigvalsh(H).real)
    
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    best_params = None
    best_energy = np.inf
    
    # Generate initial points
    inits = _heuristic_init(n, field)
    # Add some random inits
    for _ in range(10):
        inits.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
    
    for init in inits:
        if not _check_deadline(margin=10.0):
            break
        
        # Flatten for scipy
        x0 = init.flatten()
        
        def fun(x):
            p = x.reshape(layers, n)
            return _energy(p, n, coupling, field, H, ground_energy)
        
        def fun_grad(x):
            p = x.reshape(layers, n)
            e, g = _energy_and_grad(p, n, coupling, field, H, ground_energy)
            return e, g.flatten()
        
        try:
            result = minimize(
                fun_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        best_params = np.zeros((layers, n))
    
    best_params = np.nan_to_num(best_params, nan=0.0, posinf=0.0, neginf=0.0)
    return best_params
# EVOLVE-BLOCK-END
