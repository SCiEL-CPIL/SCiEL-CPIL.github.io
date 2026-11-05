"""Optimized quantum circuit parameters using global stochastic search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_DEADLINE = 148.0
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
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        term = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i or q == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term

    # X terms: -field * X_i
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
    """Build circuit state from parameters."""
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            new_state = np.zeros_like(state)
            for idx in range(2 ** n):
                bit = (idx >> q) & 1
                if bit == 0:
                    new_state[idx] += c * state[idx]
                    new_state[idx | (1 << q)] += s * state[idx]
                else:
                    new_state[idx] += c * state[idx]
                    new_state[idx ^ (1 << q)] += -s * state[idx]
            state = new_state
        
        if layer < layers - 1:
            # Apply CZ on adjacent pairs
            for q in range(n - 1):
                for idx in range(2 ** n):
                    if ((idx >> q) & 1) == 1 and ((idx >> (q + 1)) & 1) == 1:
                        state[idx] *= -1.0
    
    return state

def _objective(x, n, coupling, field, layers, H, ground_energy):
    params = x.reshape(layers, n)
    state = _circuit_state(params, n, layers)
    energy = np.vdot(state, H @ state).real
    return (energy - ground_energy) / n

def angles(n, coupling, field, layers):
    global _START_TIME
    _START_TIME = time.time()
    
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = np.min(np.linalg.eigvalsh(H).real)
    
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # Differential Evolution is a global optimizer
    # maxiter controls the number of generations
    # popsize controls the population size
    # We need to be careful with the time budget.
    # For n=6, dim=18. 
    # A reasonable popsize is 15-20. Maxiter 50-100.
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            args=(n, coupling, field, layers, H, ground_energy),
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            init='sobol',
            polish=True
        )
        
        best_params = result.x.reshape(layers, n)
        
    except Exception:
        # Fallback to a simple heuristic if DE fails or times out unexpectedly
        best_params = np.zeros((layers, n))
        
    best_params = np.nan_to_num(best_params, nan=0.0, posinf=0.0, neginf=0.0)
    return best_params
# EVOLVE-BLOCK-END
