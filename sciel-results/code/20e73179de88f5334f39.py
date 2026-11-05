"""Optimized quantum circuit parameters using vectorized tensor contraction and local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    
    for i in range(n - 1):
        term = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i or q == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term

    for i in range(n):
        term = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term

    return H

def _circuit_state_vectorized(params, n, layers):
    """
    Vectorized circuit state computation.
    params: shape (layers, n)
    Returns state vector of size 2^n.
    """
    # Precompute bit masks for qubits
    masks = np.array([1 << q for q in range(n)])
    
    # Initial state: |0>^n is index 0. 
    # We represent the state as a complex array of length 2^n.
    # Only index 0 is 1.0 initially.
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    
    # We can vectorize the Ry rotations across all basis states.
    # However, for n=6, 2^6=64. It's small enough that even simple loops are fast,
    # but let's try to be efficient.
    
    # Ry(theta) on qubit q:
    # |0> -> cos(theta/2)|0> + sin(theta/2)|1>
    # |1> -> sin(theta/2)|0> - cos(theta/2)|1>
    
    for layer in range(layers):
        # Apply Ry to all qubits.
        # We can do this in-place or via temporary array.
        # For each qubit, we need to update amplitudes.
        # Since n is small, we can iterate qubits.
        
        # To vectorize across basis states:
        # For a given qubit q, we group indices by the bit at position q.
        # bit_q = (indices >> q) & 1
        
        # Let's just use a simple loop over qubits, but vectorize the index operations.
        # Actually, for n=6, 64 states. Looping 6 times is negligible.
        
        # Precompute cos/sin for this layer
        c = np.cos(params[layer] / 2)
        s = np.sin(params[layer] / 2)
        
        for q in range(n):
            mask = 1 << q
            # Indices where bit q is 0
            # We can generate all indices and mask them.
            # But it's easier to just update the array.
            
            # We need to swap/add amplitudes between idx and idx^mask.
            # Let's create a new state array to avoid dependency issues within the same qubit update?
            # No, Ry on a single qubit only mixes |0> and |1> for that qubit.
            # So for a fixed q, idx and idx^mask are distinct.
            # We can do it in place if we are careful, or use a temp.
            
            # Let's use a temp array for safety and clarity, or just update.
            # Since we are changing state[idx] and state[idx^mask] based on old values,
            # we should use a copy or careful ordering.
            # For n=6, copying is cheap.
            
            new_state = state.copy()
            
            # For all idx where bit q is 0:
            # new_state[idx] = c * state[idx] + s * state[idx^mask]
            # new_state[idx^mask] = s * state[idx] - c * state[idx^mask]
            
            # Vectorized approach:
            # Create array of all indices
            all_idx = np.arange(2 ** n)
            bit_q = (all_idx >> q) & 1
            
            # Indices where bit is 0
            idx_0 = all_idx[bit_q == 0]
            idx_1 = idx_0 | mask
            
            # Get old amplitudes
            amp_0 = state[idx_0]
            amp_1 = state[idx_1]
            
            # Compute new amplitudes
            new_amp_0 = c[q] * amp_0 + s[q] * amp_1
            new_amp_1 = s[q] * amp_0 - c[q] * amp_1
            
            # Update new_state
            new_state[idx_0] = new_amp_0
            new_state[idx_1] = new_amp_1
            
            state = new_state
        
        if layer < layers - 1:
            # Apply CZ on adjacent pairs in ascending order: (0,1), (1,2), ..., (n-2, n-1)
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                mask_both = mask1 | mask2
                
                # Find indices where both bits are 1
                # (idx & mask1) != 0 and (idx & mask2) != 0
                all_idx = np.arange(2 ** n)
                cond = ((all_idx & mask1) != 0) & ((all_idx & mask2) != 0)
                idx_cz = all_idx[cond]
                
                state[idx_cz] *= -1.0

    return state

def _energy_and_grad(x, n, coupling, field, layers, H, ground_energy):
    """
    Compute energy and gradient using finite differences or analytic?
    For n=6, 18 params. Finite differences with 18 calls is okay.
    But analytic gradient via parameter shift rule is exact and fast.
    Parameter shift rule:
    dE/dtheta_j = 0.5 * (E(theta_j + pi/2) - E(theta_j - pi/2))
    This requires 2 * 18 = 36 state evaluations.
    State evaluation is O(2^n * n * layers). For n=6, 64 * 6 * 3 = 1152 ops. Very fast.
    """
    params = x.reshape(layers, n)
    
    # Current energy
    state = _circuit_state_vectorized(params, n, layers)
    energy = np.vdot(state, H @ state).real
    obj_val = (energy - ground_energy) / n
    
    # Gradient via parameter shift
    grad = np.zeros_like(x)
    for i in range(layers * n):
        # Perturb + pi/2
        x_plus = x.copy()
        x_plus[i] += np.pi / 2
        p_plus = x_plus.reshape(layers, n)
        s_plus = _circuit_state_vectorized(p_plus, n, layers)
        e_plus = np.vdot(s_plus, H @ s_plus).real
        
        # Perturb - pi/2
        x_minus = x.copy()
        x_minus[i] -= np.pi / 2
        p_minus = x_minus.reshape(layers, n)
        s_minus = _circuit_state_vectorized(p_minus, n, layers)
        e_minus = np.vdot(s_minus, H @ s_minus).real
        
        grad[i] = 0.5 * (e_plus - e_minus)
        grad[i] /= n  # Objective is (E - E0)/n, so gradient scales by 1/n
    
    return obj_val, grad

def angles(n, coupling, field, layers):
    global _START_TIME
    _START_TIME = time.time()
    
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = np.min(np.linalg.eigvalsh(H).real)
    
    # Heuristic initialization
    # For small field, ground state is close to |0000>
    # For large field, ground state is close to |+>^n
    # |+> corresponds to Ry(pi/2)
    # Let's use a mix.
    
    # Start with a few random restarts and one heuristic.
    best_params = None
    best_score = np.inf
    
    n_params = layers * n
    bounds = [(-np.pi, np.pi)] * n_params
    
    # Heuristic 1: All zeros
    x0_list = []
    x0_list.append(np.zeros(n_params))
    
    # Heuristic 2: All pi/2 (maximally mixed-ish, good for transverse field)
    x0_list.append(np.full(n_params, np.pi / 2))
    
    # Heuristic 3: Based on field strength
    # If field is strong, Ry should be large.
    # Simple linear interp?
    # Let's just add some random starts.
    rng = np.random.default_rng(42)
    for _ in range(10):
        x0_list.append(rng.uniform(-np.pi, np.pi, n_params))
    
    for x0 in x0_list:
        if not _check_deadline(margin=5.0):
            break
            
        try:
            # Use L-BFGS-B with analytic gradient
            res = minimize(
                _energy_and_grad,
                x0,
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                jac=True, # We return (f, g)
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.zeros(n_params)
        
    best_params = best_params.reshape(layers, n)
    best_params = np.nan_to_num(best_params, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_params
# EVOLVE-BLOCK-END
