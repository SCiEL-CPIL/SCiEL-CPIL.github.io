"""Optimized quantum circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 148.0
_START_TIME = time.time()

def _check_deadline(margin=3.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)

def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i using Kronecker products."""
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

def _get_ground_state(H):
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs[:, 0]

def _circuit_state(params, n, layers):
    """Build circuit state from parameters using vectorized operations."""
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # Vectorized Ry application
            # Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
            # We apply this to qubit q.
            # Reshape state to tensor [2]*n
            # Qubit q is at index q in the tensor (little-endian bit order? 
            # The task says "qubit i is bit i". Standard quantum convention usually 
            # has qubit 0 as LSB. Let's verify _build_hamiltonian logic.
            # In _build_hamiltonian, we used Kronecker products. 
            # np.kron(A, B) where A is first qubit. 
            # If we build term for q=0 as kron(X, I, ...), that acts on the FIRST 
            # subsystem in the kron chain.
            # In numpy, index [0,1,2] corresponds to kron(A,B,C) where A is slowest varying?
            # No, np.kron(A, B) has A's rows varying slowest. 
            # So index [i, j] corresponds to A[i] x B[j].
            # This means qubit 0 is the MOST SIGNIFICANT bit in the standard matrix index 
            # if we treat the vector index as a multi-index.
            # However, the task says "qubit i is bit i". 
            # Let's check the bit extraction in the original code:
            # b1 = (state >> i) & 1. This is LSB first.
            # My Kronecker construction: 
            # term = 1
            # for q in range(n):
            #   if q==0: term = kron(X, term) ? No, the code above does:
            #   term = np.kron(term, Z) if q==i...
            #   Wait, the loop is `for q in range(n)`.
            #   If q=0, term = kron(term, Z). So Z is on the RIGHTMOST factor.
            #   In np.kron(A, B), the last factor B corresponds to the fastest varying index (LSB).
            #   So Z on the rightmost factor acts on bit 0 (LSB).
            #   This matches "qubit i is bit i" (LSB).
            
            # So in the state vector, index `idx` has bit `q` at `idx >> q & 1`.
            # This corresponds to the tensor index where the q-th dimension (from the right) 
            # is the bit value.
            # Numpy array reshape: state.reshape([2]*n).
            # Axis 0 is the slowest varying (MSB), Axis n-1 is fastest (LSB).
            # So qubit q corresponds to axis `n-1-q`.
            
            axis = n - 1 - q
            state_tensor = state.reshape([2] * n)
            
            # Apply Ry
            # New[0, ...] = c * Old[0, ...] - s * Old[1, ...]  (Wait, Ry matrix)
            # Ry = [[c, -s], [s, c]]
            # So |0> -> c|0> + s|1>
            # |1> -> -s|0> + c|1>
            # Let's check standard Ry: 
            # Ry(theta) = exp(-i theta Y/2) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
            # Acting on |0> = [1,0]: [c, s]^T. So c*|0> + s*|1>.
            # Acting on |1> = [0,1]: [-s, c]^T. So -s*|0> + c*|1>.
            
            # We need to update the tensor.
            # Let's extract the two slices for qubit q.
            # Slice 0: where bit q is 0.
            # Slice 1: where bit q is 1.
            
            # Using numpy indexing is tricky with dynamic axes.
            # Let's use a simpler approach for small n: loop over indices but vectorize the math?
            # Or just use the loop from the original but optimized?
            # For n=6, 64 states. Loop is fine.
            
            new_state = np.zeros_like(state)
            for idx in range(2 ** n):
                bit = (idx >> q) & 1
                if bit == 0:
                    # state[idx] contributes to new_state[idx] (c) and new_state[idx | (1<<q)] (s)
                    new_state[idx] += c * state[idx]
                    new_state[idx | (1 << q)] += s * state[idx]
                else:
                    # state[idx] contributes to new_state[idx] (c) and new_state[idx ^ (1<<q)] (-s)
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

def _objective(x, n, coupling, field, layers, H, ground_state):
    params = x.reshape(layers, n)
    state = _circuit_state(params, n, layers)
    # Energy gap is not the objective directly, but the task says 
    # "minimizes the mean energy gap per spin relative to an independent exact eigensolution"
    # Wait, the task says: "Objective: mean (circuit energy - exact ground energy)/n."
    # So we minimize the energy of the circuit state relative to the ground energy.
    # This is equivalent to maximizing the overlap if the state were the ground state,
    # but since it's an energy expectation value, it's a different metric.
    # However, minimizing <H> is the standard VQE objective.
    energy = np.vdot(state, H @ state).real
    ground_energy = np.min(np.linalg.eigvalsh(H).real)
    return (energy - ground_energy) / n

def _heuristic_init(n, field):
    """
    Heuristic initialization based on the structure of the Ising ground state.
    For small fields, the ground state is close to |0000> or |1111> depending on coupling sign.
    Here coupling=1 (ferromagnetic), so |0000> and |1111> are degenerate in Z-basis if field=0.
    With field, it tilts.
    For n=4,6, we can just try a few simple patterns.
    """
    # Pattern 1: All zeros
    p1 = np.zeros((3, n))
    
    # Pattern 2: Small angles to induce some X component
    p2 = np.zeros((3, n))
    p2[0, :] = np.arcsin(np.sqrt(field / (1 + field))) * 0.5
    p2[1, :] = np.arcsin(np.sqrt(field / (1 + field))) * 0.5
    p2[2, :] = np.arcsin(np.sqrt(field / (1 + field))) * 0.5
    
    # Pattern 3: Alternating
    p3 = np.zeros((3, n))
    for i in range(n):
        p3[0, i] = 0.1 if i % 2 == 0 else -0.1
        p3[1, i] = 0.1 if i % 2 == 0 else -0.1
        p3[2, i] = 0.1 if i % 2 == 0 else -0.1
        
    return [p1, p2, p3]

def angles(n, coupling, field, layers):
    global _START_TIME
    _START_TIME = time.time()
    
    H = _build_hamiltonian(n, coupling, field)
    ground_state = _get_ground_state(H)
    ground_energy = np.min(np.linalg.eigvalsh(H).real)
    
    best_params = None
    best_score = np.inf
    
    # Try heuristic initializations
    inits = _heuristic_init(n, field)
    
    # Also add a random init
    np.random.seed(42)
    inits.append(np.random.uniform(-np.pi/4, np.pi/4, size=(layers, n)))
    
    for init in inits:
        if not _check_deadline(margin=5.0):
            break
        
        try:
            # Use L-BFGS-B with numerical gradient (fast for small n)
            result = minimize(
                _objective,
                init.flatten(),
                args=(n, coupling, field, layers, H, ground_state),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            continue
    
    # If we still have time, do a second pass with tighter tolerances from the best found
    if _check_deadline(margin=5.0) and best_params is not None:
        try:
            result = minimize(
                _objective,
                best_params.flatten(),
                args=(n, coupling, field, layers, H, ground_state),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
            
    # If best_params is still None, return zeros
    if best_params is None:
        best_params = np.zeros((layers, n))
        
    best_params = np.nan_to_num(best_params, nan=0.0, posinf=0.0, neginf=0.0)
    return best_params
# EVOLVE-BLOCK-END
