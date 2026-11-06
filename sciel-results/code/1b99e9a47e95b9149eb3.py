"""
Optimized shallow-circuit Ising ground state approximation.
Strategy: Global optimization using differential evolution with a time guard.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
import sys

# Cache for exact ground energies and Hamiltonians
_HAM_CACHE = {}
_EE_CACHE = {}

def _get_hamiltonian(n, coupling):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    key = (n, coupling)
    if key in _HAM_CACHE:
        return _HAM_CACHE[key]
    
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # X terms: -field * sum X_i (field is handled in energy calc, so we build J part here? 
    # No, better to build full H for each field? 
    # H = -J sum Z_i Z_{i+1} - h sum X_i
    # Since h varies, we can build H_J and H_X separately or just rebuild.
    # Rebuilding for n=6 (64x64) is fast.
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H[state, state] -= coupling
            else:
                H[state, state] += coupling
                
    # X_i terms
    for i in range(n):
        for state in range(dim):
            # X_i flips bit i
            flipped = state ^ (1 << i)
            H[state, flipped] -= 1.0 # Coefficient 1.0, field is multiplied later? 
            # Actually, let's just include field in the build or keep it separate.
            # Let's keep H_J and H_X separate for clarity and speed if needed.
            
    # Let's restart H construction to be cleaner: H = -J * H_IZ - h * H_IX
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # H_IZ = sum Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H[state, state] -= 1.0
            else:
                H[state, state] += 1.0
    H_IZ = H.copy()
    
    # H_IX = sum X_i
    H_IX = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H_IX[state, flipped] = 1.0
            
    _HAM_CACHE[key] = (H_IZ, H_IX)
    return _HAM_CACHE[key]

def _compute_exact_ground_energy(n, coupling, field):
    """Compute the exact ground state energy."""
    key = (n, coupling, field)
    if key in _EE_CACHE:
        return _EE_CACHE[key]
    
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    # Exact diagonalization
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0].real
    _EE_CACHE[key] = e0
    return e0

def _apply_circuit(theta, n):
    """
    Apply the circuit to |0...0> and return the state vector.
    theta: array of shape (layers, n)
    """
    dim = 1 << n
    # Start with |0...0>
    # In numpy, index 0 corresponds to |00...0> if we use standard binary mapping where LSB is qubit 0.
    # The problem states "qubit i is bit i".
    # Standard convention: |b_{n-1} ... b_1 b_0>.
    # |0...0> is index 0.
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    layers = theta.shape[0]
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
        # Acting on |0> -> cos(theta/2)|0> + sin(theta/2)|1>
        # Acting on |1> -> -sin(theta/2)|0> + cos(theta/2)|1>
        
        # Efficient application:
        # We can iterate over qubits and update the state vector.
        for i in range(n):
            c = np.cos(theta[l, i] / 2)
            s = np.sin(theta[l, i] / 2)
            
            # Create masks for bit i being 0 or 1
            # This is slow for n=6 if done naively with loops over states.
            # Better: use reshape or vectorized operations.
            
            # Reshape state to tensor [2]*n
            # Qubit i is the i-th dimension.
            # In numpy, if state is flat, index = sum(b_j * 2^j).
            # So dimension 0 is bit 0, dimension 1 is bit 1, etc.
            # state_tensor = state.reshape([2]*n)
            
            # Apply 2x2 matrix to axis i
            # M = [[c, -s], [s, c]]
            # new_state[b_0...b_i...b_{n-1}] = c*state[b_0...0...b_{n-1}] - s*state[b_0...1...b_{n-1}]
            # This requires careful indexing.
            
            # Simpler: Loop over all states? n=6 -> 64 states. 64*6*3 = 1152 ops. Very fast.
            # Let's just loop for simplicity and reliability, given small n.
            
            new_state = np.zeros_like(state)
            for idx in range(dim):
                bit_val = (idx >> i) & 1
                if bit_val == 0:
                    # state[idx] contributes to new state[idx] with c
                    # state[idx ^ (1<<i)] (bit i is 1) contributes to new state[idx] with -s
                    # Wait, Ry acts on the basis |0> and |1>.
                    # |0> -> c|0> + s|1>
                    # |1> -> -s|0> + c|1>
                    
                    # Let |psi> = sum c_k |k>.
                    # Component at k (where bit i is 0): c_0 |0> + c_1 |1>
                    # After Ry:
                    # coeff of |k> (bit i 0) in new state = c * c_0 - s * c_1
                    # coeff of |k ^ (1<<i)> (bit i 1) in new state = s * c_0 + c * c_1
                    
                    # So:
                    # new_state[k] (bit i 0) = c * state[k] - s * state[k ^ (1<<i)]
                    # new_state[k ^ (1<<i)] (bit i 1) = s * state[k] + c * state[k ^ (1<<i)]
                    
                    partner = idx ^ (1 << i)
                    new_state[idx] = c * state[idx] - s * state[partner]
                    new_state[partner] = s * state[idx] + c * state[partner]
                # If bit_val == 1, we've already handled it in the partner step?
                # No, the loop iterates over all idx.
                # If we process idx with bit 0, we update idx and partner.
                # If we process idx with bit 1, we update idx and partner (which is bit 0).
                # This double counts or overwrites incorrectly if not careful.
                
            # Let's do it properly by iterating only over indices where bit i is 0.
            new_state = np.zeros_like(state)
            for idx in range(dim):
                if ((idx >> i) & 1) == 0:
                    partner = idx ^ (1 << i)
                    new_state[idx] = c * state[idx] - s * state[partner]
                    new_state[partner] = s * state[idx] + c * state[partner]
            state = new_state
            
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                # CZ: |11> -> -|11>
                # Only affects states where bit i and bit i+1 are both 1.
                # Index has bits i and i+1 set.
                # Mask: (1 << i) | (1 << (i+1))
                mask = (1 << i) | (1 << (i + 1))
                for idx in range(dim):
                    if (idx & mask) == mask:
                        state[idx] *= -1.0
                        
    return state

def _compute_energy(theta, n, coupling, field):
    """Compute the energy of the state produced by the circuit."""
    state = _apply_circuit(theta, n)
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    # <psi| H |psi>
    Hpsi = H.dot(state)
    energy = np.vdot(state, Hpsi).real
    return energy

def _objective(params, n, coupling, field, e_exact):
    """Objective function for the optimizer."""
    theta = params.reshape(3, n)
    e_circ = _compute_energy(theta, n, coupling, field)
    return e_circ - e_exact

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Compute exact ground energy
    e_exact = _compute_exact_ground_energy(n, coupling, field)
    
    # 2. Define bounds
    # Angles are in radians. Ry(theta) is periodic with 2pi.
    # We can restrict to [-pi, pi] or [0, 2pi].
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # 3. Initial guess
    # Heuristic:
    # If field is large, |+> is good -> theta ~ pi/2
    # If coupling is large, |0> is good -> theta ~ 0
    # Let's try a simple interpolation or just start with the seed program's guess.
    # Seed: 0,0,pi/2
    x0 = np.zeros(layers * n)
    x0[-n:] = np.pi / 2
    
    # 4. Optimization
    # Differential Evolution is robust for small global problems.
    # We need to ensure it finishes in time.
    
    best_x = x0
    best_val = _objective(x0, n, coupling, field, e_exact)
    
    def callback(xk, convergence):
        nonlocal best_x, best_val
        # Check time
        if time.time() > deadline:
            return True # Stop
        # Update best if better
        val = _objective(xk, n, coupling, field, e_exact)
        if val < best_val:
            best_val = val
            best_x = xk.copy()
        return False

    try:
        # Use differential evolution
        # maxiter and popsize control the effort.
        # For 18 params, popsize=10, maxiter=50 is 500 evals.
        # Each eval is ~ms. 500ms is fine.
        res = differential_evolution(
            _objective, 
            bounds, 
            args=(n, coupling, field, e_exact),
            maxiter=100, 
            popsize=10, 
            tol=1e-6, 
            seed=42,
            callback=callback,
            polish=True, # Use L-BFGS-B for final polish
            init='sobol'
        )
        
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception:
        # Fallback to seed if anything goes wrong
        pass
    
    # Final check: if time is running out, return best found.
    # Convert back to (layers, n) array
    theta_opt = best_x.reshape(layers, n)
    
    return theta_opt

# EVOLVE-BLOCK-END
