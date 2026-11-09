"""Optimize shallow circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def _apply_ry(state, n, i, theta):
    """Apply Ry(theta) to qubit i of the state vector (vectorized)."""
    dim = 1 << n
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    # Create indices for bit i being 0 and 1
    # Bit i corresponds to the (n-1-i)th bit in standard binary representation 
    # if we map qubit 0 to LSB? 
    # The task says: "qubit i is bit i". 
    # Usually in Qiskit/standard, qubit 0 is LSB (rightmost).
    # Let's check the seed code: 
    # in_bit = (basis >> (n - 1 - i)) & 1
    # This implies qubit 0 is the MSB (leftmost) in the integer representation of the basis state.
    # i.e., basis = b_{n-1} ... b_1 b_0 where b_i is the bit for qubit i.
    # So qubit i is at position (n-1-i) from the right.
    
    # We need to reshape the state to apply the gate.
    # It's easier to just iterate or use advanced indexing.
    # For n=6, dim=64. Iteration is fine, but vectorization is better.
    
    # Let's use the "split" method.
    # We want to group states by the value of bit i.
    # Indices where bit i is 0: (k >> (n-1-i)) & 1 == 0
    # Indices where bit i is 1: (k >> (n-1-i)) & 1 == 1
    
    # This is complex to vectorize directly without reshaping.
    # Reshape approach:
    # Move axis i to the front?
    # State is 1D. We can view it as tensor of shape (2,)*n.
    # Axis 0 corresponds to qubit n-1 (MSB), Axis n-1 corresponds to qubit 0 (LSB).
    # Wait, if basis integer is k, and qubit i is bit (n-1-i), then:
    # k = sum_{j=0}^{n-1} b_j 2^{n-1-j} ? No.
    # Standard integer: k = sum b_j 2^j.
    # Seed code: in_bit = (basis >> (n - 1 - i)) & 1.
    # If n=4, i=0. shift by 3. This is the MSB.
    # So qubit 0 is the MSB.
    # This means the state vector index $k$ corresponds to binary string $b_{n-1} \dots b_0$ 
    # where $b_i$ is the value of qubit $i$?
    # Let's trace: n=2.
    # i=0: shift by 1. bit 1 of k.
    # i=1: shift by 0. bit 0 of k.
    # So qubit 0 is bit 1 (MSB), qubit 1 is bit 0 (LSB).
    # This is "Big Endian" qubit ordering in the integer index.
    
    # To apply Ry to qubit i:
    # We can reshape state to (2, 2, ..., 2).
    # The axis corresponding to qubit i is (n-1-i).
    # Let's verify:
    # n=2. Shape (2,2).
    # Index [b1, b0].
    # Qubit 0 is b1. Axis 0.
    # Qubit 1 is b0. Axis 1.
    # So axis for qubit i is (n-1-i).
    
    state_tensor = state.reshape((2,) * n)
    axis = n - 1 - i
    
    # Move the target axis to the front for easy manipulation
    state_tensor = np.moveaxis(state_tensor, axis, 0)
    
    # Now shape is (2, 2, ..., 2) where the first dimension is the qubit i.
    # state_tensor[0, ...] is the part where qubit i is 0.
    # state_tensor[1, ...] is the part where qubit i is 1.
    
    new_state_tensor = np.empty_like(state_tensor)
    # Ry matrix: [[c, -s], [s, c]]
    # new_0 = c * old_0 - s * old_1
    # new_1 = s * old_0 + c * old_1
    
    new_state_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
    new_state_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
    
    # Move axis back
    new_state_tensor = np.moveaxis(new_state_tensor, 0, axis)
    
    return new_state_tensor.reshape(-1)

def _apply_cz(state, n, i):
    """Apply CZ between qubit i and i+1."""
    dim = 1 << n
    # Qubit i is at bit position (n-1-i).
    # Qubit i+1 is at bit position (n-1-(i+1)) = (n-2-i).
    
    # We need to flip the sign of amplitudes where both bits are 1.
    # We can create a mask.
    # For efficiency, we can iterate over the "other" qubits? 
    # Or just use a precomputed mask? 
    # Since n is small, a simple loop or vectorized mask is fine.
    
    # Vectorized mask creation:
    # indices = np.arange(dim)
    # bit_i = (indices >> (n-1-i)) & 1
    # bit_j = (indices >> (n-1-i-1)) & 1
    # mask = (bit_i == 1) & (bit_j == 1)
    # state[mask] *= -1
    
    # This is O(dim) per CZ. There are n-1 CZs per layer.
    # Total O(n * dim) per layer. For n=6, dim=64, this is negligible.
    
    indices = np.arange(dim)
    shift_i = n - 1 - i
    shift_j = n - 1 - (i + 1)
    
    bit_i = (indices >> shift_i) & 1
    bit_j = (indices >> shift_j) & 1
    
    mask = (bit_i == 1) & (bit_j == 1)
    
    state = state.copy() # Avoid in-place if we want to be safe, or just modify
    state[mask] *= -1
    
    return state

def _build_state(theta, n, layers):
    """Build the circuit state from angles."""
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0...0>
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            state = _apply_ry(state, n, i, theta[layer, i])
        
        # Apply CZ gates between adjacent pairs (if not last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, n, i)
                
    return state

def _compute_energy(state, n, coupling, field):
    """Compute <H> where H = -coupling*sum Z_i Z_{i+1} - field*sum X_i"""
    dim = 1 << n
    energy = 0.0
    probs = np.abs(state) ** 2
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        shift_i = n - 1 - i
        shift_j = n - 1 - (i + 1)
        # We can vectorize this too
        indices = np.arange(dim)
        z_i = np.where((indices >> shift_i) & 1, -1.0, 1.0)
        z_j = np.where((indices >> shift_j) & 1, -1.0, 1.0)
        zz_vals = -coupling * z_i * z_j
        energy += np.sum(zz_vals * probs)
        
    # X terms: off-diagonal
    for i in range(n):
        shift = n - 1 - i
        # Flip bit i
        indices = np.arange(dim)
        flipped_indices = indices ^ (1 << shift)
        
        # <X_i> = sum_k state[k^e_i] * conj(state[k])
        # Since X is Hermitian, <X> is real.
        # <X> = 2 * Re( sum_{k < k^e_i} state[k^e_i] conj(state[k]) )
        # Or simply sum over all k, but each pair counted twice?
        # <psi|X|psi> = sum_k <psi|k><X|k><k|psi> = sum_k psi_k^* <X|k> psi_k
        # <X|k> = |k^e_i>
        # So <psi|X|psi> = sum_k psi_k^* psi_{k^e_i}
        # This sum is real because it's an expectation value of a Hermitian operator.
        
        cross_terms = np.conj(state) * state[flipped_indices]
        energy += -field * np.sum(cross_terms).real
        
    return energy

def _objective(params, n, coupling, field, layers):
    """Objective function for optimization."""
    theta = params.reshape(layers, n)
    state = _build_state(theta, n, layers)
    energy = _compute_energy(state, n, coupling, field)
    return energy

def _physics_init(n, field, layers):
    """Physics-based initialization."""
    theta = np.zeros((layers, n))
    # For small field, want angles near 0 (ferromagnetic, Z aligned)
    # For large field, want angles near pi/2 (paramagnetic, X aligned)
    # Ry(pi/2)|0> = |+>
    # Ry(0)|0> = |0>
    # Interpolate
    frac = min(field / 2.0, 1.0)
    base_angle = frac * np.pi / 2
    theta[:, :] = base_angle
    return theta

def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate Ising ground state."""
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    # Physics-based initialization
    theta0 = _physics_init(n, field, layers)
    params0 = theta0.flatten()
    
    # Compute initial energy
    state0 = _build_state(theta0, n, layers)
    energy0 = _compute_energy(state0, n, coupling, field)
    
    best_params = params0.copy()
    best_energy = energy0
    
    # Optimization strategy: L-BFGS-B with restarts
    # L-BFGS-B is much better than Nelder-Mead for smooth functions with gradients.
    # We will use numerical gradients provided by SciPy, which is robust.
    
    try:
        # Main run
        result = scipy.optimize.minimize(
            _objective, 
            params0, 
            args=(n, coupling, field, layers),
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x.copy()
            
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return best_params.reshape(layers, n)
        
        # Restart strategy: 
        # Try a few different initializations to escape local minima.
        # 1. All zeros
        # 2. All pi/2
        # 3. Random perturbations around best
        
        inits = [
            np.zeros(layers * n),
            np.full(layers * n, np.pi / 2),
            best_params + np.random.normal(0, 0.1, size=layers * n),
            best_params + np.random.normal(0, 0.5, size=layers * n)
        ]
        
        for init in inits:
            if time.time() - start_time > deadline - 10:
                break
                
            result_r = scipy.optimize.minimize(
                _objective, 
                init, 
                args=(n, coupling, field, layers),
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result_r.fun < best_energy:
                best_energy = result_r.fun
                best_params = result_r.x.copy()
                
    except Exception as e:
        # Fallback to best found so far
        pass
        
    theta_best = best_params.reshape(layers, n)
    return theta_best
# EVOLVE-BLOCK-END
