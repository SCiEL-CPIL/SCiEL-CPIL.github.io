import numpy as np
import scipy.optimize
import time

def _ry_matrix(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])

def _cz_matrix():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]])

def _simulate_circuit(theta, n, layers):
    # theta is (layers, n)
    # Start state |0...0>
    # Basis index: qubit i is bit i. So |0> is index 0.
    # Standard Kronecker product order: qubit 0 is the leftmost (most significant) or rightmost?
    # "qubit i is bit i". Usually in numpy kron, the first argument is the leftmost.
    # Let's assume standard computational basis where state |b_{n-1} ... b_0> has index sum b_i 2^i.
    # If we use kron(A, B), A acts on the higher index bit?
    # Let's stick to a consistent convention.
    # If we represent the state as a vector of length 2^n.
    # Operator on qubit i.
    # Let's use the convention: tensor product order is qubit 0, 1, ..., n-1.
    # So state is |q0> |q1> ... |q_{n-1}>.
    # Index = q0 * 2^{n-1} + q1 * 2^{n-2} + ... + q_{n-1} * 2^0.
    # This is the "big-endian" bit ordering if q0 is MSB.
    # The task says "qubit i is bit i". Usually bit 0 is LSB.
    # Let's assume bit i corresponds to the $2^i$ place in the integer index.
    # So qubit 0 is LSB, qubit n-1 is MSB.
    # This means the tensor product order for operators is reversed if we use kron(A0, A1...)?
    # Let's construct the full operator matrix for each gate to be safe, or apply gates sequentially.
    
    # Sequential application is safer.
    # State is a vector.
    # To apply a 1-qubit gate on qubit i:
    # We need to reshape the state into a tensor of shape (2, 2, ..., 2).
    # If bit i is the $2^i$ bit, then in the tensor, it is the axis corresponding to that bit.
    # If we flatten in C-order (row-major), the last axis is the fastest varying index (LSB).
    # So axis 0 is MSB (qubit n-1), axis n-1 is LSB (qubit 0).
    # So qubit i corresponds to axis `n - 1 - i`.
    
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0 # |0...0>

    # Helper to apply a gate on a specific qubit
    def apply_gate(state, gate_matrix, qubit_idx):
        # qubit_idx is 0..n-1. Bit i.
        # Axis in tensor: n - 1 - qubit_idx
        axis = n - 1 - qubit_idx
        # Reshape state
        shape = [2] * n
        state_tensor = state.reshape(shape)
        # Move the axis to the end for easy matmul? Or use np.tensordot
        # Let's move axis to 0
        state_tensor = np.moveaxis(state_tensor, axis, 0)
        # Apply gate
        # gate_matrix is 2x2.
        # state_tensor shape is (2, 2, ..., 2) with the target qubit at axis 0.
        new_state_tensor = np.tensordot(gate_matrix, state_tensor, axes=([1], [0]))
        # tensordot reduces the axis 0 of state_tensor and axis 1 of gate_matrix.
        # Result shape: (2, 2, ..., 2) but the order of remaining axes is preserved.
        # Wait, tensordot(A, B, axes=([1], [0])):
        # A is (2,2). B is (2, 2, ..., 2).
        # Sum over A's axis 1 and B's axis 0.
        # Result shape: A's axis 0 (size 2) + B's remaining axes (2, ..., 2).
        # So the new state tensor has the new qubit state at axis 0.
        # We need to move it back to the original axis position.
        new_state_tensor = np.moveaxis(new_state_tensor, 0, axis)
        return new_state_tensor.reshape(-1)

    def apply_cz(state, qubit_i, qubit_j):
        # CZ is diagonal. We can just multiply the amplitudes where both qubits are 1.
        # Or use the matrix.
        # Let's use the matrix method for consistency, or direct multiplication.
        # Direct multiplication is faster.
        # Bit i and Bit j are 1.
        # Index mask: bits i and j are set.
        # We can iterate or use numpy masks.
        # For small n, iterating over 2^n states is fine.
        # But let's use the tensor method.
        axis_i = n - 1 - qubit_i
        axis_j = n - 1 - qubit_j
        
        state_tensor = state.reshape([2] * n)
        
        # We want to multiply by -1 if state_tensor[..., 1, ...] at axis_i and axis_j is 1.
        # This is equivalent to applying the CZ matrix.
        # Let's just use the matrix method to be safe and generic.
        
        # Move axes to front
        # This gets complicated with two axes.
        # Let's stick to the 4x4 matrix on the pair.
        
        # Extract the pair.
        # This is tricky with arbitrary axes.
        
        # Alternative: Construct the full CZ operator for the whole system? No, too big.
        
        # Let's use the explicit index logic for CZ.
        # CZ(i, j) flips the sign if bit i == 1 AND bit j == 1.
        # Create a mask of indices where bit i and bit j are 1.
        # Indices are 0 to 2^n - 1.
        
        # Bit i is 1 if (idx >> i) & 1 == 1
        # Bit j is 1 if (idx >> j) & 1 == 1
        
        # For efficiency, we can compute the mask once.
        # But doing this for every CZ call might be slow if done in Python loops.
        # Vectorized numpy:
        indices = np.arange(2**n)
        mask_i = (indices >> qubit_i) & 1
        mask_j = (indices >> qubit_j) & 1
        mask = (mask_i == 1) & (mask_j == 1)
        state[mask] *= -1
        return state

    for l in range(layers):
        # Apply Ry(theta[l, i]) to qubit i
        for i in range(n):
            theta_val = theta[l, i]
            if abs(theta_val) > 1e-10:
                gate = _ry_matrix(theta_val)
                state = apply_gate(state, gate, i)
        
        # Apply CZ between adjacent pairs (i, i+1) for i in 0..n-2
        # "between layers apply CZ on every adjacent pair in ascending order"
        # So after layer l, if l < layers-1, apply CZs.
        if l < layers - 1:
            for i in range(n - 1):
                state = apply_cz(state, i, i + 1)

    return state

def _energy(state, n, coupling, field):
    # H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    # <Z_i Z_{i+1}> = sum_k |psi_k|^2 * z_i(k) * z_{i+1}(k)
    # where z_i(k) is +1 if bit i is 0, -1 if bit i is 1.
    # <X_i> = sum_k |psi_k|^2 * x_i(k)
    # x_i(k) is 1 if bit i flips? No, X_i |k> = |k'> where k' is k with bit i flipped.
    # <psi| X_i |psi> = sum_k psi_k^* psi_{k \oplus 2^i}
    
    # Probabilities
    probs = np.abs(state)**2
    
    # Z terms
    z_energy = 0.0
    for i in range(n - 1):
        # z_i = 1 if bit 0, -1 if bit 1
        # Let's compute the expectation of Z_i Z_{i+1}
        # z_val_i = 1 - 2 * bit_i
        # z_val_{i+1} = 1 - 2 * bit_{i+1}
        # Product = (1 - 2b_i)(1 - 2b_{i+1}) = 1 - 2b_i - 2b_{i+1} + 4b_i b_{i+1}
        
        indices = np.arange(2**n)
        b_i = (indices >> i) & 1
        b_ip1 = (indices >> (i+1)) & 1
        
        z_prod = (1 - 2*b_i) * (1 - 2*b_ip1)
        
        z_energy += np.sum(probs * z_prod)
        
    z_part = -coupling * z_energy
    
    # X terms
    x_energy = 0.0
    for i in range(n):
        # <X_i> = sum_k psi_k^* psi_{k \oplus 2^i}
        # Note: X_i is Hermitian.
        # Let's compute the vector X_i |psi>
        # Or just sum the products.
        # k' = k ^ (1 << i)
        indices = np.arange(2**n)
        flipped = indices ^ (1 << i)
        
        # sum psi_k^* psi_{k'}
        # This is a real number for a single X term?
        # <psi| X |psi> is real.
        x_val = np.sum(np.conj(state) * state[flipped])
        # Ensure real
        x_val = np.real(x_val)
        x_energy += x_val
        
    x_part = -field * x_energy
    
    return z_part + x_part

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial guess: all zeros? Or random?
    # Random is better for DE.
    
    # Bounds for angles: [-pi, pi]
    bounds = [(-np.pi, np.pi)] * (n * layers)
    
    def objective(theta_flat):
        if time.time() > deadline:
            return np.inf
        theta = theta_flat.reshape(layers, n)
        try:
            state = _simulate_circuit(theta, n, layers)
            e = _energy(state, n, coupling, field)
            # Normalize by n? The task says "mean energy gap per spin".
            # Objective to minimize: (E_circuit - E_ground) / n
            # We don't know E_ground here directly without computing it.
            # But for optimization, minimizing E_circuit is equivalent to minimizing the gap
            # if E_ground is constant for a given (n, coupling, field).
            # So we can just minimize E_circuit.
            return e
        except Exception:
            return np.inf

    # Use Differential Evolution for global search
    # Maxiter 100 is usually enough for 18 params.
    # Tolerance 1e-8
    
    try:
        res = scipy.optimize.differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-8,
            polish=False, # We will polish manually
            workers=-1 # Use all cores
        )
        
        best_theta_flat = res.x
        best_val = res.fun
        
        # Polish with L-BFGS-B
        if time.time() < deadline:
            res2 = scipy.optimize.minimize(
                objective, 
                best_theta_flat, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res2.fun < best_val:
                best_theta_flat = res2.x
                best_val = res2.fun
                
        theta_opt = best_theta_flat.reshape(layers, n)
        
    except Exception:
        # Fallback to random or zero
        theta_opt = np.random.uniform(-np.pi, np.pi, (layers, n))
        
    return theta_opt
