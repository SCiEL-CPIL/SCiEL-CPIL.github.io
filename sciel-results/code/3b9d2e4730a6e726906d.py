import numpy as np
import scipy.optimize as opt
import time
import sys

# Cache for exact ground energies
_ground_energy_cache = {}

def _get_ground_energy(n, coupling, field):
    """Compute exact ground state energy of the open transverse-field Ising chain."""
    key = (n, coupling, field)
    if key in _ground_energy_cache:
        return _ground_energy_cache[key]
    
    # Build Hamiltonian
    # H = -J * sum Z_i Z_{i+1} - h * sum X_i
    # Basis: |00...0> to |11...1>, qubit i is bit i (little endian in standard numpy kron? 
    # Usually kron(A, B) puts A as most significant. 
    # Let's stick to standard: state |b_{n-1} ... b_0>. 
    # The problem says "qubit i is bit i". 
    # In Python/NumPy, if we use kron(X, I), X acts on the first qubit (MSB).
    # If "qubit i is bit i", it usually implies bit 0 is the least significant bit.
    # Let's assume standard little-endian for the problem statement "qubit i is bit i" 
    # means qubit 0 is the rightmost in the ket notation |... b_1 b_0>.
    # However, the circuit applies Ry to every qubit and CZ to adjacent pairs.
    # The physical result is invariant to the labeling convention as long as it's consistent 
    # between the Hamiltonian construction and the circuit simulation.
    
    # Let's use a consistent convention: 
    # State vector index corresponds to integer value.
    # Qubit i corresponds to the i-th bit (2^i).
    # This is little-endian.
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Identity
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Helper to build operator on qubit i (little endian)
    def build_op_single(op, qubit_idx, n_qubits):
        # op is 2x2 matrix
        # We want op to act on qubit qubit_idx
        # In little endian, qubit 0 is the last factor in kron if we do kron(A, B, C) -> A is MSB.
        # So if we construct kron(A_{n-1}, ..., A_1, A_0), A_0 is qubit 0.
        ops = [I] * n_qubits
        ops[qubit_idx] = op
        # Construct kron from MSB to LSB
        result = ops[-1]
        for i in range(len(ops) - 2, -1, -1):
            result = np.kron(ops[i], result)
        return result

    # ZZ terms
    for i in range(n - 1):
        # Z on i and Z on i+1
        # This is kron(Z on i, Z on i+1)
        # Using the helper:
        # We need Z_i Z_{i+1}
        # Let's build it explicitly
        # Z_i is Z at position i, I elsewhere.
        # Z_{i+1} is Z at position i+1, I elsewhere.
        # Product is Z at i and i+1.
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        zz_term = ops[-1]
        for k in range(len(ops) - 2, -1, -1):
            zz_term = np.kron(ops[k], zz_term)
        H -= coupling * zz_term

    # X terms
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        x_term = ops[-1]
        for k in range(len(ops) - 2, -1, -1):
            x_term = np.kron(ops[k], x_term)
        H -= field * x_term

    # Exact ground energy is the smallest eigenvalue
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0].real
    _ground_energy_cache[key] = e0
    return e0

def _state_from_params(params, n, layers):
    """
    Simulate the circuit.
    params: shape (layers, n)
    Circuit:
    Start |0>^n
    Layer 0: Ry(theta_0) on all
    CZ on adjacent pairs (0-1, 1-2, ...)
    Layer 1: Ry(theta_1) on all
    CZ on adjacent pairs
    Layer 2: Ry(theta_2) on all
    No CZ after last layer.
    """
    dim = 2 ** n
    # Initial state |0...0>
    # In little endian, |0...0> is index 0.
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    
    # Pre-compute Rz/Ry matrices? No, just apply.
    # Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
    
    # Helper to apply single qubit gate
    def apply_single_qubit_gate(psi, qubit_idx, gate, n_qubits):
        # gate is 2x2
        # Reshape psi to tensor of shape (2, 2, ..., 2)
        # In little endian, axis 0 corresponds to qubit 0?
        # If psi is flattened with index = sum b_i 2^i, 
        # then psi.reshape([2]*n) has axes (n-1, ..., 0) if we use C-order?
        # Let's check:
        # Index 1 (01 in binary) -> b_0=1, b_1=0.
        # If shape is (2, 2) for n=2.
        # psi[0,0] = |00> (idx 0)
        # psi[0,1] = |01> (idx 1)? Or psi[1,0]?
        # NumPy reshape is C-order (row-major).
        # For n=2, shape (2,2).
        # Index 0: 0,0
        # Index 1: 0,1
        # Index 2: 1,0
        # Index 3: 1,1
        # So axis 1 is the least significant bit (qubit 0).
        # Axis 0 is the most significant bit (qubit n-1).
        
        # To apply gate on qubit_idx:
        # The axis corresponding to qubit_idx is (n_qubits - 1 - qubit_idx).
        axis = n_qubits - 1 - qubit_idx
        
        # Move axis to end, reshape, apply, reshape back
        psi_t = psi.reshape([2] * n_qubits)
        # Move target axis to the end
        psi_t = np.moveaxis(psi_t, axis, -1)
        # Reshape to (-1, 2)
        psi_t = psi_t.reshape(-1, 2)
        # Apply gate
        psi_t = np.einsum('ij,kj->ki', gate, psi_t)
        # Reshape back
        psi_t = psi_t.reshape([2] * n_qubits)
        # Move axis back
        psi_t = np.moveaxis(psi_t, -1, axis)
        return psi_t.reshape(-1)

    # Helper to apply CZ gate
    def apply_cz(psi, qubit_i, qubit_j, n_qubits):
        # CZ is diagonal.
        # Only affects states where both bits are 1.
        # In little endian, bit i and bit j.
        # We can iterate or use tensor ops.
        # Tensor op:
        psi_t = psi.reshape([2] * n_qubits)
        axis_i = n_qubits - 1 - qubit_i
        axis_j = n_qubits - 1 - qubit_j
        
        # Ensure axis_i < axis_j for consistency in moveaxis logic if needed, 
        # but moveaxis handles any order.
        
        # We need to multiply by -1 if bit_i=1 and bit_j=1.
        # Create a mask or just iterate over the two axes.
        # Efficient way:
        # Move both axes to the end
        # This is tricky with moveaxis if they are not adjacent.
        # Simpler: just loop over the 4 combinations of the two qubits?
        # No, n is small.
        
        # Let's use the property that CZ is diagonal.
        # We can construct the CZ operator and multiply? 
        # H is 2^n x 2^n. For n=6, 64x64. That's fine.
        # But applying gate directly is faster.
        
        # Let's stick to the moveaxis method for single qubit.
        # For 2-qubit, it's easier to just construct the unitary if n is small, 
        # or use the diagonal property.
        
        # Diagonal property:
        # For each basis state index k:
        # if (k >> qubit_i) & 1 == 1 and (k >> qubit_j) & 1 == 1:
        #     psi[k] *= -1
        
        # This is O(2^n) which is very fast for n=6 (64 ops).
        mask = np.zeros(dim, dtype=bool)
        for k in range(dim):
            if ((k >> qubit_i) & 1) and ((k >> qubit_j) & 1):
                mask[k] = True
        psi[mask] *= -1
        return psi

    for layer in range(layers):
        # Apply Ry
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            ry = np.array([[c, -s], [s, c]], dtype=complex)
            psi = apply_single_qubit_gate(psi, q, ry, n)
            
        # Apply CZ if not last layer
        if layer < layers - 1:
            for q in range(n - 1):
                psi = apply_cz(psi, q, q + 1, n)
                
    return psi

def _energy(params, n, coupling, field, e0):
    """Calculate the energy of the state generated by params."""
    psi = _state_from_params(params, n, layers)
    # Energy = <psi|H|psi>
    # We need H. Let's rebuild H or cache it.
    # Rebuilding H is expensive. Let's cache H too.
    key = (n, coupling, field)
    if key not in _ground_energy_cache or not hasattr(_ground_energy_cache, 'H'):
         # We stored only e0 in _ground_energy_cache. 
         # Let's create a separate cache for H.
         pass
         
    # For efficiency, let's just re-compute H in a helper or store it.
    # Since n is small, we can just build H every time? 
    # Building H takes ~n * 2^n ops. 6 * 64 = 384 ops. Very fast.
    # But eigvalsh is the slow part. We already cached e0.
    # We need H for the expectation value.
    
    # Let's refactor to cache H.
    if 'H_cache' not in globals():
        _H_cache = {}
    else:
        _H_cache = globals()['_H_cache']
        
    if key not in _H_cache:
         # Re-use logic from _get_ground_energy to build H
         dim = 2 ** n
         H = np.zeros((dim, dim), dtype=complex)
         I = np.eye(2, dtype=complex)
         X = np.array([[0, 1], [1, 0]], dtype=complex)
         Z = np.array([[1, 0], [0, -1]], dtype=complex)
         
         def build_op_single(op, qubit_idx, n_qubits):
             ops = [I] * n_qubits
             ops[qubit_idx] = op
             result = ops[-1]
             for i in range(len(ops) - 2, -1, -1):
                 result = np.kron(ops[i], result)
             return result

         for i in range(n - 1):
             ops = [I] * n
             ops[i] = Z
             ops[i+1] = Z
             zz_term = ops[-1]
             for k in range(len(ops) - 2, -1, -1):
                 zz_term = np.kron(ops[k], zz_term)
             H -= coupling * zz_term

         for i in range(n):
             ops = [I] * n
             ops[i] = X
             x_term = ops[-1]
             for k in range(len(ops) - 2, -1, -1):
                 x_term = np.kron(ops[k], x_term)
             H -= field * x_term
             
         _H_cache[key] = H
         
    H = _H_cache[key]
    
    # Expectation value
    # psi is column vector
    energy = np.vdot(psi, H.dot(psi)).real
    return energy

def _objective_and_grad(params, n, coupling, field, e0):
    """
    Objective: mean (circuit energy - exact ground energy)/n
    Since e0 is constant, minimizing (E_circuit - e0)/n is same as minimizing E_circuit.
    However, the problem asks for the value, so we return the objective value.
    Gradient: dE/dtheta.
    """
    # Finite difference gradient is slow but robust.
    # Analytical gradient is better.
    # Let's use finite difference for simplicity and correctness, 
    # given the small number of parameters (18 for n=6).
    # 18 params * 2 evals * cost of eval.
    # Cost of eval is ~ n * 2^n = 6*64 = 384 flops + overhead.
    # This is extremely fast.
    
    energy = _energy(params, n, coupling, field, e0)
    obj_val = (energy - e0) / n
    
    # Gradient via central difference
    grad = np.zeros_like(params)
    h = 1e-6
    for idx in np.ndindex(params.shape):
        old_val = params[idx]
        params[idx] = old_val + h
        f_plus = _energy(params, n, coupling, field, e0)
        params[idx] = old_val - h
        f_minus = _energy(params, n, coupling, field, e0)
        params[idx] = old_val
        grad[idx] = (f_plus - f_minus) / (2 * h) / n
        
    return obj_val, grad

def _optimize_one(n, coupling, field, e0, start_params, deadline):
    """Run optimization from a specific start."""
    layers = 3
    # Flatten params for scipy
    x0 = start_params.flatten()
    
    def fun(x):
        if time.time() > deadline:
            return np.inf
        p = x.reshape((layers, n))
        val, _ = _objective_and_grad(p, n, coupling, field, e0)
        return val
        
    def grad(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        p = x.reshape((layers, n))
        _, g = _objective_and_grad(p, n, coupling, field, e0)
        return g.flatten()
        
    try:
        res = opt.minimize(fun, x0, jac=grad, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        if time.time() > deadline:
            return None
        return res.x.reshape((layers, n)), res.fun
    except Exception:
        return None

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Get exact ground energy
    e0 = _get_ground_energy(n, coupling, field)
    
    # 2. Heuristic initialization
    # The ground state of TFIM is close to |+> for small field, 
    # and |0> or |1> for large field?
    # Actually, for transverse field, it's a superposition.
    # Let's try a few random starts and a heuristic start.
    
    best_params = None
    best_score = np.inf
    
    # Heuristic: 
    # For field ~ 1, the state is roughly uniform in X basis? 
    # Ry(pi/2) gives |+>. 
    # Let's start with all pi/2.
    init1 = np.full((layers, n), np.pi / 2)
    
    # Random starts
    np.random.seed(42)
    inits = [init1]
    for _ in range(5):
        # Random angles between 0 and pi
        inits.append(np.random.uniform(0, np.pi, (layers, n)))
        
    # Optimization
    for start in inits:
        if time.time() > deadline:
            break
            
        # Optimize
        result = _optimize_one(n, coupling, field, e0, start, deadline)
        if result:
            params, score = result
            if score < best_score:
                best_score = score
                best_params = params
                
    # If no optimization succeeded, return heuristic
    if best_params is None:
        return init1
        
    return best_params

# Initialize global cache if needed
_H_cache = {}
