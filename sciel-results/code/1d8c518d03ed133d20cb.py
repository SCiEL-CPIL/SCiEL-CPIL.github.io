"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

# Cache for optimized angles
_ANGLE_CACHE = {}

def _build_hamiltonian(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_list(ops)
        
    # X terms: -field * sum X_i
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
        
    return H

def _get_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    # Use eigh for Hermitian matrices
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0].real

def _apply_cz_layer(state, n):
    # state is a vector of size 2^n
    # Apply CZ on (0,1), (1,2), ..., (n-2, n-1)
    # CZ = diag(1, 1, 1, -1) in basis 00, 01, 10, 11
    # It's easier to iterate or use matrix multiplication.
    # For n<=6, we can build the CZ operator for the pair and apply.
    # Or simply loop through basis states? No, that's O(2^n * n).
    # Let's build the full CZ layer operator or apply sequentially.
    
    # Sequential application is safer and clear.
    for i in range(n - 1):
        # Build CZ operator for qubits i and i+1
        # Identity on others
        ops = [np.eye(2, dtype=complex)] * n
        cz_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)
        # We need to insert cz_mat into the kron product.
        # This is expensive if done naively.
        # Alternative: Use the property that CZ is diagonal in Z basis.
        # CZ |b> = (-1)^{b_i b_{i+1}} |b>
        # So we can just multiply the state vector by a phase factor for each basis state.
        # This is O(2^n).
        dim = 1 << n
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            if bit_i and bit_ip1:
                state[b] *= -1
    return state

def _circuit_state(theta, n, layers):
    # theta is (layers, n)
    # Start with |0>^n
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0 # |0...0>
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        # Ry(t) = cos(t/2) I - i sin(t/2) X
        # We can apply these as a tensor product of single-qubit gates.
        # Since they are all independent, the total gate is kron(Ry_0, Ry_1, ...)
        # But building the full 2^n x 2^n matrix is expensive.
        # However, we can apply them sequentially or use the fact that they commute.
        # Actually, applying Ry to each qubit is just a diagonal-like operation in a rotated basis?
        # No, Ry is not diagonal.
        # But we can construct the full operator for the Ry layer.
        ry_ops = []
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2)
            s = np.sin(t / 2)
            ry = np.array([[c, -1j*s], [1j*s, c]], dtype=complex)
            ry_ops.append(ry)
        
        # Construct full Ry layer matrix
        Ry_layer = ry_ops[0]
        for op in ry_ops[1:]:
            Ry_layer = np.kron(Ry_layer, op)
            
        state = Ry_layer @ state
        
        # Apply CZ layer if not the last layer
        if l < layers - 1:
            state = _apply_cz_layer(state, n)
            
    return state

def _energy(theta, n, coupling, field, H, E0):
    state = _circuit_state(theta, n, 3)
    # Energy = <psi| H |psi>
    # Use real part since it should be real
    e_val = np.vdot(state, H @ state).real
    # The objective is (E_circuit - E0)/n. We minimize this.
    # Since E0 is constant, minimizing E_circuit is equivalent.
    return e_val

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
        
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Get ground energy
    E0 = _get_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    # 2. Optimization
    # Initial guess: Seed angles
    # theta shape (3, n)
    theta0 = np.zeros((layers, n))
    theta0[-1] = np.pi / 2
    
    best_theta = theta0
    best_energy = _energy(best_theta, n, coupling, field, H, E0)
    
    # Try a few random restarts and the seed
    num_restarts = 5
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            current_theta = theta0
        else:
            # Random initialization
            # Angles in [-pi, pi]
            current_theta = rng.uniform(-np.pi, np.pi, size=(layers, n))
            
        # Optimize
        try:
            res = opt.minimize(
                _energy, 
                current_theta.flatten(), 
                args=(n, coupling, field, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
                
        except Exception:
            continue
            
    # Store result
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
