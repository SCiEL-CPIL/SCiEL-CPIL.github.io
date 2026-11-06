"""
Optimized shallow-circuit Ising ground state approximation.
Strategy: Direct state-vector matching via nonlinear least-squares (scipy.optimize.least_squares).
Instead of minimizing the scalar energy (which is a smooth but potentially misleading landscape),
we directly minimize the L2 distance between the circuit's output state vector and the exact
ground state vector. This transforms the problem into a high-dimensional (2^n real + 2^n imaginary)
least-squares problem. The objective is non-smooth at phase boundaries but often has a more
direct path to the global minimum for small n (4, 6) where the state vector is only 16 or 64
dimensional. We use a vectorized circuit simulator for speed and scipy's robust least_squares
method (trf) which handles bounds and is less prone to local minima in high-dimensional spaces
than standard gradient descent on the energy.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares, minimize
import time

# Cache for exact ground energies and Hamiltonians
_HAM_CACHE = {}
_GS_CACHE = {}

def _get_hamiltonian(n, coupling):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    key = (n, coupling)
    if key in _HAM_CACHE:
        return _HAM_CACHE[key]
    
    dim = 1 << n
    
    # H = -coupling * sum Z_i Z_(i+1) - field * sum X_i
    H_IZ = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H_IZ[state, state] += 1.0
            else:
                H_IZ[state, state] -= 1.0
                
    H_IX = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H_IX[state, flipped] = 1.0
            
    _HAM_CACHE[key] = (H_IZ, H_IX)
    return _HAM_CACHE[key]

def _compute_exact_ground_state(n, coupling, field):
    """Compute the exact ground state vector and energy."""
    key = (n, coupling, field)
    if key in _GS_CACHE:
        return _GS_CACHE[key]
    
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = eigvals[0].real
    psi0 = eigvecs[:, 0]
    
    _GS_CACHE[key] = (e0, psi0)
    return _GS_CACHE[key]

def _apply_circuit_vectorized(theta, n):
    """
    Apply the circuit to |0...0> and return the state vector.
    Uses vectorized operations for speed.
    theta: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    layers = theta.shape[0]
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        for i in range(n):
            th = theta[l, i]
            c = np.cos(th / 2)
            s = np.sin(th / 2)
            
            # Create mask for qubit i being 0 and 1
            mask_0 = (np.arange(dim) & (1 << i)) == 0
            mask_1 = ~mask_0
            
            # Partner indices
            partner = np.arange(dim) ^ (1 << i)
            
            # Update state
            # new_state[idx] = c * state[idx] - s * state[partner] for idx in mask_0
            # new_state[partner] = s * state[idx] + c * state[partner] for idx in mask_0
            
            new_state = np.zeros_like(state)
            # For indices where qubit i is 0:
            new_state[mask_0] = c * state[mask_0] - s * state[partner[mask_0]]
            # For indices where qubit i is 1:
            new_state[mask_1] = s * state[partner[mask_1]] + c * state[mask_1]
            
            state = new_state
            
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                mask = (1 << i) | (1 << (i + 1))
                # Find indices where both qubits are 1
                idxs = np.where((np.arange(dim) & mask) == mask)[0]
                state[idxs] *= -1.0
                
    return state

def _state_residuals(params, n, layers, psi_target):
    """
    Residuals for least_squares: difference between circuit state and target state.
    Returns real and imaginary parts flattened.
    """
    theta = params.reshape(layers, n)
    state = _apply_circuit_vectorized(theta, n)
    residual = state - psi_target
    # Flatten real and imaginary parts
    return np.concatenate([residual.real, residual.imag])

def _energy_residuals(params, n, layers, coupling, field, e_target):
    """
    Residuals for energy minimization (alternative, but we prefer state matching).
    """
    theta = params.reshape(layers, n)
    state = _apply_circuit_vectorized(theta, n)
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    Hpsi = H.dot(state)
    energy = np.vdot(state, Hpsi).real
    return [energy - e_target]

def _generate_initializers(n, coupling, field, layers, psi0):
    """
    Generate diverse initial guesses.
    """
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Zero
    inits.append(np.zeros((layers, n)))
    
    # 2. Pi/2 (strong field)
    inits.append(np.full((layers, n), np.pi/2))
    
    # 3. Small angles
    inits.append(np.full((layers, n), 0.2))
    
    # 4. Random
    for _ in range(10):
        inits.append(rng.uniform(-np.pi, np.pi, size=(layers, n)))
        
    # 5. Heuristic based on <Z>
    dim = 1 << n
    theta_z = np.zeros((layers, n))
    for i in range(n):
        exp_z = 0.0
        for state in range(dim):
            amp = psi0[state]
            prob = np.abs(amp)**2
            bit = (state >> i) & 1
            exp_z += prob * (1 - 2*bit)
        exp_z = np.clip(exp_z, -1.0, 1.0)
        theta_last = np.arccos(exp_z)
        theta_z[layers-1, i] = theta_last
    inits.append(theta_z)
    
    return inits

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Compute exact ground state
    e_exact, psi0 = _compute_exact_ground_state(n, coupling, field)
    
    # Normalize target state to ensure phase alignment isn't an issue for L2
    # Note: Global phase is arbitrary. Least squares on state vectors is sensitive to phase.
    # We can fix the phase of the first non-zero element of psi0 to be real and positive.
    # Or better: minimize |<psi_circ|psi_target>|^2 - 1 which is phase invariant.
    # However, least_squares requires a vector residual.
    # Let's stick to state matching but align phases.
    
    # Find first significant element
    idx_max = np.argmax(np.abs(psi0))
    phase = np.angle(psi0[idx_max])
    psi0_aligned = psi0 * np.exp(-1j * phase)
    
    def _objective_state(params, n, layers, psi_target):
        return _state_residuals(params, n, layers, psi_target)
    
    best_x = None
    best_val = np.inf
    
    # 2. Generate initializers
    inits = _generate_initializers(n, coupling, field, layers, psi0)
    
    # 3. Multi-start optimization using least_squares
    for init in inits:
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        
        try:
            res = least_squares(
                _objective_state, 
                x0, 
                args=(n, layers, psi0_aligned),
                method='trf',
                bounds=([-np.pi]* (layers*n), [np.pi]*(layers*n)),
                max_nfev=1000,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12
            )
            
            # Calculate actual energy gap to compare
            theta = res.x.reshape(layers, n)
            state = _apply_circuit_vectorized(theta, n)
            H_IZ, H_IX = _get_hamiltonian(n, coupling)
            H = -coupling * H_IZ - field * H_IX
            Hpsi = H.dot(state)
            energy = np.vdot(state, Hpsi).real
            val = energy - e_exact
            
            if val < best_val:
                best_val = val
                best_x = res.x
                
        except Exception:
            pass
            
    # 4. If time remains, do a second round of refinement using L-BFGS-B on energy
    if best_x is not None and time.time() < deadline - 10.0:
        def _energy_obj(params):
            theta = params.reshape(layers, n)
            state = _apply_circuit_vectorized(theta, n)
            H_IZ, H_IX = _get_hamiltonian(n, coupling)
            H = -coupling * H_IZ - field * H_IX
            Hpsi = H.dot(state)
            energy = np.vdot(state, Hpsi).real
            return energy - e_exact

        try:
            res = minimize(
                _energy_obj, 
                best_x, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Fallback
    if best_x is None:
        return np.zeros((layers, n))

    return best_x.reshape(layers, n)

# EVOLVE-BLOCK-END
