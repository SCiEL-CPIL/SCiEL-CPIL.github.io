"""
Optimized shallow-circuit Ising ground state approximation.
Strategy: Nonlinear least-squares statevector matching with vectorized circuit simulation.
Instead of minimizing the energy (a scalar), we directly match the circuit's output
statevector components to the exact ground statevector using scipy.optimize.least_squares.
This provides a richer loss landscape (2^n real components vs 1 scalar) and can be
more robust. We also use a fully vectorized circuit application for speed.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares, minimize
import time

# Caches
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
    Uses vectorized operations where possible.
    theta: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    layers = theta.shape[0]
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        for i in range(n):
            c = np.cos(theta[l, i] / 2)
            s = np.sin(theta[l, i] / 2)
            
            # Create masks for bit i being 0 or 1
            mask_0 = (np.arange(dim) >> i) & 1
            mask_1 = ~mask_0
            
            # Extract amplitudes for bit i = 0 and bit i = 1
            amp_0 = state[mask_0]
            amp_1 = state[mask_1]
            
            # Apply Ry rotation
            new_amp_0 = c * amp_0 - s * amp_1
            new_amp_1 = s * amp_0 + c * amp_1
            
            # Write back
            state[mask_0] = new_amp_0
            state[mask_1] = new_amp_1
            
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                mask = (1 << i) | (1 << (i + 1))
                # Find indices where both bits are 1
                idx = np.where(((np.arange(dim) >> i) & 1) == 1)[0]
                idx = idx[((idx >> (i + 1)) & 1) == 1]
                state[idx] *= -1.0
                
    return state

def _compute_energy(theta, n, coupling, field):
    """Compute the energy of the state produced by the circuit."""
    state = _apply_circuit_vectorized(theta, n)
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    Hpsi = H.dot(state)
    energy = np.vdot(state, Hpsi).real
    return energy

def _statevector_residuals(params, n, psi0):
    """
    Compute residuals for least_squares: difference between circuit state and target state.
    Returns real and imaginary parts concatenated.
    """
    theta = params.reshape(3, n)
    circuit_state = _apply_circuit_vectorized(theta, n)
    
    # Residuals: real and imaginary parts of (circuit_state - psi0)
    diff = circuit_state - psi0
    return np.concatenate([diff.real, diff.imag])

def _generate_initializers(n, coupling, field, layers, psi0):
    """
    Generate diverse initial guesses for the parameters.
    """
    inits = []
    
    # 1. Mean-field guess
    def prod_energy(theta):
        return -coupling * (n - 1) * np.cos(theta)**2 - field * n * np.sin(theta)
    
    thetas = np.linspace(-np.pi, np.pi, 1000)
    energies = np.array([prod_energy(t) for t in thetas])
    idx_min = np.argmin(energies)
    theta_opt = thetas[idx_min]
    res = minimize(prod_energy, theta_opt, method='BFGS')
    if res.success:
        theta_opt = res.x[0]
    inits.append(np.full((layers, n), theta_opt))
    
    # 2. Strong field limit (all pi/2)
    inits.append(np.full((layers, n), np.pi/2))
    
    # 3. Strong coupling limit (all 0)
    inits.append(np.zeros((layers, n)))
    
    # 4. Alternating pattern
    alt = np.zeros((layers, n))
    for l in range(layers):
        for i in range(n):
            alt[l, i] = np.pi/4 if i % 2 == 0 else -np.pi/4
    inits.append(alt)
    
    # 5. Local magnetization based
    dim = 1 << n
    theta_mag = np.zeros((layers, n))
    for i in range(n):
        exp_z = 0.0
        for state in range(dim):
            amp = psi0[state]
            prob = np.abs(amp)**2
            bit = (state >> i) & 1
            exp_z += prob * (1 - 2*bit)
        
        if exp_z > 1.0: exp_z = 1.0
        if exp_z < -1.0: exp_z = -1.0
        theta_last = np.arccos(exp_z)
        theta_mag[layers-1, i] = theta_last
        for l in range(layers-1):
            theta_mag[l, i] = 0.1 * (l+1)
    inits.append(theta_mag)
    
    # 6-10. Random uniform
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(rng.uniform(-np.pi, np.pi, size=(layers, n)))
        
    return inits

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Compute exact ground state
    e_exact, psi0 = _compute_exact_ground_state(n, coupling, field)
    
    # 2. Generate initializers
    inits = _generate_initializers(n, coupling, field, layers, psi0)
    
    best_x = None
    best_val = np.inf
    
    # 3. Multi-start optimization using least_squares for statevector matching
    for init in inits:
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        
        # Quick check of initial value using energy
        val0 = _compute_energy(x0.reshape(layers, n), n, coupling, field) - e_exact
        if val0 < best_val:
            best_val = val0
            best_x = x0
            
        # Run least_squares for statevector matching
        try:
            res = least_squares(
                _statevector_residuals,
                x0,
                args=(n, psi0),
                method='lm',  # Levenberg-Marquardt, good for least squares
                max_nfev=500
            )
            
            # Evaluate the energy of the result
            theta_res = res.x.reshape(layers, n)
            e_res = _compute_energy(theta_res, n, coupling, field) - e_exact
            
            if e_res < best_val:
                best_val = e_res
                best_x = res.x
                
        except Exception:
            pass
            
    # 4. If time remains, do a second round of basin hopping around the best solution
    if best_x is not None and time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        current_x = best_x
        current_val = best_val
        
        # Basin Hopping: perturb and re-optimize
        for _ in range(5):
            if time.time() > deadline:
                break
                
            # Perturb
            noise = rng.normal(0, 0.5, size=current_x.shape)
            perturbed_x = current_x + noise
            
            # Re-optimize from perturbed point using least_squares
            try:
                res = least_squares(
                    _statevector_residuals,
                    perturbed_x,
                    args=(n, psi0),
                    method='lm',
                    max_nfev=300
                )
                
                theta_res = res.x.reshape(layers, n)
                e_res = _compute_energy(theta_res, n, coupling, field) - e_exact
                
                if e_res < best_val:
                    best_val = e_res
                    best_x = res.x
                    current_x = best_x
                    current_val = best_val
                    
            except Exception:
                pass

    # Fallback if no optimization succeeded
    if best_x is None:
        # Return mean-field guess
        theta_opt = np.full((layers, n), 0.5)
        return theta_opt

    return best_x.reshape(layers, n)

# EVOLVE-BLOCK-END
