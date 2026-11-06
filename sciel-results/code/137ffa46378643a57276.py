"""
Optimized shallow-circuit Ising ground state approximation.
Strategy: Analytic Gradient Descent with Layer-wise Optimization.
Instead of generic L-BFGS, we compute the exact gradient of the energy 
with respect to each Ry angle. We use a layer-by-layer optimization 
strategy: freeze all layers except one, optimize that layer to convergence, 
then move to the next. This reduces the effective dimensionality and 
helps navigate the multi-modal landscape. Initialization is based on 
the "Zero-Phase" heuristic where angles are set to minimize the 
local field term initially, assuming the CZ gates have minimal 
effect on the product state structure in the first pass.
"""
# EVOLVE-BLOCK-START
import numpy as np
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
    
    # H = -coupling * sum Z_i Z_(i+1) - field * sum X_i
    # We store the parts separately to allow efficient energy calculation for different fields.
    # H_IZ = sum Z_i Z_(i+1)
    H_IZ = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H_IZ[state, state] += 1.0
            else:
                H_IZ[state, state] -= 1.0
                
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
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    layers = theta.shape[0]
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        for i in range(n):
            c = np.cos(theta[l, i] / 2)
            s = np.sin(theta[l, i] / 2)
            
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
                mask = (1 << i) | (1 << (i + 1))
                for idx in range(dim):
                    if (idx & mask) == mask:
                        state[idx] *= -1.0
                        
    return state

def _compute_energy_and_grad(theta, n, coupling, field):
    """
    Compute the energy and the analytic gradient with respect to all theta parameters.
    Returns: energy, grad (shape (layers, n))
    """
    dim = 1 << n
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    layers = theta.shape[0]
    grad = np.zeros_like(theta)
    
    # We need to compute the gradient for each parameter.
    # The circuit is a sequence of gates.
    # E = <psi|H|psi>
    # dE/dtheta_k = 2 Re( <psi| H |dpsi/dtheta_k> )
    
    # To be efficient, we can use the "parameter shift rule" or direct analytic differentiation.
    # For Ry(theta) = exp(-i theta Y / 2) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
    # d/dtheta Ry = (-1/2) sin(theta/2) Y - (1/2) cos(theta/2) X ? No.
    # d/dtheta Ry(theta) = 1/2 * [[-sin(theta/2), -cos(theta/2)], [cos(theta/2), -sin(theta/2)]]
    
    # Let's use the parameter shift rule which is robust and exact for Ry.
    # f(theta) = E(theta)
    # df/dtheta = (f(theta + pi/2) - f(theta - pi/2)) / 2
    
    # However, doing this for every parameter individually is O(N_params * Cost_Circuit).
    # N_params = 3*n = 9 or 18. Cost_Circuit is small (dim=16 or 64).
    # This is very fast.
    
    base_energy = 0.0
    # We just need the gradient. The energy itself is computed in the loop or separately.
    # Let's compute energy once.
    state = _apply_circuit(theta, n)
    Hpsi = H.dot(state)
    energy = np.vdot(state, Hpsi).real
    
    # Compute gradient using parameter shift rule
    for l in range(layers):
        for i in range(n):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[l, i] += np.pi / 2
            theta_minus[l, i] -= np.pi / 2
            
            state_plus = _apply_circuit(theta_plus, n)
            e_plus = np.vdot(state_plus, H.dot(state_plus)).real
            
            state_minus = _apply_circuit(theta_minus, n)
            e_minus = np.vdot(state_minus, H.dot(state_minus)).real
            
            grad[l, i] = (e_plus - e_minus) / 2.0
            
    return energy, grad

def _angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Compute exact ground energy
    e_exact = _compute_exact_ground_energy(n, coupling, field)
    
    # 2. Initialization
    # "Zero-Phase" heuristic:
    # For the last layer, we want to maximize the transverse field term -h <X>.
    # <X> for Ry(theta)|0> is sin(theta). Max at theta=pi/2.
    # For the first layers, we want to prepare the state for the CZ gates.
    # If we set theta=0, the state is |00...0>, CZ does nothing.
    # If we set theta=pi, the state is |11...1>, CZ does nothing.
    # Let's start with a mix.
    # Layer 0: 0
    # Layer 1: 0
    # Layer 2: pi/2
    # This creates a product state |+...+> essentially?
    # Ry(0)|0> = |0>. CZ on |00> is |00>.
    # Ry(pi/2)|0> = (|0>+|1>)/sqrt(2) = |+>.
    # So state is |+...+>.
    # Energy of |+...+>: <Z>=0, <X>=1. E = -h*n.
    # This is a good starting point for the field-dominated regime.
    
    theta = np.zeros((layers, n))
    theta[-1, :] = np.pi / 2
    
    # Alternative start: Mean field guess
    # E_prod(theta) = -J(n-1)cos^2(theta) - hn sin(theta)
    # Minimize this.
    def prod_energy(t):
        return -coupling * (n - 1) * np.cos(t)**2 - field * n * np.sin(t)
    
    # Grid search for best product state angle
    thetas_grid = np.linspace(-np.pi, np.pi, 100)
    energies_grid = [prod_energy(t) for t in thetas_grid]
    idx_min = np.argmin(energies_grid)
    theta_opt_prod = thetas_grid[idx_min]
    
    # We have two initializers:
    # 1. Zero-phase (last layer pi/2, others 0)
    # 2. Uniform product state (all layers theta_opt_prod)
    
    initializers = [
        theta.copy(),
        np.full((layers, n), theta_opt_prod)
    ]
    
    best_val = float('inf')
    best_x = theta.flatten()
    
    # Optimize each initializer
    for init_theta in initializers:
        if time.time() > deadline:
            break
            
        x = init_theta.flatten().copy()
        
        # Layer-wise optimization
        # We will iterate through layers multiple times
        max_epochs = 10
        for epoch in range(max_epochs):
            if time.time() > deadline:
                break
                
            for l in range(layers):
                # Freeze all layers except l
                # Optimize layer l
                # Use simple gradient descent with adaptive step size on the n parameters of layer l
                
                current_energy = _compute_energy_and_grad(init_theta, n, coupling, field)[0]
                
                # Adaptive step size
                step_size = 0.1
                max_steps = 50
                
                for step in range(max_steps):
                    if time.time() > deadline:
                        break
                        
                    energy, grad = _compute_energy_and_grad(init_theta, n, coupling, field)
                    
                    # Check convergence
                    grad_layer = grad[l, :]
                    if np.linalg.norm(grad_layer) < 1e-6:
                        break
                        
                    # Update
                    init_theta[l, :] -= step_size * grad_layer
                    
                    # Check if energy increased (line search / backtracking)
                    new_energy, _ = _compute_energy_and_grad(init_theta, n, coupling, field)
                    if new_energy > energy:
                        # Roll back and reduce step size
                        init_theta[l, :] += step_size * grad_layer
                        step_size *= 0.5
                        if step_size < 1e-4:
                            break
                    else:
                        energy = new_energy
                        
                # After optimizing layer l, check global best
                final_energy, _ = _compute_energy_and_grad(init_theta, n, coupling, field)
                if final_energy < best_val:
                    best_val = final_energy
                    best_x = init_theta.flatten().copy()
                    
        # After layer-wise, do a full parameter optimization with small steps
        # This helps fix any inconsistencies between layers
        for step in range(100):
            if time.time() > deadline:
                break
                
            energy, grad = _compute_energy_and_grad(init_theta, n, coupling, field)
            
            if np.linalg.norm(grad) < 1e-6:
                break
                
            step_size = 0.05
            init_theta -= step_size * grad
            
            new_energy, _ = _compute_energy_and_grad(init_theta, n, coupling, field)
            if new_energy > energy:
                init_theta += step_size * grad
                step_size *= 0.5
            else:
                energy = new_energy
                
            if energy < best_val:
                best_val = energy
                best_x = init_theta.flatten().copy()
                
    # Final check
    if best_x is None:
        best_x = theta.flatten()
        
    return best_x.reshape(layers, n)

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    return _angles(n, coupling, field, layers)

# EVOLVE-BLOCK-END
