"""Combined program for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}

def _build_hamiltonian(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    I = np.eye(2, dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_list(ops)
        
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
        
    return H

def _get_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0]

def _circuit_state_forward(theta, n, layers):
    """
    Forward pass of the circuit.
    Returns the final state and a list of intermediate states for backprop.
    state is a flat vector of length 2^n.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    # Store states after each Ry gate for backprop
    # Structure: states[l][i] is the state after applying Ry(l, i)
    # We only need the state *before* the gate to compute the gradient term <psi_before| dU/dtheta |psi_before>? 
    # No, the gradient is <psi_after| dU/dtheta |psi_before>.
    # So we need psi_before and psi_after for each gate.
    # Storing all states is 3*n * dim floats. For n=6, dim=64, 3*6*64 = 1152 floats. Very small.
    
    states_before = []
    
    for l in range(layers):
        for i in range(n):
            states_before.append(state.copy())
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            
            mask = 1 << i
            idx0 = np.arange(dim)
            idx0 = idx0[(idx0 & mask) == 0]
            idx1 = idx0 | mask
            
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if l < layers - 1:
            # Apply CZ on adjacent pairs
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                both_set = (np.arange(dim) & mask_i) != 0
                both_set &= (np.arange(dim) & mask_ip1) != 0
                state[both_set] *= -1.0
                
    return state, states_before

def _energy_and_grad_analytical(theta_flat, n, coupling, field, H):
    """
    Compute energy and analytical gradient using the adjoint method.
    """
    theta = theta_flat.reshape((3, n))
    layers = 3
    dim = 1 << n
    
    # Forward pass
    state_final, states_before = _circuit_state_forward(theta, n, layers)
    
    # Energy
    H_state = H @ state_final
    energy = np.dot(state_final, H_state)
    
    # Backward pass
    # Adjoint state starts as H |psi_final>
    adj_state = H_state.copy()
    
    grad = np.zeros_like(theta_flat)
    
    # We need to iterate backwards through the gates
    # The gates are applied in order: Ry(0,0), Ry(0,1)..., CZs, Ry(1,0)...
    # So we reverse this order.
    
    gate_idx = 0 # index into states_before (0 to layers*n - 1)
    # We iterate l from layers-1 down to 0
    # Inside l, we iterate i from n-1 down to 0
    
    # We need to reconstruct the CZ application in reverse
    # CZ is self-adjoint, so applying CZ in reverse is just applying CZ again.
    
    # The sequence of operations in forward pass:
    # for l in range(layers):
    #   for i in range(n):
    #     Ry(l, i)
    #   if l < layers - 1:
    #     CZs
    #
    # Reverse sequence:
    # for l in range(layers-1, -1, -1):
    #   if l < layers - 1:
    #     CZs (reverse)
    #   for i in range(n-1, -1, -1):
    #     Ry(l, i) (reverse)
    
    # We have states_before[k] corresponding to the state *before* the k-th Ry gate.
    # The k-th Ry gate is at layer l = k // n, index i = k % n.
    # Wait, the CZs are between layers.
    # states_before[k] is the state before Ry(k).
    # After Ry(k), the state is updated.
    # If k % n == n-1 and k < (layers-1)*n, then CZs are applied after Ry(k).
    
    # Let's trace the adjoint state back.
    
    # Start with adj_state = H |psi_final>
    
    for l in range(layers - 1, -1, -1):
        # If this layer was followed by CZs, we must reverse them first
        if l < layers - 1:
            # Apply CZs in reverse order? CZs are applied in ascending order i=0..n-2.
            # Since CZs commute? No, CZ(i) and CZ(i+1) do not commute if they share a qubit?
            # CZ(0) acts on q0,q1. CZ(1) acts on q1,q2.
            # They don't commute.
            # So we must reverse the order: apply CZ(n-2), ..., CZ(0).
            for i in range(n - 2, -1, -1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                both_set = (np.arange(dim) & mask_i) != 0
                both_set &= (np.arange(dim) & mask_ip1) != 0
                adj_state[both_set] *= -1.0

        # Now reverse the Ry gates for this layer
        for i in range(n - 1, -1, -1):
            k = l * n + i
            t = theta[l, i]
            
            # Gradient contribution:
            # dE/dtheta = 2 Re <adj_state_after | dRy/dtheta | state_before>
            # Note: adj_state here is the adjoint state *after* the gate (before reversing the gate).
            # The adjoint state *before* the gate will be R_y^dagger |adj_state_after>.
            
            mask = 1 << i
            idx0 = np.arange(dim)
            idx0 = idx0[(idx0 & mask) == 0]
            idx1 = idx0 | mask
            
            state_before = states_before[k]
            
            # Extract relevant components
            sb0 = state_before[idx0]
            sb1 = state_before[idx1]
            sa0 = adj_state[idx0]
            sa1 = adj_state[idx1]
            
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            
            # dRy/dtheta matrix elements:
            # dRy/dtheta = 0.5 * [ [-sin, -cos], [cos, -sin] ]
            # Let M = dRy/dtheta
            # <sa| M |sb> = 0.5 * ( -s*sa0*sb0 - c*sa0*sb1 + c*sa1*sb0 - s*sa1*sb1 )
            
            term = 0.5 * (
                -s * np.dot(sa0, sb0) - c * np.dot(sa0, sb1) + 
                c * np.dot(sa1, sb0) - s * np.dot(sa1, sb1)
            )
            
            grad[k] = 2.0 * np.real(term)
            
            # Update adj_state to be before the gate:
            # adj_before = Ry^dagger |adj_after>
            # Ry^dagger = Ry(-t) = [ [c, s], [-s, c] ]
            # adj0_before = c*sa0 + s*sa1
            # adj1_before = -s*sa0 + c*sa1
            
            adj_state[idx0] = c * sa0 + s * sa1
            adj_state[idx1] = -s * sa0 + c * sa1
            
    return energy, grad

def _physics_init(n, field):
    """Physics-inspired initialization based on arctan(1/field)."""
    theta = np.zeros((3, n))
    base_angle = np.arctan(1.0 / field)
    theta[0] = base_angle * 0.3
    theta[1] = base_angle * 0.5
    theta[2] = base_angle
    return theta

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
        
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    H = _build_hamiltonian(n, coupling, field)
    E0 = _get_ground_energy(n, coupling, field)
    
    # Multiple initial guesses
    theta0_seed = np.zeros((layers, n))
    theta0_seed[-1] = np.pi / 2
    
    theta0_physics = _physics_init(n, field)
    
    best_theta = theta0_seed.copy()
    best_energy, _ = _energy_and_grad_analytical(theta0_seed.flatten(), n, coupling, field, H)
    
    # Try physics init as well
    e_phys, _ = _energy_and_grad_analytical(theta0_physics.flatten(), n, coupling, field, H)
    if e_phys < best_energy:
        best_energy = e_phys
        best_theta = theta0_physics.copy()
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: L-BFGS-B with analytical gradient from multiple starting points
    starts = [theta0_seed, theta0_physics]
    
    # Add structured random starts
    for r in range(8):
        if time.time() > deadline - 15:
            break
        
        if r % 2 == 0:
            current_theta = rng.uniform(-np.pi, np.pi, size=(layers, n))
        else:
            current_theta = best_theta + rng.normal(0, 0.5, size=(layers, n))
        starts.append(current_theta)
    
    def objective_with_grad(x):
        """Objective function with analytical gradient for L-BFGS-B."""
        e, g = _energy_and_grad_analytical(x, n, coupling, field, H)
        return e, g
    
    for st in starts:
        if time.time() > deadline - 10:
            break
        
        try:
            res = opt.minimize(
                objective_with_grad, 
                st.flatten(), 
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Strategy 2: Nelder-Mead polish from best (derivative-free, robust)
    if time.time() < deadline - 20:
        try:
            def objective_only(x):
                e, _ = _energy_and_grad_analytical(x, n, coupling, field, H)
                return e
            
            res = opt.minimize(
                objective_only, 
                best_theta.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Final polish with L-BFGS-B from best using gradient
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(
                objective_with_grad, 
                best_theta.flatten(), 
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Ensure output is finite real array
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = theta0_seed
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
