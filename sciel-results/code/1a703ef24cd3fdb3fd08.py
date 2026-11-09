import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    Qubit i is bit i (little-endian).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H[b, b] -= coupling
            else:
                H[b, b] += coupling
            
    # X terms: -field * sum X_i
    for i in range(n):
        for b in range(dim):
            b_flipped = b ^ (1 << i)
            H[b_flipped, b] -= field
            
    return H

def _ground_energy(n, coupling, field):
    """Compute exact ground energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def _circuit_state(params, n, layers):
    """
    Apply the 3-layer Ry/CZ circuit to |0>^n and return the state vector.
    params: (layers, n) array of angles
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            state_tensor = state.reshape([2] * n)
            state_tensor = np.moveaxis(state_tensor, q, 0)
            
            new_tensor = np.empty_like(state_tensor)
            new_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
            new_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
            
            state_tensor = np.moveaxis(new_tensor, 0, q)
            state = state_tensor.reshape(dim)
        
        # Apply CZ between adjacent pairs (except after last layer)
        if l < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                indices = np.arange(dim)
                both_one = (indices & mask1) != 0
                both_one &= (indices & mask2) != 0
                state[both_one] *= -1
    
    return state

def _energy(params, n, layers, H, E0):
    """Compute energy minus ground energy."""
    psi = _circuit_state(params, n, layers)
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    return energy - E0

def _energy_and_grad(params, n, layers, H, E0):
    """Compute energy and analytic gradient via backpropagation."""
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    # Store states after each gate for backprop
    states = [state.copy()]
    ops = []
    
    for l in range(layers):
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            state_tensor = state.reshape([2] * n)
            state_tensor = np.moveaxis(state_tensor, q, 0)
            
            new_tensor = np.empty_like(state_tensor)
            new_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
            new_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
            
            state_tensor = np.moveaxis(new_tensor, 0, q)
            state = state_tensor.reshape(dim)
            states.append(state.copy())
            ops.append(('ry', l, q))
        
        if l < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                indices = np.arange(dim)
                both_one = (indices & mask1) != 0
                both_one &= (indices & mask2) != 0
                state[both_one] *= -1
                states.append(state.copy())
                ops.append(('cz', q))
    
    psi = states[-1]
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    
    # Backward pass
    v = Hpsi.copy()
    grad = np.zeros((layers, n))
    
    for i in range(len(ops) - 1, -1, -1):
        op = ops[i]
        psi_before = states[i]
        
        if op[0] == 'ry':
            l, q = op[1], op[2]
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # dU/dtheta = 0.5 * [[-s, -c], [c, -s]]
            # (dU/dtheta)^dagger = 0.5 * [[-s, c], [-c, -s]]
            v_tensor = v.reshape([2] * n)
            v_tensor = np.moveaxis(v_tensor, q, 0)
            
            A_00 = -0.5 * s
            A_01 = 0.5 * c
            A_10 = -0.5 * c
            A_11 = -0.5 * s
            
            new_v = np.empty_like(v_tensor)
            new_v[0] = A_00 * v_tensor[0] + A_01 * v_tensor[1]
            new_v[1] = A_10 * v_tensor[0] + A_11 * v_tensor[1]
            
            v_tensor = np.moveaxis(new_v, 0, q)
            v_after_adj = v_tensor.reshape(dim)
            
            g = 2.0 * np.vdot(psi_before, v_after_adj).real
            grad[l, q] = g
            
            # Propagate v backwards: v_before = U^dagger v_after
            # U^dagger = [[c, s], [-s, c]]
            v_tensor_orig = v.reshape([2] * n)
            v_tensor_orig = np.moveaxis(v_tensor_orig, q, 0)
            
            new_v_prop = np.empty_like(v_tensor_orig)
            new_v_prop[0] = c * v_tensor_orig[0] + s * v_tensor_orig[1]
            new_v_prop[1] = -s * v_tensor_orig[0] + c * v_tensor_orig[1]
            
            v_tensor_orig = np.moveaxis(new_v_prop, 0, q)
            v = v_tensor_orig.reshape(dim)
            
        elif op[0] == 'cz':
            q = op[1]
            mask1 = 1 << q
            mask2 = 1 << (q + 1)
            indices = np.arange(dim)
            both_one = (indices & mask1) != 0
            both_one &= (indices & mask2) != 0
            v[both_one] *= -1
    
    return energy - E0, grad

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0
    
    E0 = _ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    best_result = None
    best_energy = np.inf
    
    # Closed-form initial guess: solve single-qubit effective field equations
    # For each qubit, the effective transverse field and longitudinal field
    # determine the optimal Ry angle. The CZ layers are diagonal, so they
    # only modify the effective longitudinal field via the correlation terms.
    
    # Heuristic closed-form: for each layer and qubit, set theta to align
    # the local magnetization with the local effective field direction.
    # The effective field at qubit i in layer l depends on the fields and
    # couplings. A reasonable initial guess is to set all angles based on
    # the ratio of transverse to longitudinal effective fields.
    
    # For the transverse-field Ising model, the ground state has a certain
    # magnetization pattern. We use a simple closed-form based on the
    # mean-field approximation: theta_i = arctan(h_eff_i / J_eff_i)
    
    # Compute effective fields using mean-field
    # For open chain with n qubits, the effective longitudinal field on qubit i
    # comes from its neighbors' magnetizations times the coupling.
    
    # Simple closed-form initialization:
    # In the strong coupling limit, spins align, so Ry angles should be small.
    # In the strong field limit, spins are transverse, so Ry angles ~ pi/2.
    
    # Use a parameterization that interpolates:
    # theta_i = arctan(field / (coupling * n_neighbors))
    
    n_neighbors = np.zeros(n)
    for i in range(n):
        if i > 0:
            n_neighbors[i] += 1
        if i < n - 1:
            n_neighbors[i] += 1
    
    # For qubits with no neighbors (n=1 case, but we have n>=4), handle edge case
    n_neighbors[n_neighbors == 0] = 1  # Avoid division by zero
    
    # Mean-field effective longitudinal field
    J_eff = coupling * n_neighbors
    h_eff = field
    
    # Optimal single-qubit angle: theta = 2 * arctan(h_eff / J_eff)
    # This comes from diagonalizing the single-qubit effective Hamiltonian
    # H_eff = -J_eff * Z - h_eff * X
    # The ground state is |theta> = cos(theta/2)|0> + sin(theta/2)|1>
    # where tan(theta/2) = h_eff / J_eff
    
    theta_single = 2.0 * np.arctan2(h_eff, J_eff)
    
    # For the 3-layer circuit, distribute the rotation across layers
    # and account for the fact that CZ gates modify the effective field.
    # A reasonable distribution: put most rotation in the first layer,
    # some in the second, and fine-tune in the third.
    
    closed_form = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            # Distribute the total rotation across layers
            # Layer 0: 50%, Layer 1: 30%, Layer 2: 20%
            fractions = [0.5, 0.3, 0.2]
            closed_form[l, q] = fractions[l] * theta_single[q]
    
    # Now use this as the primary initial guess, plus a few random starts
    n_restarts = 15
    rng = np.random.default_rng(42)
    
    # First try the closed-form solution
    init_list = [closed_form.flatten().copy()]
    
    # Add random starts
    for i in range(n_restarts - 1):
        init_list.append(rng.uniform(-np.pi, np.pi, (layers, n)).flatten())
    
    for init_flat in init_list:
        if time.time() > deadline - 5.0:
            break
        
        def objective(flat_params):
            if time.time() > deadline:
                return 1e10
            params = flat_params.reshape((layers, n))
            try:
                e, _ = _energy_and_grad(params, n, layers, H, E0)
                return e
            except Exception:
                return 1e10
        
        def grad_func(flat_params):
            if time.time() > deadline:
                return np.zeros_like(flat_params)
            params = flat_params.reshape((layers, n))
            try:
                _, g = _energy_and_grad(params, n, layers, H, E0)
                return g.flatten()
            except Exception:
                return np.zeros_like(flat_params)
        
        try:
            res = minimize(
                objective,
                init_flat,
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_result = res.x.copy()
                
        except Exception:
            pass
    
    if best_result is None:
        return closed_form
    
    return best_result.reshape((layers, n))
