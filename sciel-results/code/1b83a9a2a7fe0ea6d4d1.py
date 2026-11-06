import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_circuit_indices_cache = {}
_gs_cache = {}
_mps_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms: -field * X_i
    for i in range(n):
        mask = 1 << i
        for state in range(dim):
            flipped = state ^ mask
            H[state, flipped] += -field
    _H_cache[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0

def _get_circuit_indices(n, layers):
    """Pre-compute index pairs for fast state evolution."""
    key = (n, layers)
    if key in _circuit_indices_cache:
        return _circuit_indices_cache[key]
    
    dim = 1 << n
    idx = np.arange(dim)
    
    # Pre-compute indices for Ry(q)
    ry_indices = []
    for q in range(n):
        mask = 1 << q
        mask_bit = (idx & mask) == 0
        idx_0 = idx[mask_bit]
        idx_1 = idx_0 + mask
        ry_indices.append((idx_0, idx_1))
        
    # Pre-compute indices for CZ(i, i+1)
    cz_indices = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        both_set = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        cz_indices.append(both_set)
        
    _circuit_indices_cache[key] = (ry_indices, cz_indices)
    return _circuit_indices_cache[key]

def _circuit_state(n, thetas, layers):
    """Fast circuit state computation using pre-computed indices."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    
    ry_indices, cz_indices = _get_circuit_indices(n, layers)
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0, idx_1 = ry_indices[q]
            
            v0 = state[idx_0].copy()
            v1 = state[idx_1].copy()
            
            state[idx_0] = c * v0 - s * v1
            state[idx_1] = s * v0 + c * v1
        
        # Apply CZ between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                mask_idx = cz_indices[i]
                state[mask_idx] *= -1.0
    
    return state

def _circuit_state_and_grad(n, thetas, layers):
    """Compute circuit state and its gradient with respect to thetas.
    
    Returns:
        state: complex array of shape (dim,)
        grads: list of complex arrays, one per theta parameter, 
               where grads[l*n+q] = d|psi>/d(theta[l,q])
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    
    ry_indices, cz_indices = _get_circuit_indices(n, layers)
    
    # grads[l*n+q] will hold d|psi>/d(theta[l,q])
    grads = [np.zeros(dim, dtype=np.complex128) for _ in range(layers * n)]
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0, idx_1 = ry_indices[q]
            
            v0 = state[idx_0].copy()
            v1 = state[idx_1].copy()
            
            # New state after Ry
            new_v0 = c * v0 - s * v1
            new_v1 = s * v0 + c * v1
            
            state[idx_0] = new_v0
            state[idx_1] = new_v1
            
            # Gradient with respect to theta
            # d/dtheta of Ry(theta) = 0.5 * [[-sin(theta/2), -cos(theta/2)],
            #                                  [cos(theta/2), -sin(theta/2)]]
            dc = -0.5 * s  # d(cos(theta/2))/dtheta
            ds = 0.5 * c   # d(sin(theta/2))/dtheta
            
            grad_v0 = dc * v0 - ds * v1
            grad_v1 = ds * v0 + dc * v1
            
            g_idx = layer * n + q
            grads[g_idx][idx_0] = grad_v0
            grads[g_idx][idx_1] = grad_v1
        
        # Apply CZ between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                mask_idx = cz_indices[i]
                state[mask_idx] *= -1.0
                # CZ is diagonal, so it just multiplies the gradient by -1 too
                for g in grads:
                    g[mask_idx] *= -1.0
    
    return state, grads

def _get_ground_state(n, coupling, field):
    """Get the exact ground state vector."""
    key = (n, coupling, field)
    if key in _gs_cache:
        return _gs_cache[key]
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    gs = eigvecs[:, 0]
    # Ensure real if possible
    if np.max(np.abs(gs.imag)) < 1e-10:
        gs = gs.real
    _gs_cache[key] = gs
    return gs

def _fit_mps_to_ground_state(n, coupling, field, bond_dim=4):
    """Fit an MPS to the ground state using variational approach."""
    key = (n, coupling, field, bond_dim)
    if key in _mps_cache:
        return _mps_cache[key]
    
    gs = _get_ground_state(n, coupling, field)
    gs = gs.reshape([2] * n)
    
    # Initialize MPS tensors
    # Tensor shape: [bond_left, phys, bond_right]
    tensors = []
    for i in range(n):
        if i == 0:
            t = np.random.randn(1, 2, bond_dim) * 0.1 + 1.0
        elif i == n - 1:
            t = np.random.randn(bond_dim, 2, 1) * 0.1 + 1.0
        else:
            t = np.random.randn(bond_dim, 2, bond_dim) * 0.1 + 1.0
        tensors.append(t)
    
    def _mps_state(tensors):
        """Compute MPS state from tensors."""
        # Contract tensors: A0 @ A1 @ ... @ An-1
        result = tensors[0]
        for t in tensors[1:]:
            result = np.tensordot(result, t, axes=([2], [0]))
            result = result.transpose(0, 1, 2)  # [phys, phys, bond]
        # Flatten to state vector
        state = result.reshape(2 ** n)
        norm = np.linalg.norm(state)
        if norm > 1e-15:
            state = state / norm
        return state
    
    def _objective_mps(params):
        """MPS parameter objective: maximize overlap with ground state."""
        # Reconstruct tensors from flat params
        new_tensors = []
        idx = 0
        for i in range(n):
            if i == 0:
                size = 1 * 2 * bond_dim
            elif i == n - 1:
                size = bond_dim * 2 * 1
            else:
                size = bond_dim * 2 * bond_dim
            t = params[idx:idx + size].reshape(
                (1 if i == 0 else bond_dim, 2, bond_dim if i < n - 1 else 1)
            )
            new_tensors.append(t)
            idx += size
        
        mps = _mps_state(new_tensors)
        overlap = np.abs(np.vdot(gs, mps))
        return 1.0 - overlap  # Minimize 1 - overlap
    
    # Flatten initial tensors
    init_params = np.concatenate([t.flatten() for t in tensors])
    
    # Optimize
    result = minimize(_objective_mps, init_params, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12})
    
    # Extract best MPS
    best_params = result.x
    best_tensors = []
    idx = 0
    for i in range(n):
        if i == 0:
            size = 1 * 2 * bond_dim
        elif i == n - 1:
            size = bond_dim * 2 * 1
        else:
            size = bond_dim * 2 * bond_dim
        t = best_params[idx:idx + size].reshape(
            (1 if i == 0 else bond_dim, 2, bond_dim if i < n - 1 else 1)
        )
        best_tensors.append(t)
        idx += size
    
    _mps_cache[key] = best_tensors
    return best_tensors

def _angles_from_mps(mps_tensors, n, layers=3):
    """Extract approximate Ry angles from MPS tensors."""
    thetas = np.zeros((layers, n))
    
    for i in range(n):
        t = mps_tensors[i]
        # For single qubit, the MPS tensor encodes the local state
        # Extract effective rotation angle from the tensor
        if i == 0:
            # Shape: [1, 2, bond_dim]
            # Take the first bond index
            local = t[0]  # [2, bond_dim]
        elif i == n - 1:
            # Shape: [bond_dim, 2, 1]
            local = t[:, :, 0]  # [bond_dim, 2]
        else:
            # Shape: [bond_dim, 2, bond_dim]
            # Average over bond indices
            local = np.mean(t, axis=(0, 2))  # [2]
        
        # Extract amplitudes
        a0 = np.abs(local[0])
        a1 = np.abs(local[1])
        total = a0 + a1
        if total > 1e-15:
            p0 = a0 / total
            p1 = a1 / total
            # For Ry(theta)|0>, P(0) = cos^2(theta/2), P(1) = sin^2(theta/2)
            # So cos^2(theta/2) = p0
            cos2 = p0
            cos2 = np.clip(cos2, 0, 1)
            theta = 2.0 * np.arccos(np.sqrt(cos2))
            # Assign to all layers (simple initialization)
            for l in range(layers):
                thetas[l, i] = theta
    
    return thetas

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state(n, thetas, layers)
    return np.real(np.vdot(psi, H @ psi))

def _circuit_energy_and_grad(n, coupling, field, thetas, layers):
    """Compute circuit energy and its gradient with respect to thetas."""
    H = _build_H(n, coupling, field)
    psi, grads = _circuit_state_and_grad(n, thetas, layers)
    
    # Energy = <psi|H|psi>
    Hpsi = H @ psi
    energy = np.real(np.vdot(psi, Hpsi))
    
    # dE/dtheta_k = 2 * Re(<dpsi_k/dtheta_k|H|psi>)
    # = 2 * Re(<grad_k|Hpsi>)
    # But we need to be careful: <psi|H|psi> is real, and 
    # d/dtheta <psi|H|psi> = <dpsi/dtheta|H|psi> + <psi|H|dpsi/dtheta>
    # = 2*Re(<dpsi/dtheta|H|psi>)
    
    energy_grad = np.zeros(layers * n)
    for k in range(layers * n):
        gk = grads[k]
        # <gk|Hpsi> = vdot(gk, Hpsi) = conj(gk)^T Hpsi
        val = np.vdot(gk, Hpsi)
        energy_grad[k] = 2.0 * np.real(val)
    
    return energy, energy_grad

def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _objective_and_grad(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit, e_grad = _circuit_energy_and_grad(n, coupling, field, thetas, layers)
    obj_val = (e_circuit - e0) / n
    obj_grad = e_grad / n
    return obj_val, obj_grad

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    e0 = _exact_ground_energy(n, coupling, field)
    
    def obj(p):
        return _objective(p, n, coupling, field, layers, e0)
    
    def obj_and_grad(p):
        return _objective_and_grad(p, n, coupling, field, layers, e0)

    best_x = np.zeros(layers * n)
    best_val = obj(best_x)
    
    # Generate initial points
    inits = []
    
    # 1. Zero (Ferromagnetic limit)
    inits.append(np.zeros(layers * n))
    
    # 2. Pi/2 (Paramagnetic limit)
    inits.append(np.full(layers * n, np.pi/2))
    
    # 3. Physics-informed mean-field estimate
    # tan(theta/2) ~ field/coupling
    ratio = field / coupling if coupling > 0 else 1.0
    theta_mf = 2.0 * np.arctan(ratio)
    inits.append(np.full(layers * n, theta_mf))
    
    # 4. Alternating pattern (Antiferromagnetic hint)
    alt = np.zeros(layers * n).reshape(layers, n)
    for l in range(layers):
        for q in range(n):
            if q % 2 == 0:
                alt[l, q] = theta_mf
            else:
                alt[l, q] = -theta_mf
    inits.append(alt.flatten())
    
    # 5. MPS-informed initialization
    try:
        if time.time() < deadline - 30:
            mps_tensors = _fit_mps_to_ground_state(n, coupling, field, bond_dim=4)
            thetas_mps = _angles_from_mps(mps_tensors, n, layers)
            inits.append(thetas_mps.flatten())
            
            # Also try with different bond dimension
            if time.time() < deadline - 20:
                mps_tensors_bd6 = _fit_mps_to_ground_state(n, coupling, field, bond_dim=6)
                thetas_mps6 = _angles_from_mps(mps_tensors_bd6, n, layers)
                inits.append(thetas_mps6.flatten())
    except Exception:
        pass
    
    # 6. Random restarts
    # Number of restarts depends on n
    num_random = 25 if n <= 4 else 15
    
    for i in range(num_random):
        if time.time() > deadline - 20:
            break
        # Mix of uniform and gaussian around physics-informed center
        if i % 2 == 0:
            x_init = np.random.uniform(-np.pi, np.pi, size=(layers * n))
        else:
            # Center around the mean-field estimate
            x_init = theta_mf + np.random.normal(0, 0.8, size=(layers * n))
        
        inits.append(x_init)
        
    # Sort inits by objective value to prioritize good starts
    init_objs = []
    for x_init in inits:
        if time.time() > deadline - 15:
            break
        try:
            v = obj(x_init)
            init_objs.append((v, x_init))
        except Exception:
            continue
    
    init_objs.sort(key=lambda x: x[0])
    
    # Run L-BFGS-B from each start with explicit gradients
    for v_init, x_init in init_objs:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(obj_and_grad, x_init, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13,
                                   'maxls': 50})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # Final polish with Nelder-Mead if time remains
    if time.time() < deadline - 10.0:
        try:
            res_nm = minimize(obj, best_x, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x.copy()
        except Exception:
            pass
            
    # One more L-BFGS-B polish with tighter tolerances
    if time.time() < deadline - 2.0:
        try:
            res = minimize(obj_and_grad, best_x, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape((layers, n))
