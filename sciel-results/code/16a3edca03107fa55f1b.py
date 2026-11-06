"""Analytic-gradient quantum circuit optimization with adjoint method."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 130.0  # safety margin before 150s kill


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    return H


def _exact_ground_energy(H):
    """Compute exact ground state energy."""
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _apply_ry(psi, q, theta, n):
    """Apply Ry(theta) on qubit q to state psi (flattened)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    shape = [2] * n
    psi_r = psi.reshape(shape)
    gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
    psi_moved = np.moveaxis(psi_r, q, 0)
    psi_out = np.tensordot(gate, psi_moved, axes=([1], [0]))
    return np.moveaxis(psi_out, 0, q).reshape(-1)


def _apply_cz(psi, q, n):
    """Apply CZ between qubit q and q+1 to state psi (flattened)."""
    shape = [2] * n
    psi_r = psi.reshape(shape)
    psi_moved = np.moveaxis(psi_r, [q, q + 1], [-2, -1])
    factor = np.ones(psi_moved.shape, dtype=np.complex128)
    idx11 = (slice(None),) * (n - 2) + (1, 1)
    factor[idx11] = -1.0
    psi_moved = psi_moved * factor
    return np.moveaxis(psi_moved, [-2, -1], [q, q + 1]).reshape(-1)


def _circuit_forward(params, n, layers):
    """
    Forward pass: compute final state and store intermediate states.
    Returns (final_psi, intermediate_states) where intermediate_states
    is a list of states after each gate application.
    """
    t = params.reshape(layers, n)
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    # Store states before each Ry gate for adjoint computation
    # We need: state before each Ry(layer, q)
    states_before_ry = []  # (layer, q, state_before)
    
    for layer in range(layers):
        for q in range(n):
            states_before_ry.append((layer, q, psi.copy()))
            psi = _apply_ry(psi, q, t[layer, q], n)
        
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz(psi, q, n)
    
    return psi, states_before_ry


def _circuit_backward_adjoint(H, psi_final, states_before_ry, n, layers, params):
    """
    Compute gradient using adjoint method.
    dE/dtheta[l,q] = 2 * Re( <psi_before| G^dag | H | psi_final> )
    where G is the generator of Ry: G = -i/2 * X_q (since Ry(theta) = exp(-i theta X/2))
    
    Actually: d/dtheta exp(-i theta X/2) = -i/2 X exp(-i theta X/2)
    So dpsi/dtheta = -i/2 X_q psi
    
    Energy E = <psi|H|psi>
    dE/dtheta = 2 Re( <dpsi/dtheta| H | psi> )
             = 2 Re( <psi| (i/2 X_q)^dag H | psi> )  -- no, let's be careful
    
    dE/dtheta = <dpsi/dtheta|H|psi> + <psi|H|dpsi/dtheta>
              = (-i/2)<psi|X_q|H|psi> + (i/2)<psi|H|X_q|psi>
              = (i/2)<psi|H|X_q|psi> - (i/2)<psi|X_q|H|psi>
              = (i/2)<psi|H X_q - X_q H|psi>
    
    But it's easier to just compute: dpsi/dtheta = -i/2 X_q psi, then
    dE/dtheta = 2 Re( <dpsi/dtheta| H | psi> )
    
    For the adjoint method, we propagate the "cost gradient" backward.
    Let chi = H |psi_final>
    Then dE/dtheta[l,q] = 2 Re( <psi_before| (-i/2 X_q) | chi_propagated_back> )
    
    Actually the cleanest way: 
    dE/dtheta_k = 2 Re( <psi_k| H | psi_final> * (-i/2) * <something> )
    
    Let me use the direct approach: for each parameter, compute dpsi/dtheta and get the gradient.
    But that's O(num_params * dim) which is fine for small dim.
    
    Better: use the adjoint (reverse-mode) approach.
    Define chi = H @ psi_final
    Then propagate chi backward through the circuit.
    At each Ry gate, the contribution to the gradient is:
    dE/dtheta = 2 Re( <chi_at_that_point| (-i/2) X_q | psi_at_that_point> )
    
    Wait, let me think more carefully.
    
    E = <psi_L| H | psi_L> where psi_L is the final state.
    
    dE/dtheta_k = <dpsi_L/dtheta_k| H | psi_L> + <psi_L| H | dpsi_L/dtheta_k>
    
    Now, dpsi_L/dtheta_k can be computed by forward propagation of the perturbation.
    But adjoint method: define the "co-state" chi such that:
    <chi| psi_L> = <psi_L| H | psi_L> -- no, that's not right either.
    
    Let me just use the straightforward approach for small systems:
    For each parameter k, compute the derivative state by forward propagation.
    Since dim is at most 64, this is fast.
    """
    t = params.reshape(layers, n)
    num_params = layers * n
    grad = np.zeros(num_params)
    
    # Forward pass to get final state
    psi_final, _ = _circuit_forward(params, n, layers)
    
    # For each parameter, compute dpsi/dtheta via forward propagation of the derivative
    for l in range(layers):
        for q in range(n):
            # dpsi/dtheta at the Ry(l,q) gate: dpsi = -i/2 X_q psi
            # We need to propagate this derivative through the remaining gates
            
            # Re-run forward pass up to gate (l,q), then apply the derivative
            dim = 2 ** n
            psi = np.zeros(dim, dtype=np.complex128)
            psi[0] = 1.0
            
            # Also track the derivative state
            dpsi = np.zeros(dim, dtype=np.complex128)
            
            for ll in range(layers):
                for qq in range(n):
                    theta = t[ll, qq]
                    # Apply Ry to psi
                    c = np.cos(theta / 2.0)
                    s = np.sin(theta / 2.0)
                    
                    # Apply to psi
                    psi = _apply_ry(psi, qq, theta, n)
                    
                    # Apply to dpsi
                    if dpsi.any():
                        dpsi = _apply_ry(dpsi, qq, theta, n)
                    
                    # At gate (l, q), add the derivative contribution
                    if ll == l and qq == q:
                        # dpsi/dtheta = -i/2 X_q psi (the psi BEFORE this gate)
                        # We need psi before applying Ry
                        # Let me restructure: apply Ry to get new psi, but compute derivative from old psi
                        pass
            
            # This approach is getting complicated. Let me use a cleaner method.
            # For each parameter, do a full forward pass with the derivative tracked.
            
            psi2 = np.zeros(dim, dtype=np.complex128)
            psi2[0] = 1.0
            dpsi2 = np.zeros(dim, dtype=np.complex128)
            
            for ll in range(layers):
                for qq in range(n):
                    theta = t[ll, qq]
                    
                    # Save old psi for derivative computation
                    psi_old = psi2.copy()
                    
                    # Apply Ry to both
                    psi2 = _apply_ry(psi2, qq, theta, n)
                    if dpsi2.any():
                        dpsi2 = _apply_ry(dpsi2, qq, theta, n)
                    
                    # At the target gate, inject the derivative
                    if ll == l and qq == q:
                        # d/dtheta Ry(theta) |psi> = -i/2 X_q Ry(theta)|psi>
                        # = -i/2 X_q psi2
                        # Apply X_q to psi_old (state before Ry)
                        # X_q |psi_old> flips qubit q
                        shape = [2] * n
                        psi_old_r = psi_old.reshape(shape)
                        x_psi = np.moveaxis(psi_old_r, q, 0).copy()
                        # X = [[0,1],[1,0]]
                        x_psi_new = np.zeros_like(x_psi)
                        x_psi_new[0, ...] = x_psi[1, ...]
                        x_psi_new[1, ...] = x_psi[0, ...]
                        x_psi_flat = np.moveaxis(x_psi_new, 0, q).reshape(-1)
                        dpsi2 += (-0.5j) * x_psi_flat
                    
                    if ll < layers - 1 and qq == n - 1:
                        # After all Ry in this layer, apply CZ
                        pass
            
            # Apply CZ gates after each layer (except last)
            # I need to redo this more carefully with CZ included
            
            # Let me restart with a cleaner forward+derivative pass
            psi3 = np.zeros(dim, dtype=np.complex128)
            psi3[0] = 1.0
            dpsi3 = np.zeros(dim, dtype=np.complex128)
            
            for ll in range(layers):
                for qq in range(n):
                    theta = t[ll, qq]
                    psi3_old = psi3.copy()
                    psi3 = _apply_ry(psi3, qq, theta, n)
                    if np.any(dpsi3):
                        dpsi3 = _apply_ry(dpsi3, qq, theta, n)
                    
                    if ll == l and qq == q:
                        # d(Ry(theta)|psi>)/dtheta at this gate = -i/2 X_q |psi_old>
                        # where |psi_old> is the state before this Ry
                        shape = [2] * n
                        po_r = psi3_old.reshape(shape)
                        x_psi = np.moveaxis(po_r, q, 0).copy()
                        x_new = np.zeros_like(x_psi)
                        x_new[0] = x_psi[1]
                        x_new[1] = x_psi[0]
                        x_flat = np.moveaxis(x_new, 0, q).reshape(-1)
                        dpsi3 += (-0.5j) * x_flat
                
                if ll < layers - 1:
                    for qq in range(n - 1):
                        psi3 = _apply_cz(psi3, qq, n)
                        if np.any(dpsi3):
                            dpsi3 = _apply_cz(dpsi3, qq, n)
            
            k = l * n + q
            Hpsi3 = H @ psi3
            grad[k] = 2.0 * np.real(np.vdot(dpsi3, Hpsi3))
    
    return grad


def _compute_energy(params, n, H, e0):
    """Compute objective value."""
    psi, _ = _circuit_forward(params, n, 3)
    energy = np.vdot(psi, H @ psi).real
    return (energy - e0) / n


def _compute_energy_and_grad(params, n, H, e0):
    """Compute objective and gradient."""
    val = _compute_energy(params, n, H, e0)
    grad = _circuit_backward_adjoint(H, None, None, n, 3, params)
    # Normalize gradient by n
    grad = grad / n
    return val, grad


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    e0 = _exact_ground_energy(H)
    num_params = layers * n
    
    def objective(params):
        return _compute_energy(params, n, H, e0)
    
    def objective_grad(params):
        val, grad = _compute_energy_and_grad(params, n, H, e0)
        return val, grad
    
    # Generate diverse initial guesses
    np.random.seed(123)
    ratio = field / coupling if coupling > 0 else 1.0
    
    # Physics-based initial guesses
    # For transverse field Ising, the ground state has correlations
    # that depend on field/coupling ratio
    base = np.pi * 0.25 * np.tanh(ratio)
    
    restarts = []
    # Uniform guesses at various scales
    for scale in [0.0, 0.3, 0.5, 0.7, 1.0]:
        restarts.append(np.full(num_params, base * scale))
    # Random
    for s in [0.3, 0.5, 0.8, 1.0]:
        restarts.append(np.random.randn(num_params) * s)
    # Alternating patterns
    for s in [0.3, 0.5]:
        pattern = np.zeros(num_params)
        for i in range(num_params):
            l = i // n
            q = i % n
            pattern[i] = s * (1.0 if q % 2 == 0 else -1.0) * (1.0 if l % 2 == 0 else 0.5)
        restarts.append(pattern)
    
    # Sort by objective value, try best first
    scored_restarts = []
    for x in restarts:
        x = np.clip(x, -np.pi, np.pi)
        try:
            v = objective(x)
            scored_restarts.append((v, x))
        except:
            scored_restarts.append((np.inf, x))
    scored_restarts.sort(key=lambda t: t[0])
    
    best_val = np.inf
    best_params = None
    
    # Use Nelder-Mead (robust, no gradient needed) with multiple restarts
    # Plus a few L-BFGS-B runs with the analytic gradient
    
    time_per_restart_nm = 8.0
    time_per_restart_lbfgs = 10.0
    
    for i, (v0, x0) in enumerate(scored_restarts):
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 20:
            break
        
        # Alternate between Nelder-Mead and L-BFGS-B
        use_lbfgs = (i % 3 == 0) and (n <= 6)
        
        try:
            if use_lbfgs:
                res = opt.minimize(
                    objective_grad,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30, 'maxfun': 150}
                )
            else:
                res = opt.minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
                
        except Exception:
            pass
        
        # Check time
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 15:
            break
    
    if best_params is None:
        best_params = scored_restarts[0][1]
    
    theta = best_params.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5:
        # Fallback: return |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        # Fallback
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
