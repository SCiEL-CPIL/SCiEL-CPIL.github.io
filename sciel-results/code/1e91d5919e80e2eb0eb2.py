import numpy as np
import time
from scipy.optimize import minimize

def _build_idx_cache(n):
    """Precompute index arrays for fast gate application."""
    dim = 1 << n
    ry_cache = []
    for q in range(n):
        mask = 1 << q
        idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int32)
        idx1 = idx0 | mask
        ry_cache.append((idx0, idx1))
    
    cz_cache = []
    for q in range(n - 1):
        mask1 = 1 << q
        mask2 = 1 << (q + 1)
        idx_both = np.array([i for i in range(dim) if (i & mask1) and (i & mask2)], dtype=np.int32)
        cz_cache.append(idx_both)
        
    return {'ry': ry_cache, 'cz': cz_cache}

def _hamiltonian_kron(n, coupling, field):
    """Build Hamiltonian using Kronecker products."""
    dim = 1 << n
    I = np.eye(2)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        term = np.array([1.0])
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
    for i in range(n):
        term = np.array([1.0])
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
    return H

def _circuit_state(n, layers, params, idx_cache):
    """Compute state from circuit using precomputed indices."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = idx_cache['ry'][q]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if layer < layers - 1:
            for q in range(n - 1):
                idx_both = idx_cache['cz'][q]
                state[idx_both] = -state[idx_both]
    
    return state

def _circuit_state_and_grads(n, layers, params, idx_cache):
    """Compute state and analytical gradients via backpropagation (adjoint method).
    
    Returns (state, grads) where grads has shape (layers, n).
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # Store intermediate states for backprop
    # After each Ry gate on qubit q at layer l, store the pre-gate amplitudes
    # We need: for each (layer, q), the state just before the Ry gate
    pre_states = []  # list of (layer, q, idx0_vals, idx1_vals)
    
    for layer in range(layers):
        for q in range(n):
            idx0, idx1 = idx_cache['ry'][q]
            # Store pre-gate values
            pre0 = state[idx0].copy()
            pre1 = state[idx1].copy()
            pre_states.append((layer, q, pre0, pre1))
            
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            state[idx0] = c * pre0 - s * pre1
            state[idx1] = s * pre0 + c * pre1
        
        if layer < layers - 1:
            for q in range(n - 1):
                idx_both = idx_cache['cz'][q]
                state[idx_both] = -state[idx_both]
    
    return state, pre_states

def _hamiltonian(n, coupling, field):
    """Build Hamiltonian efficiently."""
    return _hamiltonian_kron(n, coupling, field)

def _energy_and_grad_analytic(n, layers, params, idx_cache, H, e_ground):
    """Compute energy and exact gradient via backpropagation.
    
    The circuit is a sequence of unitary gates U = U_L ... U_2 U_1.
    Energy E = <psi|H|psi> where |psi> = U|0>.
    dE/dtheta = 2 Re[<psi| dU/dtheta U† |psi> ... ] -- actually:
    dE/dtheta = 2 Re[ <psi| H |dpsi/dtheta> ]
    
    We use the adjoint method:
    |psi> = U_total |0>
    d|psi>/dtheta_k = U_total ... U_{k+1} (dU_k/dtheta) U_k^{-1} ... U_1 |0>
    
    More efficiently, we propagate the adjoint backward.
    """
    dim = 1 << n
    state, pre_states = _circuit_state_and_grads(n, layers, params, idx_cache)
    energy = np.vdot(state, H @ state).real
    obj_val = (energy - e_ground) / n
    
    # Gradient: dE/dtheta_{l,q} = 2 Re[ <psi| H |dpsi/dtheta_{l,q}> ]
    # |dpsi/dtheta> for Ry gate at (l,q) with angle theta:
    # dRy/dtheta = (1/2) * Ry'(theta) where Ry'(theta) = -0.5*(sin(theta/2) I + cos(theta/2) sigma_x) per qubit
    # Actually: d/dtheta Ry(theta) = -1/2 * sin(theta/2) * |0><1| - 1/2 * cos(theta/2) * |1><0| ... 
    # Let me be more careful.
    # Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
    # dRy/dtheta = 1/2 * [[-sin(t/2), -cos(t/2)], [cos(t/2), -sin(t/2)]]
    #             = 1/2 * [[-s, -c], [c, -s]] where s=sin(t/2), c=cos(t/2)
    #
    # For a gate on qubit q: dU_q/dtheta acts on the (0,1) block of qubit q.
    # If |psi_pre> is the state before the gate, then:
    # |psi_post> = U_q |psi_pre>
    # d|psi_post>/dtheta = (dU_q/dtheta) |psi_pre>
    #
    # The contribution to d|psi_final>/dtheta involves propagating this forward through remaining gates.
    # 
    # Adjoint method: Let |chi> = H|psi_final> (or more precisely, we need <psi|H|dpsi/dtheta>)
    # Actually: dE/dtheta = 2 Re[ <psi| H |dpsi/dtheta> ]
    #           = 2 Re[ <dpsi/dtheta| H |psi> ]  (since H is Hermitian and these are complex)
    #
    # Using adjoint: define |mu> such that <psi|H|dpsi/dtheta> = <mu|dpsi/dtheta>
    # where |mu> = H|psi> ... no wait.
    # <psi|H|dpsi/dtheta> = (<psi|H)|dpsi/dtheta> = <H†psi|dpsi/dtheta> = <H psi|dpsi/dtheta>
    # So |mu> = H|psi>, and we need <mu|dpsi/dtheta> for each theta.
    #
    # But |dpsi/dtheta> = U_remaining (dU_k/dtheta) |psi_before_k>
    # So <mu|dpsi/dtheta> = <U_remaining† mu| (dU_k/dtheta) |psi_before_k>
    #
    # This requires propagating |mu> backward through U_remaining†.
    #
    # Let's do it step by step. We'll propagate the adjoint state backward.
    
    # |mu> = H |psi>
    mu = H @ state  # this is H|psi>
    # We need <mu| dpsi/dtheta> = vdot(mu, dpsi/dtheta) for each theta
    
    # Propagate mu backward through the circuit to get the adjoint at each gate
    # The circuit is: [Ry layer 0] [CZ] [Ry layer 1] [CZ] [Ry layer 2]
    # Forward: |0> -> Ry_0 -> CZ_0 -> Ry_1 -> CZ_1 -> Ry_2 -> |psi>
    # Backward: |psi> -> Ry_2† -> CZ_1† -> Ry_1† -> CZ_0† -> Ry_0† -> |0>
    
    # CZ is self-inverse (Hermitian and unitary), so CZ† = CZ
    # Ry†(theta) = Ry(-theta)
    
    # We need to compute, for each gate k:
    # <mu_at_k| dU_k/dtheta |psi_before_k>
    # where mu_at_k is the adjoint state just after gate k (before propagating further back)
    
    # Let's propagate mu backward and compute contributions at each Ry gate.
    
    grad = np.zeros((layers, n), dtype=np.float64)
    
    # Start with mu = H|psi>
    # We'll go backward through the gates in reverse order.
    
    # Reverse order of gates:
    # Last gate: Ry at (layer=2, q) for q=n-1, ..., q=0
    # Then CZ at (n-2), ..., 0  (if layers > 2, i.e., layer < layers-1)
    # Then Ry at (layer=1, q) for q=n-1, ..., q=0
    # Then CZ at (n-2), ..., 0
    # Then Ry at (layer=0, q) for q=n-1, ..., q=0
    
    # Actually, the forward order within a layer is q=0,1,...,n-1.
    # So backward is q=n-1,...,0.
    
    mu_cur = mu.copy()
    
    # Build list of gates in reverse order
    # Forward gates (in order):
    # For layer in range(layers):
    #   For q in range(n): Ry(layer, q)
    #   If layer < layers-1:
    #     For q in range(n-1): CZ(q)
    
    # Reverse:
    # For layer in range(layers-1, -1, -1):
    #   If layer < layers-1 (i.e., layer <= layers-2):
    #     For q in range(n-2, -1, -1): CZ(q)  -- CZ is its own inverse
    #   For q in range(n-1, -1, -1): Ry(layer, q)†
    
    # Let's iterate backward and track mu.
    # At each Ry gate (layer l, qubit q), before applying Ry† to mu,
    # we compute the gradient contribution.
    
    # The gradient contribution for Ry at (l, q):
    # dU_q/dtheta |psi_pre> where |psi_pre> is the state before this gate.
    # We have pre_states stored.
    # <mu_at_this_point| dU_q/dtheta |psi_pre>
    
    # mu_at_this_point is mu_cur just before we apply Ry†(theta) to propagate back.
    # After Ry: |psi_post> = Ry |psi_pre>
    # mu_before_Ry = Ry† mu_after_Ry  (this is how we propagate back)
    # The contribution is <mu_after_Ry| dRy/dtheta |psi_pre>
    
    # dRy/dtheta = 1/2 [[-s, -c], [c, -s]] where s=sin(theta/2), c=cos(theta/2)
    # Applied to |psi_pre> = a|0> + b|1> on qubit q:
    # (dRy/dtheta)|psi_pre> = 1/2 [(-s*a - c*b)|0> + (c*a - s*b)|1>]
    
    # So <mu| dRy/dtheta |psi_pre> = 1/2 [ conj(mu_0)*(-s*a - c*b) + conj(mu_1)*(c*a - s*b) ]
    # where a,b are the pre-gate amplitudes and mu_0, mu_1 are the post-gate (forward) 
    # adjoint amplitudes on qubit q.
    
    # Let me re-index: pre_states stores (layer, q, pre0, pre1) in forward order.
    # I need to match them up with the backward propagation.
    
    # Let me redo this more carefully by propagating mu backward and computing 
    # the contribution at each Ry gate.
    
    # Forward gate order (index 0 to total-1):
    # gate 0: Ry(0,0), gate 1: Ry(0,1), ..., gate n-1: Ry(0,n-1)
    # gate n: CZ(0), gate n+1: CZ(1), ..., gate 2n-2: CZ(n-2)  [if layers>1]
    # gate 2n-1: Ry(1,0), ...
    
    # Let me just iterate through layers in reverse and compute.
    
    # I'll store pre_states in a dict for easy lookup
    pre_dict = {}
    for (l, q, p0, p1) in pre_states:
        pre_dict[(l, q)] = (p0, p1)
    
    # Propagate mu backward
    # mu currently = H|psi_final>
    
    for l in range(layers - 1, -1, -1):
        # First, undo the CZ gates (if any) that came after this layer's Ry gates
        # CZ gates are between layer l and l+1, so they come after Ry layer l in forward
        # In backward, we undo CZ before undoing Ry layer l
        if l < layers - 1:
            for q in range(n - 2, -1, -1):
                idx_both = idx_cache['cz'][q]
                mu_cur[idx_both] = -mu_cur[idx_both]
        
        # Now undo Ry gates for layer l, from q=n-1 down to q=0
        for q in range(n - 1, -1, -1):
            idx0, idx1 = idx_cache['ry'][q]
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # Compute gradient contribution BEFORE updating mu
            # mu_cur is the adjoint just after this Ry gate (in forward direction)
            # i.e., mu_cur = Ry† mu_further_back ... no wait.
            # 
            # Let me think again. In the forward direction:
            # |psi_after> = Ry |psi_before>
            # In the adjoint/backward direction:
            # mu_before = Ry† mu_after
            # 
            # mu_cur currently holds mu_after (the adjoint after this gate in forward order)
            # We need <mu_after| dRy/dtheta |psi_before>
            
            mu0 = mu_cur[idx0]
            mu1 = mu_cur[idx1]
            pre0, pre1 = pre_dict[(l, q)]
            
            # dRy/dtheta |psi_pre> on the two-component subspace:
            # = 1/2 [(-s*pre0 - c*pre1), (c*pre0 - s*pre1)]
            
            # <mu| dRy/dtheta |psi_pre> = conj(mu0)*(-s*pre0 - c*pre1)/2 + conj(mu1)*(c*pre0 - s*pre1)/2
            contrib = np.conj(mu0) * (-s * pre0 - c * pre1) + np.conj(mu1) * (c * pre0 - s * pre1)
            contrib /= 2.0
            
            # dE/dtheta = 2 Re[<psi|H|dpsi/dtheta>] = 2 Re[<mu|dpsi/dtheta>]
            # where |mu> = H|psi> and we've propagated it back.
            # But wait -- the adjoint propagation gives us <mu_after| dpsi_after/dtheta>
            # which equals <mu_before| dpsi_before/dtheta> only if the remaining gates are unitary.
            # Actually, the full derivative |dpsi_final/dtheta> = U_remaining (dU_k/dtheta) |psi_before_k>
            # So <mu_final| dpsi_final/dtheta> = <U_remaining† mu_final| dU_k/dtheta |psi_before_k>
            # And U_remaining† mu_final is exactly what mu_cur represents at this point.
            # So contrib = <mu_cur| dRy/dtheta |psi_pre> is correct.
            
            grad[l, q] = 2.0 * np.real(contrib) / n  # divide by n for objective
            
            # Now propagate mu backward through this Ry gate
            # mu_before = Ry† mu_after = Ry(-theta) mu_after
            # Ry(-theta) = [[c, s], [-s, c]]
            new_mu0 = c * mu0 + s * mu1
            new_mu1 = -s * mu0 + c * mu1
            mu_cur[idx0] = new_mu0
            mu_cur[idx1] = new_mu1
    
    return obj_val, grad

def _mean_field_angle(n, coupling, field):
    """Compute mean-field optimal angle."""
    ratio = field / coupling
    s_target = ratio * n / (2 * (n - 1))
    if s_target > 1.0:
        return np.pi / 2
    elif s_target < -1.0:
        return -np.pi / 2
    else:
        return np.arcsin(s_target)

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Precompute
    idx_cache = _build_idx_cache(n)
    H = _hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p, idx_cache)
        energy = np.vdot(st, H @ st).real
        return (energy - e_ground) / n
    
    def obj_grad(xf):
        p = xf.reshape((layers, n))
        val, grad = _energy_and_grad_analytic(n, layers, p, idx_cache, H, e_ground)
        return val, grad.flatten()
    
    # Mean-field optimal angle
    base = _mean_field_angle(n, coupling, field)
    
    # Initial guesses
    guesses = []
    guesses.append(np.full((layers, n), base))
    guesses.append(np.full((layers, n), 0.0))
    guesses.append(np.full((layers, n), np.pi / 2))
    guesses.append(np.full((layers, n), -base))
    
    # Layer-dependent profiles
    for scale in [0.7, 1.0, 1.3]:
        g = np.zeros((layers, n))
        for l in range(layers):
            frac = 0.5 + 0.5 * l / max(layers - 1, 1)
            g[l, :] = base * frac * scale
        guesses.append(g)
    
    # Alternating sign
    g = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            g[l, q] = base * (1.0 if q % 2 == 0 else -1.0)
    guesses.append(g)
    
    # One random
    np.random.seed(42)
    g = np.random.uniform(-np.pi, np.pi, (layers, n))
    guesses.append(g)
    
    best_params = guesses[0].copy()
    best_val = obj(best_params.flatten())
    
    # Phase 1: Gradient-based L-BFGS-B from each guess (now fast with analytic grad)
    for g in guesses:
        if time.time() > deadline - 20:
            break
        x0 = g.flatten()
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Phase 2: High-precision polish from best
    if time.time() < deadline - 5:
        try:
            res = minimize(obj_grad, best_params.flatten(), method='L-BFGS-B', jac=True,
                         options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Phase 3: Final ultra-precision polish
    if time.time() < deadline - 2:
        try:
            res = minimize(obj_grad, best_params.flatten(), method='L-BFGS-B', jac=True,
                         options={'maxiter': 300, 'ftol': 1e-18, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Final check: make sure all values are finite
    if not np.all(np.isfinite(best_params)):
        best_params = np.full((layers, n), base)
    
    return best_params
