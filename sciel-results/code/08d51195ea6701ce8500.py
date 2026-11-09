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

def _circuit_state(n, layers, params, idx_cache):
    """Compute state from circuit using precomputed indices."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = idx_cache['ry'][q]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        # Apply CZ between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                idx_both = idx_cache['cz'][q]
                state[idx_both] = -state[idx_both]
    
    return state

def _hamiltonian_terms(n, coupling, field):
    """Precompute diagonal ZZ part and off-diagonal X part separately."""
    dim = 1 << n
    # ZZ diagonal: for each basis state, compute sum of -coupling * Z_i Z_{i+1}
    zz_diag = np.zeros(dim, dtype=np.float64)
    for b in range(dim):
        val = 0.0
        for i in range(n - 1):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zi = 1 - 2 * bi
            zj = 1 - 2 * bj
            val += -coupling * zi * zj
        zz_diag[b] = val
    
    # X terms: for each qubit i, H has -field * |b><b^(1<<i)| for all b
    x_terms = []
    for i in range(n):
        mask = 1 << i
        # For each b, the partner is b ^ mask
        # We'll store (b, partner) pairs where b < partner to avoid double counting
        # But actually we need all pairs for the matrix-vector product
        # Let's just store the mask
        x_terms.append(mask)
    
    return zz_diag, x_terms

def _energy_and_grad(n, layers, params, idx_cache, zz_diag, x_masks, coupling, field, e_ground):
    """Compute energy and gradient analytically."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # We need to track the state after each Ry gate for gradient computation
    # Energy = <psi|H|psi> = sum_b zz_diag[b] |psi_b|^2 - field * sum_i sum_b (psi_b^* psi_{b^mask_i} + psi_{b^mask_i}^* psi_b)
    # Actually H = ZZ_diag + sum_i (-field) * X_i
    # <H> = sum_b zz_diag[b] |psi[b]|^2 + sum_i (-field) * sum_b psi[b]^* psi[b^mask_i]
    # But X_i is Hermitian, so <X_i> = sum_b psi[b]^* psi[b^mask_i] (this is real since X is real and symmetric)
    # Actually <psi|X_i|psi> = sum_{b,b'} psi[b]^* X_i[b,b'] psi[b'] = sum_b psi[b]^* psi[b^mask_i]
    # This is real because summing over all b, and the terms pair up.
    
    # For gradient, we use: d/dtheta <psi|H|psi> = 2 Re[<dpsi/dtheta|H|psi>]
    # where dpsi/dtheta can be computed by differentiating the circuit.
    
    # Let's compute the state and also the "Jacobian" states for each parameter.
    # This is expensive but n is small.
    
    # Alternative: use finite differences for gradient, but that's 2*layers*n evaluations.
    # Better: compute analytically.
    
    # Let's store states after each Ry application for backpropagation.
    # Actually, let's just compute the state and use the parameter-shift rule or direct differentiation.
    
    # For a shallow circuit, let's compute the gradient by differentiating each gate.
    # The state is a product of gates. Let's use the "state at each step" approach.
    
    # Simpler approach: compute state, then for each parameter, compute the derivative state
    # by re-running the circuit with a perturbed parameter and using finite differences.
    # With layers*n parameters, that's 2*layers*n state computations.
    # For n=6, layers=3: 36 extra state computations per gradient eval.
    # Each state computation is O(dim * n * layers) ~ 64*6*3 = 1152 ops.
    # Total per gradient: 37 * 1152 ~ 42k ops. That's fast enough.
    
    # But let's do it analytically for speed.
    
    # Actually, let's just compute the state and use a trick:
    # For Ry(theta) on qubit q:
    #   |psi'> = Ry(theta) |psi>
    #   d/dtheta |psi'> = (1/2)(-sin(theta/2) I - i cos(theta/2) X) |psi>
    #                  = (1/2)(-sin(theta/2) |psi> - i cos(theta/2) X_q |psi>)
    
    # This gets complicated with the CZ gates in between.
    
    # Let's just use finite differences for now, but vectorized.
    
    # Compute the base state
    st = _circuit_state(n, layers, params, idx_cache)
    
    # Energy
    energy = np.sum(zz_diag * np.abs(st)**2).real
    for i, mask in enumerate(x_masks):
        # <X_i> = sum_b st[b]^* st[b^mask]
        # This equals 2 * Re[sum_{b < b^mask} st[b]^* st[b^mask]]
        # But we can just compute it directly
        # For each b, b^mask is the partner
        # sum_b st[b]^* st[b^mask] = sum over all b
        # This is real because it equals <psi|X_i|psi> and X_i is Hermitian
        x_val = np.sum(np.conj(st) * np.roll(st, shift=0, axis=0))  # This is wrong
        # Let me think again.
        # <psi|X_i|psi> = sum_{b} psi[b]^* (X_i psi)[b] = sum_b psi[b]^* psi[b^mask]
        # Let's compute this properly
        partner = np.array([b ^ mask for b in range(dim)])
        x_val = np.sum(np.conj(st) * st[partner])
        energy += -field * x_val.real
    
    energy = energy / n
    
    # Now compute gradient via finite differences
    grad = np.zeros((layers, n), dtype=np.float64)
    eps = 1e-7
    for layer in range(layers):
        for q in range(n):
            # +eps
            p_plus = params.copy()
            p_plus[layer, q] += eps
            st_plus = _circuit_state(n, layers, p_plus, idx_cache)
            e_plus = np.sum(zz_diag * np.abs(st_plus)**2).real
            for i, mask in enumerate(x_masks):
                partner = np.array([b ^ mask for b in range(dim)])
                x_val = np.sum(np.conj(st_plus) * st_plus[partner])
                e_plus += -field * x_val.real
            e_plus /= n
            
            # -eps
            p_minus = params.copy()
            p_minus[layer, q] -= eps
            st_minus = _circuit_state(n, layers, p_minus, idx_cache)
            e_minus = np.sum(zz_diag * np.abs(st_minus)**2).real
            for i, mask in enumerate(x_masks):
                partner = np.array([b ^ mask for b in range(dim)])
                x_val = np.sum(np.conj(st_minus) * st_minus[partner])
                e_minus += -field * x_val.real
            e_minus /= n
            
            grad[layer, q] = (e_plus - e_minus) / (2 * eps)
    
    return energy, grad.flatten()

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute
    idx_cache = _build_idx_cache(n)
    zz_diag, x_masks = _hamiltonian_terms(n, coupling, field)
    
    # Compute exact ground energy
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    H[np.arange(dim), np.arange(dim)] = zz_diag
    for i, mask in enumerate(x_masks):
        for b in range(dim):
            b2 = b ^ mask
            H[b, b2] -= field
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p, idx_cache)
        energy = np.sum(zz_diag * np.abs(st)**2).real
        for i, mask in enumerate(x_masks):
            partner = np.array([b ^ mask for b in range(dim)])
            x_val = np.sum(np.conj(st) * st[partner])
            energy += -field * x_val.real
        return (energy - e_ground) / n
    
    def obj_grad(xf):
        p = xf.reshape((layers, n))
        e, g = _energy_and_grad(n, layers, p, idx_cache, zz_diag, x_masks, coupling, field, e_ground)
        return e, g
    
    # Mean-field optimal angle
    ratio = field / coupling
    s_target = ratio * n / (2 * (n - 1))
    if s_target > 1.0:
        base = np.pi / 2
    elif s_target < -1.0:
        base = -np.pi / 2
    else:
        base = np.arcsin(s_target)
    
    # Initial guesses - more diverse
    guesses = []
    guesses.append(np.full((layers, n), base))
    guesses.append(np.full((layers, n), 0.0))
    guesses.append(np.full((layers, n), np.pi / 2))
    guesses.append(np.full((layers, n), -base))
    guesses.append(np.full((layers, n), base * 0.5))
    guesses.append(np.full((layers, n), base * 1.5))
    
    # Layer-dependent
    for scale in [0.3, 0.6, 0.9, 1.2, 1.5]:
        g = np.zeros((layers, n))
        for l in range(layers):
            g[l, :] = base * (0.3 + 0.7 * l / max(layers - 1, 1)) * scale
        guesses.append(g)
    
    # Alternating
    for sign in [1, -1]:
        g = np.zeros((layers, n))
        for l in range(layers):
            for q in range(n):
                g[l, q] = base * sign * (1 if q % 2 == 0 else -1)
        guesses.append(g)
    
    # Random
    np.random.seed(42)
    for _ in range(8):
        g = np.random.uniform(-np.pi, np.pi, (layers, n))
        guesses.append(g)
    
    # Near-base with small random perturbations
    for _ in range(8):
        g = base + np.random.normal(0, 0.3, (layers, n))
        guesses.append(g)
        
    best_params = guesses[0].copy()
    best_val = obj(best_params.flatten())
    
    # Optimize from each guess
    for gi, g in enumerate(guesses):
        if time.time() > deadline - 3:
            break
        x0 = g.flatten()
        try:
            res = minimize(obj, x0, method='L-BFGS-B', jac=obj_grad,
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Polish from best with tighter tolerance
    if time.time() < deadline - 1:
        try:
            res = minimize(obj, best_params.flatten(), method='L-BFGS-B', jac=obj_grad,
                         options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 30})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    return best_params
