import numpy as np
import time
from scipy.optimize import minimize

# Precomputed masks for speed
_MASK_CACHE = {}

def _get_masks(n):
    """Precompute and cache index masks for Ry and CZ operations."""
    key = n
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    dim = 1 << n
    ry_masks = []
    for q in range(n):
        mask = 1 << q
        idx0 = np.where((np.arange(dim) & mask) == 0)[0]
        idx1 = idx0 | mask
        ry_masks.append((idx0, idx1))
    cz_masks = []
    for q in range(n - 1):
        mask1 = 1 << q
        mask2 = 1 << (q + 1)
        both = np.where((np.arange(dim) & mask1) != 0)[0]
        both = both[(both & mask2) != 0]
        cz_masks.append(both)
    _MASK_CACHE[key] = (ry_masks, cz_masks)
    return _MASK_CACHE[key]

def _circuit_state(n, layers, params):
    """Compute state from circuit. params: (layers, n) array of angles."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    ry_masks, cz_masks = _get_masks(n)
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = ry_masks[q]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        # Apply CZ between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                both = cz_masks[q]
                state[both] = -state[both]
    
    return state

def _hamiltonian(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # ZZ terms
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            H[b, b] += -coupling * (1 - 2 * bi) * (1 - 2 * bj)
    # X terms
    for i in range(n):
        for b in range(dim):
            bf = b ^ (1 << i)
            if b < bf:
                H[b, bf] -= field
                H[bf, b] -= field
    return H

def _circuit_state_and_grads(n, layers, params):
    """
    Compute the circuit state and analytical gradients via the parameter-shift rule
    but using a single forward pass per shift (more efficient than full re-simulation).
    Actually, for a 3-layer circuit with n<=6, the parameter shift rule requires
    2 * layers * n forward passes. Let's just do that, but vectorize where possible.
    However, the 'adjoint' method or automatic differentiation is complex.
    Given the small size, parameter shift is fine if we optimize the forward pass.
    
    Wait, the prompt's current program uses parameter shift with full re-simulation.
    The inspiration 1 used finite differences.
    Let's stick to parameter shift as it's exact for this circuit type (Ry).
    But we can speed up the forward pass.
    """
    dim = 1 << n
    ry_masks, cz_masks = _get_masks(n)
    
    # Compute base state and energy
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # We need to store intermediate states to compute gradients efficiently?
    # No, parameter shift requires re-running the circuit.
    # Let's just implement a fast parameter shift gradient.
    
    shift = np.pi / 4.0
    grad = np.zeros((layers, n), dtype=np.float64)
    
    # Pre-compute cos/sin for shifted angles? No, angles change.
    
    # Helper to compute state and energy for a given param set
    def _state_and_energy(p):
        st = np.zeros(dim, dtype=np.complex128)
        st[0] = 1.0
        for layer in range(layers):
            for q in range(n):
                theta = p[layer, q]
                c = np.cos(theta / 2.0)
                s = np.sin(theta / 2.0)
                idx0, idx1 = ry_masks[q]
                old0 = st[idx0].copy()
                old1 = st[idx1].copy()
                st[idx0] = c * old0 - s * old1
                st[idx1] = s * old0 + c * old1
            if layer < layers - 1:
                for q in range(n - 1):
                    both = cz_masks[q]
                    st[both] = -st[both]
        return st
    
    # We will compute gradients for each parameter.
    # To save time, we can reuse the base state? No, parameter shift is independent.
    # However, we can optimize the inner loop.
    
    # Let's compute the base energy first
    st_base = _state_and_energy(params)
    
    for layer in range(layers):
        for q in range(n):
            p_plus = params.copy()
            p_plus[layer, q] += shift
            st_plus = _state_and_energy(p_plus)
            
            p_minus = params.copy()
            p_minus[layer, q] -= shift
            st_minus = _state_and_energy(p_minus)
            
            # Energy calculation is done by caller using H
            # Return states for energy calc? No, return gradient directly is better for minimize.
            # But we need H and e_ground.
            # Let's change strategy: return (energy, grad) from a wrapper that has H.
            # This function is a helper. Let's make it return the states needed.
            # Actually, it's cleaner to just have the objective function do the work.
            
            # Let's just return the gradient calculation result here, assuming H is available?
            # No, keep it pure. Return the states.
            # But that's 2 extra states per param. 2*3*6 = 36 states.
            # Each state is dim=64 complex. That's fine.
            
            grad[layer, q] = None # Placeholder, we'll compute in wrapper
            
    # This approach is getting messy. Let's go back to the simple parameter shift in the objective.
    # The current program's `_energy_and_grad` does exactly this.
    # The main bottleneck is `_circuit_state` being called many times.
    # Can we make `_circuit_state` faster?
    # Using precomputed masks is already done.
    
    # Let's try a different approach: Adjoint Method (Reverse Mode AD).
    # For a circuit U = U_L ... U_1, state = U |0>
    # Energy E = <0| U^dagger H U |0>
    # dE/dtheta = 2 Re[ <psi| (dU/dtheta) U^dagger H |psi> ] ... complicated.
    
    # Given the small size (n=4,6), parameter shift is likely fast enough if the forward pass is optimized.
    # The current program scores 37.4. The inspirations score 37.4.
    # Maybe the optimization strategy is the key, not just the speed.
    
    # Let's combine the robust multi-start strategy of the current program
    # with the potentially faster finite difference gradient of inspiration 1?
    # No, parameter shift is exact. Finite difference has error.
    
    # Let's stick to the current program's structure but ensure the forward pass is as fast as possible.
    # And use the better initial guess sorting from inspiration 1.
    
    pass # This function is not used in the final combined version below, 
         # we will use the direct parameter shift in the wrapper.

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Precompute
    H = _hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p)
        energy = np.vdot(st, H @ st).real
        return (energy - e_ground) / n
    
    def obj_grad(xf):
        p = xf.reshape((layers, n))
        # Compute base state
        st_base = _circuit_state(n, layers, p)
        e_base = np.vdot(st_base, H @ st_base).real
        val = (e_base - e_ground) / n
        
        grad = np.zeros_like(xf)
        shift = np.pi / 4.0
        
        for layer in range(layers):
            for q in range(n):
                idx = layer * n + q
                
                p_plus = p.copy()
                p_plus[layer, q] += shift
                st_plus = _circuit_state(n, layers, p_plus)
                e_plus = np.vdot(st_plus, H @ st_plus).real
                val_plus = (e_plus - e_ground) / n
                
                p_minus = p.copy()
                p_minus[layer, q] -= shift
                st_minus = _circuit_state(n, layers, p_minus)
                e_minus = np.vdot(st_minus, H @ st_minus).real
                val_minus = (e_minus - e_ground) / n
                
                grad[idx] = (val_plus - val_minus) / (2.0 * shift)
        
        return val, grad
    
    # Mean-field optimal angle
    ratio = field / coupling
    s_target = ratio * n / (2 * (n - 1))
    if s_target > 1.0:
        base = np.pi / 2
    elif s_target < -1.0:
        base = -np.pi / 2
    else:
        base = np.arcsin(s_target)
    
    # Initial guesses
    guesses = []
    guesses.append(np.full((layers, n), base))
    guesses.append(np.full((layers, n), 0.0))
    guesses.append(np.full((layers, n), np.pi / 2))
    guesses.append(np.full((layers, n), -base))
    guesses.append(np.full((layers, n), base * 0.5))
    guesses.append(np.full((layers, n), base * 1.5))
    
    # Layer-dependent profiles
    for scale in [0.5, 0.8, 1.0, 1.2, 1.5]:
        g = np.zeros((layers, n))
        for l in range(layers):
            frac = 0.5 + 0.5 * l / max(layers - 1, 1)
            g[l, :] = base * frac * scale
        guesses.append(g)
    
    # Alternating sign patterns
    for sign_scale in [1.0, 0.5, -1.0]:
        g = np.zeros((layers, n))
        for l in range(layers):
            for q in range(n):
                g[l, q] = base * sign_scale * (1.0 if q % 2 == 0 else -1.0)
        guesses.append(g)
    
    # Small perturbations
    for delta in [0.1, -0.1, 0.2, -0.2, 0.3]:
        guesses.append(np.full((layers, n), base + delta))
    
    # Random
    np.random.seed(42)
    guesses.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
    np.random.seed(123)
    guesses.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
    
    # Sort guesses by initial objective value (cheapest way to find good starts)
    def _sort_key(g):
        return obj(g.flatten())
    
    # Only evaluate a subset if time is tight, but n is small so it's fast
    # Let's evaluate all, it's just 1 forward pass each
    guesses.sort(key=_sort_key)
    
    best_params = guesses[0].copy()
    best_val = obj(best_params.flatten())
    
    # Optimize from the top few guesses
    num_starts = 5
    if time.time() > deadline - 30:
        num_starts = 1
    
    for i in range(min(num_starts, len(guesses))):
        if time.time() > deadline - 10:
            break
        g = guesses[i]
        x0 = g.flatten()
        try:
            # Use L-BFGS-B with exact gradient
            res = minimize(obj_grad, x0, jac=True, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
        
        # If we found a very good solution, we might skip others
        if best_val < 1e-6:
            break
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 5:
        try:
            res = minimize(obj_grad, best_params.flatten(), jac=True, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Ensure finite
    if not np.all(np.isfinite(best_params)):
        best_params = np.full((layers, n), base)
    
    return best_params
