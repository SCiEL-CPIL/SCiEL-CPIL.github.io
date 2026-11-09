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

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 147.0  # 150s kill - 3s safety margin
    
    H = _hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p)
        return (np.vdot(st, H @ st).real - e_ground) / n
    
    def grad(xf):
        """Compute gradient via finite differences (vectorized for speed)."""
        g = np.zeros_like(xf)
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p)
        e0 = np.vdot(st, H @ st).real
        
        eps = 1e-7
        for i in range(len(xf)):
            p_plus = p.copy()
            p_plus.flat[i] += eps
            st_plus = _circuit_state(n, layers, p_plus)
            e_plus = np.vdot(st_plus, H @ st_plus).real
            g[i] = (e_plus - e0) / eps / n
        return g
    
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
    
    # Layer-dependent patterns
    for scale in [0.5, 0.8, 1.0, 1.2]:
        g = np.zeros((layers, n))
        for l in range(layers):
            g[l, :] = base * (0.5 + 0.5 * l / max(layers - 1, 1)) * scale
        guesses.append(g)
    
    # Alternating patterns
    for sign_scale in [1.0, 0.5]:
        g = np.zeros((layers, n))
        for l in range(layers):
            for q in range(n):
                g[l, q] = base * sign_scale * (1 if q % 2 == 0 else -1)
        guesses.append(g)
    
    # Small perturbations around base
    for delta in [0.1, -0.1, 0.2, -0.2]:
        guesses.append(np.full((layers, n), base + delta))
    
    best_params = guesses[0].copy()
    best_val = obj(best_params.flatten())
    
    # Sort guesses by initial objective value
    def _sort_key(g):
        return obj(g.flatten())
    
    guesses.sort(key=_sort_key)
    
    # Optimize from each good initial guess
    for g in guesses[:8]:  # Limit to top 8 to stay within time budget
        if time.time() > deadline - 10:
            break
        x0 = g.flatten()
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(obj, best_params.flatten(), jac=grad, method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    return best_params
