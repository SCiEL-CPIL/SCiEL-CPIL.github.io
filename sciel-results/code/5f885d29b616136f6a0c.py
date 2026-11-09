"""Direct structure-derived QAOA angle design with analytic gradient ascent and multi-restart polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def _precompute_cut_sizes(n, edges):
    """Precompute cut size for all 2^n states vectorized."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        states = np.arange(dim, dtype=np.int64)
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes


def _apply_x_layer(state, beta, n):
    """Apply exp(-i beta sum_i X_i) via tensor reshape per qubit."""
    c = np.cos(beta)
    s = np.sin(beta)
    tensor = state.reshape([2] * n)
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] = -1j * s * flat[0] + c * flat[1]
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    return tensor.flatten()


def _qaoa_state(n, cut_sizes, angles):
    """Simulate p-layer QAOA circuit."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    for gamma, beta in angles:
        state *= np.exp(-1j * gamma * cut_sizes)
        state = _apply_x_layer(state, beta, n)
    return state


def _expected_cut(n, cut_sizes, angles):
    """Compute E[C] = <psi|C|psi>."""
    state = _qaoa_state(n, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _objective(params, n, cut_sizes, p):
    """Negative expected cut for minimization."""
    angles = params.reshape(p, 2)
    return -_expected_cut(n, cut_sizes, angles)


def _exact_gradient(n, edges, cut_sizes, p, x):
    """
    Compute exact gradient of expected cut using parameter-shift rule.
    For QAOA with cost/mixing operators, the parameter-shift rule gives:
    dE/dtheta = (E(theta + pi/4) - E(theta - pi/4)) / (2*sin(pi/4))
    This is exact for the QAOA ansatz.
    """
    angles = x.reshape(p, 2)
    shift = np.pi / 4
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += shift
        x_minus = x.copy()
        x_minus[i] -= shift
        
        val_plus = _expected_cut(n, cut_sizes, x_plus.reshape(p, 2))
        val_minus = _expected_cut(n, cut_sizes, x_minus.reshape(p, 2))
        
        grad[i] = (val_plus - val_minus) / (2 * np.sin(shift))
    
    return grad


def _approx_hessian(n, edges, cut_sizes, p, x, grad, h=0.01):
    """
    Approximate Hessian using finite differences of the exact gradient.
    """
    m = len(x)
    H = np.zeros((m, m), dtype=np.float64)
    
    for i in range(m):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        
        g_plus = _exact_gradient(n, edges, cut_sizes, p, x_plus)
        g_minus = _exact_gradient(n, edges, cut_sizes, p, x_minus)
        
        for j in range(m):
            H[i, j] = (g_plus[j] - g_minus[j]) / (2 * h)
    
    # Symmetrize
    H = (H + H.T) / 2.0
    return H


def _newton_polish(n, edges, cut_sizes, p, x0, max_iter=20):
    """
    Newton-Raphson polish using exact gradients and approximate Hessian.
    """
    x = x0.copy()
    best_x = x.copy()
    best_val = _expected_cut(n, cut_sizes, x.reshape(p, 2))
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        val = _expected_cut(n, cut_sizes, x.reshape(p, 2))
        if val > best_val:
            best_val = val
            best_x = x.copy()
        
        grad = _exact_gradient(n, edges, cut_sizes, p, x)
        
        # Check convergence
        if np.max(np.abs(grad)) < 1e-12:
            break
        
        # Compute Hessian
        H = _approx_hessian(n, edges, cut_sizes, p, x, grad)
        
        # Solve H * delta = -grad (for maximization, we want to go in +grad direction)
        # Newton step: x_new = x + H^{-1} * grad (for maximization)
        try:
            # Add small regularization for stability
            H_reg = H + 1e-6 * np.eye(len(x))
            delta = np.linalg.solve(H_reg, grad)
            
            # Damping if step is too large
            step_norm = np.linalg.norm(delta)
            if step_norm > 0.5:
                delta = delta * (0.5 / step_norm)
            
            x_new = x + delta
            x_new = x_new % (2 * np.pi)
            
            val_new = _expected_cut(n, cut_sizes, x_new.reshape(p, 2))
            if val_new > best_val:
                best_val = val_new
                best_x = x_new.copy()
            x = x_new
        except np.linalg.LinAlgError:
            # Fallback to gradient step
            x = x + 0.05 * grad
            x = x % (2 * np.pi)
    
    return best_x, best_val


def _graph_invariant_init(n, edges, p):
    """
    Direct initialization from graph structure.
    For MaxCut QAOA p=2: gamma scales with edge density, beta with mixing scale.
    """
    m = len(edges)
    max_edges = n * (n - 1) // 2
    density = m / max_edges if max_edges > 0 else 0.0
    
    # Average degree
    avg_deg = 2.0 * m / n if n > 0 else 0.0
    
    # Degree variance as a complexity indicator
    deg = np.zeros(n)
    for (i, j) in edges:
        deg[i] += 1
        deg[j] += 1
    deg_var = np.var(deg)
    
    # Heuristic: denser graphs need larger gamma in first layer
    # Literature: gamma1 ~ pi/4 for dense, smaller for sparse
    gamma1 = np.pi * 0.35 + 0.3 * density
    gamma2 = np.pi * 0.15 + 0.2 * density
    
    # Beta: mixing layer, typically smaller
    beta1 = np.pi * 0.25 - 0.1 * density
    beta2 = np.pi * 0.15
    
    # Adjust for degree variance (more heterogeneous -> slightly larger angles)
    scale = 1.0 + 0.1 * (deg_var / (avg_deg + 1e-9)) if avg_deg > 0 else 1.0
    gamma1 *= scale
    gamma2 *= scale
    
    angles = np.array([
        [gamma1 % (2 * np.pi), beta1 % (2 * np.pi)],
        [gamma2 % (2 * np.pi), beta2 % (2 * np.pi)]
    ], dtype=np.float64)
    return angles


def _nelder_mead(x0, n, cut_sizes, p, maxiter=200, tol=1e-10):
    """Nelder-Mead with time guard."""
    try:
        if _time_left() < 5.0:
            return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))
        res = minimize(
            _objective, x0, args=(n, cut_sizes, p),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _analytic_gradient_ascent(n, edges, cut_sizes, p, x0, max_iter=500, tol=1e-10):
    """
    Analytic gradient ascent for QAOA angles.
    Uses the parameter-shift rule for exact gradients.
    """
    x = x0.copy()
    best_x = x.copy()
    best_val = _expected_cut(n, cut_sizes, x.reshape(p, 2))
    
    for it in range(max_iter):
        if _time_left() < 10.0:
            break
        
        # Compute current value
        val = _expected_cut(n, cut_sizes, x.reshape(p, 2))
        if val > best_val:
            best_val = val
            best_x = x.copy()
        
        # Compute gradient using parameter-shift rule
        grad = _exact_gradient(n, edges, cut_sizes, p, x)
        
        # Check convergence
        if np.max(np.abs(grad)) < tol:
            break
        
        # Adaptive learning rate
        lr = 0.1 / (1.0 + it * 0.01)
        
        # Gradient ascent step
        x = x + lr * grad
        # Wrap to [0, 2*pi)
        x = x % (2 * np.pi)
    
    return best_x, best_val


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    best_angles = None
    best_score = -np.inf
    
    # Build diverse initial guesses combining structure-derived and heuristic starts
    initial_guesses = []
    
    # Structure-derived direct init
    init_struct = _graph_invariant_init(n, edges, p)
    initial_guesses.append(init_struct.flatten())
    
    # Perturbed structure-derived inits
    rng = np.random.default_rng(42)
    for _ in range(6):
        noise_g = rng.normal(0, 0.15, p)
        noise_b = rng.normal(0, 0.15, p)
        perturbed = init_struct.copy()
        perturbed[:, 0] = (perturbed[:, 0] + noise_g) % (2 * np.pi)
        perturbed[:, 1] = (perturbed[:, 1] + noise_b) % (2 * np.pi)
        initial_guesses.append(perturbed.flatten())
    
    # Standard literature-based anchors
    initial_guesses.append(np.array([0.35, 0.25, 0.25, 0.15]))
    initial_guesses.append(np.array([0.5, 0.3, 0.3, 0.2]))
    initial_guesses.append(np.array([0.7, 0.2, 0.4, 0.15]))
    initial_guesses.append(np.array([np.pi/4, np.pi/8, np.pi/8, np.pi/16]))
    
    # Random uniform inits
    for _ in range(15):
        g = rng.uniform(0, np.pi, p)
        b = rng.uniform(0, np.pi, p)
        initial_guesses.append(np.column_stack((g, b)).flatten())
    
    # Multi-restart optimization: use analytic gradient ascent for first few, then Nelder-Mead
    for idx, x0 in enumerate(initial_guesses):
        if _time_left() < 15.0:
            break
        
        # Use analytic gradient ascent for first 3 restarts (faster convergence in smooth regions)
        if idx < 3:
            x_opt, score = _analytic_gradient_ascent(n, edges, cut_sizes, p, x0, max_iter=300, tol=1e-9)
        else:
            # Use Nelder-Mead for remaining restarts
            x_opt, score = _nelder_mead(x0, n, cut_sizes, p, maxiter=150, tol=1e-9)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Final high-precision polish using Newton-Raphson with exact gradients
    if best_angles is not None and _time_left() > 10.0:
        try:
            x_newton, score_newton = _newton_polish(n, edges, cut_sizes, p, best_angles.flatten(), max_iter=15)
            if score_newton > best_score:
                best_score = score_newton
                best_angles = x_newton.reshape(p, 2)
        except Exception:
            pass
    
    # Additional polish with very tight tolerance Nelder-Mead
    if best_angles is not None and _time_left() > 8.0:
        try:
            res = minimize(
                _objective, best_angles.flatten(), args=(n, cut_sizes, p),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_angles = res.x.reshape(p, 2)
        except Exception:
            pass
    
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
# EVOLVE-BLOCK-END
