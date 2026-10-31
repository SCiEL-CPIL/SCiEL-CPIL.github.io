"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))


def _smooth_max_grad(x, alpha=100.0):
    """Gradient of smooth max with respect to x."""
    x_max = np.max(x)
    exps = np.exp(alpha * (x - x_max))
    weights = exps / np.sum(exps)
    return weights


def _overlap_and_grad(h, alpha):
    """Compute overlap and gradient of smooth max w.r.t. h."""
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    w = _smooth_max_grad(overlap, alpha)
    
    grad_h = np.zeros(N)
    for i in range(N - 1):
        if w[i] > 1e-15:
            grad_h[i] += w[i] * (1.0 - h[i+1])
            grad_h[i+1] += w[i] * (-h[i])
    if w[N-1] > 1e-15:
        grad_h[N-1] += w[N-1]
    
    return overlap, smax, grad_h, w


def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        h = np.full(N, 0.5)
        return h
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    # Fix residual iteratively
    for _ in range(20):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        else:
            mask = h > 0.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        h = np.clip(h, 0.0, 1.0)
    return h


def _two_block_init():
    """Analytic 2-block initial guess."""
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A  # ~0.618
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    return h0


def _sinusoidal_h(A, phi, N=N):
    """Generate sinusoidal step heights, projected to [0,1] with sum=N/2."""
    x = np.arange(N)
    h = 0.5 + A * np.sin(2.0 * np.pi * x / N + phi)
    return _project_to_feasible(h)


def _sinusoidal_score(params):
    """Score for sinusoidal parameters [A, phi]."""
    A, phi = params
    h = _sinusoidal_h(A, phi)
    return _score(h)


def _optimize_sinusoidal(deadline):
    """Optimize sinusoidal parameters A and phi via grid search + local refinement."""
    best_score = float('inf')
    best_A = 0.0
    best_phi = 0.0
    
    # Grid search over A in [0, 0.5] and phi in [0, 2*pi)
    n_A = 50
    n_phi = 100
    for i in range(n_A):
        if time.time() > deadline - 30:
            break
        A = i * 0.5 / (n_A - 1)
        for j in range(n_phi):
            phi = 2.0 * np.pi * j / n_phi
            h = _sinusoidal_h(A, phi)
            sc = _score(h)
            if sc < best_score:
                best_score = sc
                best_A = A
                best_phi = phi
    
    # Local refinement around best grid point
    try:
        x0 = np.array([best_A, best_phi])
        result = minimize(
            _sinusoidal_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
        )
        if result.fun < best_score:
            best_score = result.fun
            best_A, best_phi = result.x
    except Exception:
        pass
    
    # Further refinement with tighter Nelder-Mead
    try:
        x0 = np.array([best_A, best_phi])
        result2 = minimize(
            _sinusoidal_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
        )
        if result2.fun < best_score:
            best_score = result.fun
            best_A, best_phi = result2.x
    except Exception:
        pass
    
    h_best = _sinusoidal_h(best_A, best_phi)
    return h_best, best_score


def _slsqp_solve(h_init, t_init, alpha, maxiter, ftol, deadline):
    """Run SLSQP with epigraph formulation."""
    def objective_epigraph(params):
        h = params[:N]
        t = params[N]
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha)
        return t + 1000.0 * np.maximum(0.0, smax - t)**2
    
    x0 = np.concatenate([h_init, [t_init]])
    bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    
    def constraint_sum(params):
        return np.sum(params[:N]) - N / 2.0
    
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    def gradient(params):
        h = params[:N]
        t = params[N]
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha)
        w = _smooth_max_grad(overlap, alpha)
        
        grad_h = np.zeros(N)
        for i in range(N - 1):
            if w[i] > 1e-15:
                grad_h[i] += w[i] * (1.0 - h[i+1])
                grad_h[i+1] += w[i] * (-h[i])
        if w[N-1] > 1e-15:
            grad_h[N-1] += w[N-1]
        
        # Gradient of penalty term
        penalty_grad = 0.0
        if smax > t:
            penalty_grad = 2000.0 * (smax - t)
        
        grad_t = 1.0 + penalty_grad
        
        return np.concatenate([grad_h * (1.0 + penalty_grad) + penalty_grad * 0, [grad_t]])
    
    res = minimize(
        objective_epigraph,
        x0,
        jac=gradient,
        method='SLSQP',
        bounds=bounds_ext,
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': ftol}
    )
    
    if res.success or res.x is not None:
        h_opt = res.x[:N]
        t_opt = res.x[N]
        return h_opt, t_opt
    return None, None


def _find_optimal_2block(k):
    """Find optimal a for 2-block by analyzing critical points analytically."""
    n = N
    
    # b = (n/2 - k*a) / (n-k)
    # Candidates for max:
    #   f1 = a*(1-a)          (if k >= 2)
    #   f2 = a*(1-b)
    #   f3 = b*(1-b)          (if n-k >= 2)
    #   f4 = b
    
    # b(a) = (n/2 - k*a) / (n-k)
    # db/da = -k/(n-k)
    
    # f2(a) = a*(1 - (n/2 - k*a)/(n-k)) = a * (n-k - n/2 + k*a)/(n-k)
    #        = a * (n/2 - k + k*a)/(n-k)
    #        = (k*a^2 + (n/2-k)*a) / (n-k)
    # df2/da = (2k*a + (n/2-k)) / (n-k)
    
    # f3(a) = b*(1-b), where b = (n/2 - k*a)/(n-k)
    # df3/da = (1-2b) * db/da = (1-2b) * (-k/(n-k))
    
    # f4(a) = b = (n/2 - k*a)/(n-k)
    # df4/da = -k/(n-k)
    
    # f1(a) = a*(1-a), df1/da = 1-2a
    
    # Critical points: where two functions are equal and their max is minimized.
    # The optimal is at a point where the active constraints cross.
    
    # Let's enumerate pairs of functions that could be equal at the optimum:
    # For each pair (fi, fj), solve fi(a) = fj(a) and check if it's a valid local min.
    
    candidates = []
    
    # Helper: evaluate all candidate values at a given a
    def eval_at(a):
        if a < 0 or a > 1:
            return None
        b = (n/2.0 - k*a) / (n-k)
        if b < -1e-12 or b > 1.0 + 1e-12:
            return None
        b = max(0.0, min(1.0, b))
        vals = []
        if k >= 2:
            vals.append(('f1', a*(1.0-a)))
        vals.append(('f2', a*(1.0-b)))
        if n-k >= 2:
            vals.append(('f3', b*(1.0-b)))
        vals.append(('f4', b))
        return max(v for _, v in vals), b
    
    # Collect candidate a values:
    # 1. Where f1 = f2: a(1-a) = a(1-b) -> if a>0: 1-a = 1-b -> a=b -> a = (n/2 - k*a)/(n-k) -> a(n-k) = n/2 - k*a -> a*n = n/2 -> a = 1/2
    if k >= 2:
        candidates.append(0.5)
    
    # 2. Where f1 = f3: a(1-a) = b(1-b)
    #    b = (n/2 - k*a)/(n-k)
    #    This is a quadratic in a. Let's solve numerically.
    if k >= 2 and n-k >= 2:
        # a(1-a) = b(1-b) where b = (n/2 - k*a)/(n-k)
        # Let's use numpy roots
        # a - a^2 = b - b^2
        # a - a^2 - b + b^2 = 0
        # b = (n/2 - k*a)/(n-k)
        # Let c = n-k, d = n/2
        # b = (d - k*a)/c
        # a - a^2 - (d - k*a)/c + ((d - k*a)/c)^2 = 0
        # Multiply by c^2:
        # a*c^2 - a^2*c^2 - (d - k*a)*c + (d - k*a)^2 = 0
        # a*c^2 - a^2*c^2 - d*c + k*a*c + d^2 - 2*d*k*a + k^2*a^2 = 0
        # a^2*(-c^2 + k^2) + a*(c^2 + k*c - 2*d*k) + (-d*c + d^2) = 0
        c = n - k
        d = n / 2.0
        A_coef = -(c**2) + k**2
        B_coef = c**2 + k*c - 2*d*k
        C_coef = -d*c + d**2
        if abs(A_coef) > 1e-15:
            disc = B_coef**2 - 4*A_coef*C_coef
            if disc >= 0:
                sqrt_disc = math.sqrt(disc)
                a1 = (-B_coef + sqrt_disc) / (2*A_coef)
                a2 = (-B_coef - sqrt_disc) / (2*A_coef)
                candidates.extend([a1, a2])
        else:
            if abs(B_coef) > 1e-15:
                candidates.append(-C_coef / B_coef)
    
    # 3. Where f1 = f4: a(1-a) = b = (n/2 - k*a)/(n-k)
    if k >= 2:
        # a(1-a) = (n/2 - k*a)/(n-k)
        # a(n-k) - a^2(n-k) = n/2 - k*a
        # -a^2(n-k) + a(n-k) + k*a - n/2 = 0
        # -a^2(n-k) + a(n-k+k) - n/2 = 0
        # -a^2(n-k) + a*n - n/2 = 0
        # a^2(n-k) - a*n + n/2 = 0
        A_coef = n - k
        B_coef = -n
        C_coef = n / 2.0
        disc = B_coef**2 - 4*A_coef*C_coef
        if disc >= 0:
            sqrt_disc = math.sqrt(disc)
            a1 = (-B_coef + sqrt_disc) / (2*A_coef)
            a2 = (-B_coef - sqrt_disc) / (2*A_coef)
            candidates.extend([a1, a2])
    
    # 4. Where f2 = f3: a(1-b) = b(1-b)
    #    If b != 1: a = 1-b -> a + b = 1
    #    a + (n/2 - k*a)/(n-k) = 1
    #    a(n-k) + n/2 - k*a = n-k
    #    a*n + n/2 = n-k
    #    a = (n-k-n/2)/n = (n/2-k)/n = 1/2 - k/n
    if n-k >= 2:
        candidates.append(1.0/2.0 - k/n)
    
    # 5. Where f2 = f4: a(1-b) = b -> a = b/(1-b) (if b < 1)
    #    a(1-b) = b -> a - a*b = b -> a = b(1+a) -> a = b/(1-a)
    #    a(1-a) = b = (n/2 - k*a)/(n-k)
    #    This is same as f1 = f4! Already covered.
    
    # 6. Where f3 = f4: b(1-b) = b -> if b > 0: 1-b = 1 -> b = 0
    #    b = 0 -> n/2 - k*a = 0 -> a = n/(2k)
    if n-k >= 2:
        candidates.append(n / (2.0 * k))
    
    # 7. Where f2 is minimized: df2/da = 0 -> 2k*a + (n/2-k) = 0 -> a = (k - n/2)/(2k)
    #    This is negative for k < n/2, so likely not in range, but check.
    candidates.append((k - n/2.0) / (2.0*k))
    
    # 8. Where f3 is minimized: df3/da = 0 -> 1-2b = 0 -> b = 1/2
    #    (n/2 - k*a)/(n-k) = 1/2 -> n/2 - k*a = (n-k)/2 -> k*a = n/2 - (n-k)/2 = k/2 -> a = 1/2
    candidates.append(0.5)
    
    # 9. Boundary points
    a_min = max(0.0, (n/2.0 - n + k) / k)
    a_max = min(1.0, n / (2.0 * k))
    candidates.extend([a_min, a_max, a_min + 1e-10, a_max - 1e-10])
    
    # Evaluate all candidates
    best_score = float('inf')
    best_a = None
    
    for a in candidates:
        result = eval_at(a)
        if result is None:
            continue
        s, b = result
        if s < best_score:
            best_score = s
            best_a = a
    
    # Also do a fine grid search to catch any missed critical points
    if a_min < a_max:
        a_grid = np.linspace(a_min + 1e-10, a_max - 1e-10, 1000)
        for a in a_grid:
            result = eval_at(a)
            if result is None:
                continue
            s, b = result
            if s < best_score:
                best_score = s
                best_a = a
    
    # Refine around best_a with very fine grid
    if best_a is not None and a_min < a_max:
        step = (a_max - a_min) / 1000.0
        a_refine = np.linspace(max(a_min, best_a - 3*step), min(a_max, best_a + 3*step), 500)
        for a in a_refine:
            result = eval_at(a)
            if result is None:
                continue
            s, b = result
            if s < best_score:
                best_score = s
                best_a = a
    
    if best_a is None:
        return None, float('inf')
    
    b = (n/2.0 - k*best_a) / (n-k)
    b = max(0.0, min(1.0, b))
    
    return best_a, best_score


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    best_h = None
    best_score = float('inf')

    # Phase 1: Analytical 2-block search (fast, gives good initial guess)
    for k in range(1, N):
        a_opt, s_exact = _find_optimal_2block(k)
        if a_opt is not None and s_exact < best_score:
            best_score = s_exact
            b_opt = (N / 2.0 - k * a_opt) / (N - k)
            b_opt = np.clip(b_opt, 0.0, 1.0)
            h = np.zeros(N)
            h[:k] = a_opt
            h[k:] = b_opt
            s = h.sum()
            diff = N / 2.0 - s
            if abs(diff) > 1e-15:
                h[k - 1] += diff
                h = np.clip(h, 0.0, 1.0)
            best_h = h.copy()
    
    if time.time() > deadline - 60:
        return best_h if best_h is not None else _two_block_init()

    # Phase 2: Sinusoidal parameter optimization (explores periodic structures)
    h_sin, score_sin = _optimize_sinusoidal(deadline - 20)
    if score_sin < best_score:
        best_score = score_sin
        best_h = h_sin.copy()
    
    if time.time() > deadline - 40:
        return best_h if best_h is not None else _two_block_init()

    # Phase 3: Full SLSQP optimization with smooth max (refines arbitrary structure)
    # Start from best of 2-block and sinusoidal
    h_init = best_h.copy() if best_h is not None else _two_block_init()
    t_init = best_score + 0.05
    
    # Try a few different alpha values for smoothness
    for alpha in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 10:
            break
        h_opt, t_opt = _slsqp_solve(h_init, t_init, alpha, maxiter=500, ftol=1e-10, deadline=deadline)
        if h_opt is not None:
            h_opt = _project_to_feasible(h_opt)
            sc = _score(h_opt)
            if sc < best_score:
                best_score = sc
                best_h = h_opt.copy()
    
    # Phase 4: Final polish with SLSQP on the best solution found
    if best_h is not None and time.time() < deadline - 5:
        h_init = best_h.copy()
        t_init = best_score + 0.01
        h_opt, t_opt = _slsqp_solve(h_init, t_init, 200.0, maxiter=300, ftol=1e-12, deadline=deadline)
        if h_opt is not None:
            h_opt = _project_to_feasible(h_opt)
            sc = _score(h_opt)
            if sc < best_score:
                best_score = sc
                best_h = h_opt.copy()

    if best_h is None:
        best_h = _two_block_init()

    return best_h
