"""Refined uncertainty inequality: Dyson-gas equilibrium Laguerre construction for C4 upper bound."""
import time
import numpy as np
from scipy.optimize import minimize, fsolve
from scipy.special import gamma, gammaln

_START = time.time()
_DEADLINE = 630.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - _MARGIN - time.time()


def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 from the Laguerre polynomial construction.
    P(x) = prod_{i=0}^{11} (x - r_i)^2, degree 24.
    Norm: ||P||^2 = integral_0^inf P(x)^2 * x^2 * exp(-x) dx
    C4_bound = 1 / sqrt(||P||^2)
    """
    # Build P(x) = [prod (x - r_i)]^2
    q_coeff = np.array([1.0])
    for r in roots:
        q_coeff = np.convolve(q_coeff, np.array([1.0, -r]))
    p_coeff = np.convolve(q_coeff, q_coeff)
    
    # ||P||^2 = sum_{i,j} p_i * p_j * Gamma(i+j+3)  (alpha=2, so i+j+alpha+1 = i+j+3)
    n = len(p_coeff)
    indices = np.arange(n)
    gamma_args = indices[:, None] + indices[None, :] + 3
    log_gv = gammaln(gamma_args)
    log_gv = np.clip(log_gv, -700, 700)
    gv = np.exp(log_gv)
    
    norm_sq = p_coeff @ (gv @ p_coeff)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return np.inf
    
    return 1.0 / np.sqrt(norm_sq)


def _dyson_equilibrium(roots, beta=2.0, confining_strength=1.0):
    """
    Compute the Dyson gas force on each root.
    For double roots (beta=2), the repulsive force between root i and j is:
      F_rep(i,j) = beta / (r_i - r_j)^2  (repulsive, pushes apart)
    The confining force from the Laguerre weight x^alpha * exp(-x) is:
      F_conf(i) = -alpha / r_i + 1  (for large x, the exp(-x) dominates giving +1 force toward origin)
    
    Actually, let me derive this properly from the norm.
    
    The norm ||P||^2 where P(x) = prod (x-r_i)^2.
    Taking d/dr_k of ln(||P||^2) and setting to 0 gives the equilibrium.
    
    For the Laguerre inner product with weight x^2 e^{-x}:
    The equilibrium condition is:
      sum_{j != k} 2/(r_k - r_j)^2 + 2/r_k - 1 = 0  (approximately, for the double-root case)
    
    More precisely, for P(x) = prod_i (x - r_i)^2:
    d/d r_k ln ||P||^2 = 0
    
    This gives: sum_{j != k} 2/(r_k - r_j) + 2/r_k - 1 = 0  (for simple roots)
    For double roots: sum_{j != k} 4/(r_k - r_j)^2 + 2/r_k - 1 = 0
    
    Let me just use the gradient of the objective directly.
    """
    forces = np.zeros_like(roots)
    for k in range(len(roots)):
        # Repulsive forces from other roots (double roots -> squared)
        for j in range(len(roots)):
            if j != k:
                diff = roots[k] - roots[j]
                if abs(diff) < 1e-12:
                    forces[k] += 1e6 * np.sign(diff)
                else:
                    forces[k] += 4.0 / (diff ** 2)  # double root repulsion
        # Confining force: from x^2 * exp(-x) weight
        # The confining potential is approximately -2*ln(r) + r for the weight
        # Force = -dV/dr = 2/r - 1
        forces[k] += 2.0 / roots[k] - 1.0
    return forces


def _dyson_initial_guess(m=12):
    """
    Generate initial guess for Dyson gas equilibrium.
    For a Dyson gas with beta=2 and linear confinement, the roots spread out.
    Use the known asymptotic: for large m, roots are approximately at positions
    where the density satisfies the equilibrium equation.
    
    For our specific case (12 roots, weight x^2 e^{-x}), start with
    a reasonable spread and let the solver refine.
    """
    # Start with roots spread from ~1 to ~50
    # The Laguerre zeros of L_{24}^{2}(x) are a good reference
    # They range from about 0.5 to about 70
    # For double roots, they'll be slightly different
    return np.linspace(1.0, 45.0, m)


def _newton_dyson(roots, max_iter=200, tol=1e-12):
    """
    Newton's method to solve the Dyson equilibrium equations.
    """
    roots = np.array(roots, dtype=float).copy()
    m = len(roots)
    
    for iteration in range(max_iter):
        if _time_left() < 10:
            break
        
        # Compute forces (residuals)
        F = np.zeros(m)
        for k in range(m):
            for j in range(m):
                if j != k:
                    diff = roots[k] - roots[j]
                    if abs(diff) < 1e-10:
                        F[k] += 1e8 * np.sign(diff + 1e-15)
                    else:
                        F[k] += 4.0 / (diff ** 2)
            F[k] += 2.0 / roots[k] - 1.0
        
        # Check convergence
        if np.max(np.abs(F)) < tol:
            break
        
        # Compute Jacobian (sparse structure)
        J = np.zeros((m, m))
        for k in range(m):
            # Diagonal: dF_k/dr_k
            J[k, k] = 2.0 / roots[k] ** 2
            for j in range(m):
                if j != k:
                    diff = roots[k] - roots[j]
                    if abs(diff) < 1e-10:
                        J[k, k] += 2e8
                        J[k, j] -= 2e8
                    else:
                        J[k, k] += 8.0 / (diff ** 3)
                        J[k, j] -= 8.0 / (diff ** 3)
        
        # Solve J * delta = -F
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            try:
                delta = np.linalg.lstsq(J, -F, rcond=None)[0]
            except:
                break
        
        # Damping to ensure positivity and avoid large steps
        max_step = 0.5
        if np.max(np.abs(delta)) > max_step:
            delta = delta * max_step / np.max(np.abs(delta))
        
        roots_new = roots + delta
        
        # Ensure positivity
        if np.any(roots_new <= 0):
            # Scale back
            for t in np.linspace(0.5, 0.01, 10):
                roots_try = roots + t * delta
                if np.all(roots_try > 0):
                    roots_new = roots_try
                    break
            else:
                roots_new = roots + 0.01 * delta
                if np.any(roots_new <= 0):
                    break
        
        roots = roots_new
    
    return roots


def _objective_direct(params):
    """Objective for direct 12-parameter optimization."""
    roots = np.exp(params)  # log-parameterization ensures positivity
    if np.any(roots > 1e6) or np.any(roots < 1e-6):
        return 1e10
    return _compute_c4_bound(roots)


def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the upper bound on C4.
    Returns a list of 12 positive root locations.
    """
    deadline = _START + _DEADLINE - _MARGIN
    
    # Phase 1: Dyson gas equilibrium initialization
    best_roots = None
    best_score = np.inf
    
    # Multiple initial guesses for the Dyson solver
    initial_guesses = [
        np.linspace(1.0, 45.0, 12),
        np.linspace(0.5, 60.0, 12),
        np.linspace(2.0, 35.0, 12),
        np.array([1, 2, 4, 6, 9, 13, 18, 24, 31, 39, 48, 58], dtype=float),
        4.0 * np.arange(1, 13, dtype=float),  # seed
        np.geomspace(1.0, 50.0, 12),
    ]
    
    for guess in initial_guesses:
        if _time_left() < 30:
            break
        try:
            # Solve Dyson equilibrium
            dyson_roots = _newton_dyson(guess)
            if np.all(dyson_roots > 0) and np.all(np.isfinite(dyson_roots)):
                # Sort and compute score
                dyson_roots = np.sort(dyson_roots)
                score = _compute_c4_bound(dyson_roots)
                if score < best_score:
                    best_score = score
                    best_roots = dyson_roots.copy()
        except Exception:
            continue
    
    # Phase 2: Direct optimization from best Dyson solution
    if best_roots is not None and _time_left() > 60:
        try:
            log_roots_init = np.log(best_roots)
            result = minimize(
                _objective_direct,
                x0=log_roots_init,
                method='L-BFGS-B',
                bounds=[(np.log(0.01), np.log(1000.0))] * 12,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_score:
                best_roots = np.exp(result.x)
                best_score = result.fun
        except Exception:
            pass
    
    # Phase 3: Multi-start direct optimization
    if _time_left() > 120:
        np.random.seed(12345)
        start_points = []
        
        # Geometric sequences
        for r0 in [0.5, 1.0, 2.0, 5.0, 10.0]:
            for q in [1.2, 1.5, 1.8, 2.0, 2.5]:
                roots = r0 * (q ** np.arange(12))
                if roots[-1] < 500 and roots[0] > 0.01:
                    start_points.append(np.log(roots))
        
        # Random starts
        for _ in range(15):
            roots = np.sort(np.random.uniform(0.5, 60.0, 12))
            start_points.append(np.log(roots))
        
        # Dyson-based starts with perturbation
        for guess in initial_guesses[:3]:
            if _time_left() < 60:
                break
            try:
                dr = _newton_dyson(guess)
                if np.all(dr > 0):
                    for scale in [0.8, 1.0, 1.2]:
                        perturbed = dr * scale + np.random.normal(0, 0.1, 12)
                        perturbed = np.sort(perturbed)
                        if np.all(perturbed > 0.01):
                            start_points.append(np.log(perturbed))
            except:
                continue
        
        for sp in start_points:
            if _time_left() < 30:
                break
            try:
                result = minimize(
                    _objective_direct,
                    x0=np.array(sp),
                    method='L-BFGS-B',
                    bounds=[(np.log(0.01), np.log(1000.0))] * 12,
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                if result.fun < best_score:
                    best_roots = np.exp(result.x)
                    best_score = result.fun
            except Exception:
                continue
    
    # Phase 4: Final refinement
    if best_roots is not None and _time_left() > 20:
        try:
            # One more Dyson solve from current best
            refined = _newton_dyson(best_roots, max_iter=100, tol=1e-14)
            if np.all(refined > 0) and np.all(np.isfinite(refined)):
                refined = np.sort(refined)
                score = _compute_c4_bound(refined)
                if score < best_score:
                    best_roots = refined
                    best_score = score
            
            # Final L-BFGS polish
            log_roots = np.log(best_roots)
            result = minimize(
                _objective_direct,
                x0=log_roots,
                method='L-BFGS-B',
                bounds=[(np.log(0.001), np.log(5000.0))] * 12,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if result.fun < best_score:
                best_roots = np.exp(result.x)
                best_score = result.fun
        except Exception:
            pass
    
    if best_roots is None:
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Ensure sorted, positive
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    
    return [float(r) for r in best_roots]
