import numpy as np
from scipy.optimize import minimize, differential_evolution, least_squares
import time

def _get_hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial Hn(x) as list [c0, c1, ..., cn] for c0 + c1*x + ... + cn*x^n."""
    # H0 = 1, H1 = 2x, H_{n+1} = 2x*Hn - 2n*H_{n-1}
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    
    H = [[1.0], [0.0, 2.0]]
    for k in range(2, n + 1):
        prev1 = H[k - 1]  # H_{k-1}
        prev2 = H[k - 2]  # H_{k-2}
        curr = [0.0] * (k + 1)
        # 2x * H_{k-1}: shift prev1 by 1 and multiply by 2
        for i in range(k):
            curr[i + 1] += 2.0 * prev1[i]
        # -2(k-1) * H_{k-2}
        for i in range(k - 1):
            curr[i] += -2.0 * (k - 1) * prev2[i]
        H.append(curr)
    return H[n]

# Precompute Hermite coefficients
_H0 = _get_hermite_coeffs(0)
_H4 = _get_hermite_coeffs(4)
_H8 = _get_hermite_coeffs(8)
_H12 = _get_hermite_coeffs(12)

# Values at x=0
_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]

# Build the linear map from (c0, c1, c2, c3) to g's monomial coefficients
# g = c0*H0 + c1*H4 + c2*H8 + c3*H12
# We want to express q = g/x^2 in terms of roots.
# q has degree 10 (since g has degree 12 and g(0)=g'(0)=0).
# 
# Key insight: The space of valid g polynomials is 3-dimensional (c0,c1,c2 free, c3 determined).
# q = g/x^2 is degree 10, but constrained to lie in a 3-dim subspace.
# 
# Instead of searching over (c0,c1,c2), we can search over the 10 roots of q,
# then project back to find the nearest valid (c0,c1,c2).

# Build matrix: columns are [H0, H4, H8, H12] coefficients (degree 0 to 12)
# Row i gives the coefficient of x^i in each Hermite polynomial
_H_matrix = np.zeros((13, 4))
for i, H in enumerate([_H0, _H4, _H8, _H12]):
    for j in range(len(H)):
        _H_matrix[j, i] = H[j]

# The constraint g(0) = 0 means: c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
# This is automatically handled by c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0)/H12_0

# For q = g/x^2 to be valid, we need g[0] = 0 and g[1] = 0.
# g[0] = c0*H0[0] + c1*H4[0] + c2*H8[0] + c3*H12[0] = 0 (by construction of c3)
# g[1] = c0*H0[1] + c1*H4[1] + c2*H8[1] + c3*H12[1] = 0 (additional constraint!)

# Wait - H0 = 1, so H0[1] = 0. Let me check H4[1] and H8[1].
# H4(x) = 16x^4 - 48x^2 + 12, so H4[1] = 0
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 112, so H8[1] = 0
# H12(x) has only even powers, so H12[1] = 0
# So g[1] = 0 is automatically satisfied!

# Therefore q = g/x^2 is a degree-10 polynomial whose coefficients lie in a 3-dim subspace.
# We can parameterize q by its 10 roots and a scale factor, then find the best (c0,c1,c2).

# Actually, let's think differently. Since q is degree 10 with 11 coefficients,
# and it lies in a 3-dim subspace, we have 8 constraints on the coefficients.
# 
# Better approach: search over (c0, c1, c2) directly but use a smarter objective
# that works with the roots.

def _compute_value(c0, c1, c2):
    """Compute the value for given coefficients."""
    # Build g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    max_deg = 12
    g_coeffs = [0.0] * (max_deg + 1)
    
    for coeff, H in [(c0, _H0), (c1, _H4), (c2, _H8)]:
        for i in range(len(H)):
            g_coeffs[i] += coeff * H[i]
    
    # Compute c3 so that g(0) = 0
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    for i in range(len(_H12)):
        g_coeffs[i] += c3 * _H12[i]
    
    # Check if g tends to negative at infinity
    lead_coeff = g_coeffs[12]
    if lead_coeff < 0:
        g_coeffs = [-c for c in g_coeffs]
    
    # q = g / x^2
    if abs(g_coeffs[0]) > 1e-15 or abs(g_coeffs[1]) > 1e-15:
        return float('inf')
    
    q_coeffs = g_coeffs[2:]  # Divide by x^2
    
    # Find largest sign-changing root
    if len(q_coeffs) == 0:
        return float('inf')
    
    coeffs_desc = list(reversed(q_coeffs))
    roots = np.roots(coeffs_desc)
    
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
    
    # For each positive real root, check if sign changes
    poly = np.poly1d(coeffs_desc)
    
    largest = None
    for r in real_roots:
        if r <= 0:
            continue
        # Check sign change: evaluate at r - eps and r + eps
        eps = 1e-8 * max(1.0, abs(r))
        val_left = poly(r - eps)
        val_right = poly(r + eps)
        if val_left * val_right < 0:
            # Sign change at r
            if largest is None or r > largest:
                largest = r
    
    if largest is None:
        return float('inf')
    
    return largest ** 2 / (2 * np.pi)

def _roots_to_coefficients(roots, scale=1.0):
    """Convert roots + scale to descending-order coefficients of a polynomial."""
    # roots are in ascending order for np.poly
    coeffs = np.poly(roots)
    return scale * coeffs

def _coefficients_to_c(q_coeffs_desc):
    """
    Given q coefficients in descending order (degree 10), find (c0, c1, c2) 
    such that q = (c0*H0 + c1*H4 + c2*H8 + c3*H12) / x^2.
    
    q_coeffs_desc[0] is the coefficient of x^10, etc.
    g_coeffs[i] = q_coeffs_desc[10-i] for i >= 2, and g_coeffs[0] = g_coeffs[1] = 0.
    """
    # Build g coefficients from q
    g_coeffs = np.zeros(13)
    for i in range(11):
        g_coeffs[i + 2] = q_coeffs_desc[i]
    
    # Now solve: g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # This is a linear system. But we have 13 equations and 4 unknowns.
    # The system is overdetermined, so we use least squares.
    # However, since the Hermite polynomials span a specific subspace, 
    # we should project onto that subspace.
    
    # A better approach: use the 3 free parameters (c0, c1, c2) with c3 determined.
    # We want to find (c0, c1, c2) that best matches g.
    
    # Build the design matrix for (c0, c1, c2, c3)
    A = np.zeros((13, 4))
    for i, H in enumerate([_H0, _H4, _H8, _H12]):
        for j in range(len(H)):
            A[j, i] = H[j]
    
    # Constraint: c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0)/H12_0
    # Substitute: c3 = -alpha*c0 - beta*c1 - gamma*c2
    alpha = _H0_0 / _H12_0
    beta = _H4_0 / _H12_0
    gamma = _H8_0 / _H12_0
    
    # New design matrix with c3 eliminated
    # g = c0*(H0 - alpha*H12) + c1*(H4 - beta*H12) + c2*(H8 - gamma*H12)
    B = np.zeros((13, 3))
    for i in range(13):
        B[i, 0] = _H0[i] - alpha * _H12[i] if i < len(_H12) else _H0[i]
        B[i, 1] = _H4[i] - beta * _H12[i] if i < len(_H12) else _H4[i]
        B[i, 2] = _H8[i] - gamma * _H12[i] if i < len(_H12) else _H8[i]
    
    # Solve least squares: B @ [c0, c1, c2]^T ≈ g_coeffs
    result, residuals, rank, sv = np.linalg.lstsq(B, g_coeffs, rcond=None)
    
    c0, c1, c2 = result[0], result[1], result[2]
    
    # Check if this is a valid representation (residual should be small)
    predicted_g = B @ result
    residual = np.linalg.norm(predicted_g - g_coeffs)
    
    return c0, c1, c2, residual

def _compute_value_fast(c0, c1, c2):
    """Faster version using precomputed values where possible."""
    # g(0) = 0 constraint gives c3
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # Build g coefficients
    g = [0.0] * 13
    for i in range(len(_H0)):
        g[i] += c0 * _H0[i]
    for i in range(len(_H4)):
        g[i] += c1 * _H4[i]
    for i in range(len(_H8)):
        g[i] += c2 * _H8[i]
    for i in range(len(_H12)):
        g[i] += c3 * _H12[i]
    
    # Check leading coefficient
    if g[12] < 0:
        g = [-c for c in g]
    
    # q = g / x^2
    if abs(g[0]) > 1e-15 or abs(g[1]) > 1e-15:
        return float('inf')
    
    q = g[2:]
    coeffs_desc = list(reversed(q))
    roots = np.roots(coeffs_desc)
    
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    if not real_roots:
        return float('inf')
    
    poly = np.poly1d(coeffs_desc)
    largest = None
    for r in real_roots:
        eps = 1e-8 * max(1.0, abs(r))
        if poly(r - eps) * poly(r + eps) < 0:
            if largest is None or r > largest:
                largest = r
    
    if largest is None:
        return float('inf')
    
    return largest ** 2 / (2 * np.pi)

def _root_based_objective(root_params):
    """
    Objective function in root space.
    root_params = [r1, r2, ..., r10, log_scale]
    where r1..r10 are the roots of q and log_scale = log(leading_coeff).
    
    We construct q from these roots, project back to (c0, c1, c2),
    and compute the actual value.
    """
    try:
        roots = root_params[:10]
        log_scale = root_params[10]
        scale = np.exp(log_scale)
        
        # Build q coefficients from roots
        q_coeffs_desc = np.poly(roots) * scale
        
        # Project to (c0, c1, c2)
        c0, c1, c2, residual = _coefficients_to_c(q_coeffs_desc)
        
        # If residual is too large, the projection is poor
        if residual > 1e-6 * np.linalg.norm(q_coeffs_desc):
            return 1e10 + residual
        
        # Compute the actual value
        val = _compute_value_fast(c0, c1, c2)
        return val
    except:
        return 1e10

def _smooth_objective(x, beta=20.0):
    """Smooth surrogate using log-sum-exp over sign-changing roots.
    
    For each sign-changing root r_i, we have a 'cost' of r_i^2 / (2*pi).
    The true objective is max_i cost_i. We approximate with:
        (1/beta) * log(sum_i exp(beta * cost_i))
    which converges to max as beta -> inf but is differentiable.
    """
    c0, c1, c2 = x
    try:
        c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
        
        g = [0.0] * 13
        for i in range(len(_H0)):
            g[i] += c0 * _H0[i]
        for i in range(len(_H4)):
            g[i] += c1 * _H4[i]
        for i in range(len(_H8)):
            g[i] += c2 * _H8[i]
        for i in range(len(_H12)):
            g[i] += c3 * _H12[i]
        
        if g[12] < 0:
            g = [-c for c in g]
        
        if abs(g[0]) > 1e-15 or abs(g[1]) > 1e-15:
            return 1e10
        
        q = g[2:]
        coeffs_desc = list(reversed(q))
        roots = np.roots(coeffs_desc)
        
        sign_changing_costs = []
        poly = np.poly1d(coeffs_desc)
        
        for r in roots:
            if abs(r.imag) < 1e-10 and r.real > 0:
                rr = r.real
                eps = 1e-8 * max(1.0, abs(rr))
                if poly(rr - eps) * poly(rr + eps) < 0:
                    cost = rr ** 2 / (2.0 * np.pi)
                    sign_changing_costs.append(cost)
        
        if not sign_changing_costs:
            return 1e10
        
        costs = np.array(sign_changing_costs)
        # Log-sum-exp for numerical stability
        max_cost = np.max(costs)
        log_sum_exp = max_cost + np.log(np.sum(np.exp(beta * (costs - max_cost))))
        return log_sum_exp / beta
    except:
        return 1e10

def _get_seed_roots(c0, c1, c2):
    """Get the roots of q for a given (c0, c1, c2) as initial root parameters."""
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    g = [0.0] * 13
    for i in range(len(_H0)):
        g[i] += c0 * _H0[i]
    for i in range(len(_H4)):
        g[i] += c1 * _H4[i]
    for i in range(len(_H8)):
        g[i] += c2 * _H8[i]
    for i in range(len(_H12)):
        g[i] += c3 * _H12[i]
    
    if g[12] < 0:
        g = [-c for c in g]
    
    q = g[2:]
    coeffs_desc = list(reversed(q))
    roots = np.roots(coeffs_desc)
    
    # Extract real roots and pair complex conjugates
    real_roots = []
    complex_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
        else:
            complex_roots.append(r)
    
    # We need exactly 10 roots. If we have complex pairs, we keep them as is
    # but for the root-based optimization, we'll work with all 10 roots.
    
    # For the parameterization, we use all 10 roots (real and complex)
    # but we only optimize over real roots that are positive.
    
    # Actually, let's just return all 10 roots and the log of the leading coefficient
    leading_coeff = coeffs_desc[0]
    log_scale = np.log(abs(leading_coeff))
    
    return roots, log_scale

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s
    
    # Seed
    best_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_val = _compute_value(best_c[0], best_c[1], best_c[2])
    
    def objective(x):
        try:
            return _compute_value_fast(x[0], x[1], x[2])
        except:
            return 1e10
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            de_result = differential_evolution(
                objective, bounds, maxiter=100, popsize=15,
                tol=1e-8, seed=42, polish=False
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
        except:
            pass
    
    # Phase 2: Multi-start Nelder-Mead for local refinement
    np.random.seed(42)
    num_starts = 30
    for i in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            x0 = best_c.copy()
        else:
            # Sample around best and random
            if i % 3 == 0:
                x0 = best_c + np.random.normal(0, 0.5, size=3)
            elif i % 3 == 1:
                x0 = np.random.uniform(-3, 3, size=3)
            else:
                x0 = best_c + np.random.normal(0, 0.1, size=3)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass
    
    # Phase 3: Root-based optimization
    # Get initial roots from the best solution
    if time.time() < deadline - 25:
        try:
            roots_init, log_scale_init = _get_seed_roots(best_c[0], best_c[1], best_c[2])
            
            # Build initial root parameters
            root_params_init = np.zeros(11)
            for i in range(10):
                if abs(roots_init[i].imag) < 1e-10:
                    root_params_init[i] = roots_init[i].real
                else:
                    # For complex roots, we'll keep them fixed and only optimize real roots
                    # This is a simplification - in practice, we might want to handle this differently
                    root_params_init[i] = roots_init[i].real
            root_params_init[10] = log_scale_init
            
            # Optimize over root parameters
            # We'll only optimize over the real positive roots
            # For simplicity, let's try optimizing all 10 roots + scale
            
            # Define bounds for roots (positive real roots) and scale
            root_bounds = [(-10, 10)] * 10 + [(-5, 5)]
            
            de_root_result = differential_evolution(
                _root_based_objective, root_bounds, maxiter=50, popsize=10,
                tol=1e-8, seed=42, polish=False
            )
            
            if de_root_result.fun < best_val:
                # Extract the best root parameters and convert back to (c0, c1, c2)
                best_root_params = de_root_result.x
                roots_opt = best_root_params[:10]
                log_scale_opt = best_root_params[10]
                scale_opt = np.exp(log_scale_opt)
                
                q_coeffs_desc = np.poly(roots_opt) * scale_opt
                c0_opt, c1_opt, c2_opt, residual = _coefficients_to_c(q_coeffs_desc)
                
                if residual < 1e-4:
                    val_opt = _compute_value_fast(c0_opt, c1_opt, c2_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_c = np.array([c0_opt, c1_opt, c2_opt])
        except:
            pass
    
    # Phase 4: Smooth surrogate refinement with log-sum-exp
    if time.time() < deadline - 20:
        for beta in [10.0, 20.0, 50.0, 100.0]:
            if time.time() > deadline - 15:
                break
            try:
                result = minimize(lambda x: _smooth_objective(x, beta), best_c,
                                 method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True})
                # Verify with exact objective
                val = objective(result.x)
                if val < best_val:
                    best_val = val
                    best_c = result.x.copy()
            except:
                pass
    
    # Phase 5: High-precision polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            result = minimize(objective, best_c, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass
    
    # Final verification
    final_val = _compute_value(best_c[0], best_c[1], best_c[2])
    if final_val < best_val:
        best_val = final_val
    
    return best_c
