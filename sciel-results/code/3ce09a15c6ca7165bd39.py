"""Structural change: root-space parameterization of q(x) with feasibility projection."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Precompute Hermite polynomial coefficients (highest degree first)
def _hermite_coeffs(n):
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    h_prev = np.array([1.0])
    h_curr = np.array([2.0, 0.0])
    for k in range(1, n):
        new_len = len(h_curr) + 1
        h_2x = np.zeros(new_len)
        h_2x[:-1] = 2.0 * h_curr
        h_2k = np.zeros(new_len)
        pad = new_len - len(h_prev)
        h_2k[pad:] = 2.0 * k * h_prev
        h_next = h_2x - h_2k
        h_prev = h_curr
        h_curr = h_next
    return h_curr

H0_c = _hermite_coeffs(0)
H4_c = _hermite_coeffs(4)
H8_c = _hermite_coeffs(8)
H12_c = _hermite_coeffs(12)

H0_0 = H0_c[-1]   # 1
H4_0 = H4_c[-1]   # 12
H8_0 = H8_c[-1]   # 20160
H12_0 = H12_c[-1] # 462

def _build_q_from_coeffs(c0, c1, c2):
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    g = np.zeros(13)
    for c, hc in [(c0, H0_c), (c1, H4_c), (c2, H8_c), (c3, H12_c)]:
        pad = 13 - len(hc)
        hc_p = np.concatenate([np.zeros(pad), hc])
        g += c * hc_p
    q = g[:-2]  # degree 10, length 11
    return q

def _compute_value(c0, c1, c2):
    q = _build_q_from_coeffs(c0, c1, c2)
    roots = np.roots(q)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    if len(real_roots) == 0:
        return float('inf')
    valid_roots = []
    for r in real_roots:
        eps = 1e-9 * max(1.0, abs(r))
        q_left = np.polyval(q, r - eps)
        q_right = np.polyval(q, r + eps)
        if q_left * q_right < 0:
            valid_roots.append(r)
    if not valid_roots:
        return float('inf')
    r = max(valid_roots)
    return r**2 / (2.0 * np.pi)

def _value_from_q(q):
    roots = np.roots(q)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    if len(real_roots) == 0:
        return float('inf')
    valid_roots = []
    for r in real_roots:
        eps = 1e-9 * max(1.0, abs(r))
        q_left = np.polyval(q, r - eps)
        q_right = np.polyval(q, r + eps)
        if q_left * q_right < 0:
            valid_roots.append(r)
    if not valid_roots:
        return float('inf')
    r = max(valid_roots)
    return r**2 / (2.0 * np.pi)

def _q_from_root_params(r1, r2, r3, r4, r5, r6, r7, r8, r9, r10):
    """Build q(x) from its 10 roots (real and complex pairs).
    We parameterize as:
    - 4 real roots: r1, r2, r3, r4
    - 3 complex pairs: (a5±b5i), (a6±b6i), (a7±b7i) -> 6 params
    Total: 10 params
    """
    roots = np.array([r1, r2, r3, r4,
                      r5 + 1j*r6, r5 - 1j*r6,
                      r7 + 1j*r8, r7 - 1j*r8,
                      r9 + 1j*r10, r9 - 1j*r10], dtype=complex)
    # Build polynomial from roots: q(x) = prod(x - ri)
    # Use numpy.poly (highest degree first)
    q = np.poly(roots).real  # Should be real if complex conjugates paired
    # Normalize: q should have degree 10
    if len(q) != 11:
        q = np.zeros(11)
        q[0] = 1.0
    return q

def _project_to_feasible(q):
    """Project q onto the feasible set: q must equal g/x^2 for some c0,c1,c2.
    The feasible set is a 3-dimensional affine subspace in the 11-dimensional
    space of degree-10 polynomials.
    
    We solve: find c0,c1,c2 that minimize ||q - q(c0,c1,c2)||^2
    This is a linear least-squares problem.
    """
    # Build the 11x3 matrix A where columns are q(1,0,0), q(0,1,0), q(0,0,1)
    A = np.zeros((11, 3))
    for i in range(3):
        coeffs = [0.0, 0.0, 0.0]
        coeffs[i] = 1.0
        A[:, i] = _build_q_from_coeffs(*coeffs)
    
    # Solve A @ c = q (least squares)
    c_res, _, _, _ = np.linalg.lstsq(A, q, rcond=None)
    q_proj = A @ c_res
    return q_proj

def _objective_root_space(params):
    """Objective in root space: project to feasible, compute value."""
    q_raw = _q_from_root_params(*params)
    q_feas = _project_to_feasible(q_raw)
    return _value_from_q(q_feas)

def construct():
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 210s limit
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_value(*seed)
    best_params = seed.copy()
    
    # Strategy 1: Direct coefficient-space optimization with multi-start Nelder-Mead
    np.random.seed(42)
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 20:
            break
        if i == 0:
            trial = seed.copy()
        else:
            trial = seed + np.random.normal(0, 0.1, 3)
        
        try:
            res = minimize(
                lambda p: _compute_value(*p),
                trial,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Root-space search
    if time.time() < deadline - 30:
        # Initialize root parameters from the best solution found so far
        q_best = _build_q_from_coeffs(*best_params)
        roots_best = np.roots(q_best)
        
        # Extract real roots and complex pair parameters
        real_roots = roots_best[np.abs(roots_best.imag) < 1e-8].real
        complex_roots = roots_best[np.abs(roots_best.imag) >= 1e-8]
        
        # Sort real roots
        real_roots = np.sort(real_roots)
        
        # Build initial root parameters
        root_params = np.zeros(10)
        # Fill in real roots
        for i in range(min(4, len(real_roots))):
            root_params[i] = real_roots[i]
        
        # Fill in complex pairs
        idx = 4
        for j in range(min(3, len(complex_roots) // 2)):
            z = complex_roots[j]
            root_params[idx] = z.real
            root_params[idx + 1] = z.imag
            idx += 2
        
        # Optimize in root space
        try:
            res_root = minimize(
                _objective_root_space,
                root_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_root.fun < best_val:
                best_val = res_root.fun
                # Project back to coefficient space
                q_opt = _q_from_root_params(*res_root.x)
                q_feas = _project_to_feasible(q_opt)
                A = np.zeros((11, 3))
                for i in range(3):
                    coeffs = [0.0, 0.0, 0.0]
                    coeffs[i] = 1.0
                    A[:, i] = _build_q_from_coeffs(*coeffs)
                c_res, _, _, _ = np.linalg.lstsq(A, q_feas, rcond=None)
                best_params = c_res
        except Exception:
            pass
    
    # Strategy 3: Fine-tune in coefficient space with many small perturbations
    if time.time() < deadline - 10:
        np.random.seed(123)
        for i in range(100):
            if time.time() > deadline - 10:
                break
            trial = best_params + np.random.normal(0, 1e-4, 3)
            v = _compute_value(*trial)
            if v < best_val:
                best_val = v
                best_params = trial.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda p: _compute_value(*p),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
