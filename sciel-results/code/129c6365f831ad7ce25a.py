"""Optimized program: analytic exactness + refined multi-stage global search."""
import numpy as np
import time
import scipy.optimize
import math
import itertools

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    Returns the product sqrt(<x^2> * <xi^2>).
    """
    # Hermite values at 0: H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=1330
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of the polynomial P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # P(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    # a0 = c0 + 12c4 + 168c8 + 1330c12 = 0
    a0 = 0.0
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12
    
    # Vector of coefficients for x^0, x^2, ..., x^12
    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    
    # Compute Norm^2 = p^T G p
    # G_{ij} = sqrt(pi) * 2^(i+j) * (i+j)!
    # We can precompute the factorials and powers
    n = len(p)
    G = np.zeros((n, n))
    Gx = np.zeros((n, n))
    
    # Precompute factorials up to 2*n
    fact = [1] * (2*n + 1)
    for i in range(1, 2*n + 1):
        fact[i] = fact[i-1] * i
    
    for i in range(n):
        for j in range(n):
            k = i + j
            G[i, j] = math.sqrt(math.pi) * (2 ** k) * fact[k]
            Gx[i, j] = math.sqrt(math.pi) * (2 ** (k + 1)) * fact[k + 1]
            
    norm_sq = p @ G @ p
    if norm_sq <= 1e-15:
        return 1e10
        
    x2 = (p @ Gx @ p) / norm_sq
    
    # Compute Q(xi) coefficients
    # Q(xi) = c0 H0(xi) - c4 H4(xi) + c8 H8(xi) - c12 H12(xi)
    # H0(x) = 1
    # H4(x) = 12x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 168
    # H12(x) = 4096x^12 - 98304x^10 + 806400x^8 - 3225600x^6 + 5806080x^4 - 4651200x^2 + 1330
    
    q = np.zeros(7)
    # H0 term: c0
    q[0] += c0
    # H4 term: -c4 * H4
    # H4 coeffs: [12, -48, 12] for x^0, x^2, x^4
    q[0] -= c4 * 12
    q[1] -= c4 * (-48)
    q[2] -= c4 * 12
    # H8 term: c8 * H8
    # H8 coeffs: [168, -13440, 11520, -3072, 256] for x^0, x^2, x^4, x^6, x^8
    q[0] += c8 * 168
    q[1] += c8 * (-13440)
    q[2] += c8 * 11520
    q[3] += c8 * (-3072)
    q[4] += c8 * 256
    # H12 term: -c12 * H12
    # H12 coeffs: [1330, -4651200, 5806080, -3225600, 806400, -98304, 4096] for x^0, x^2, ..., x^12
    q[0] -= c12 * 1330
    q[1] -= c12 * (-4651200)
    q[2] -= c12 * 5806080
    q[3] -= c12 * (-3225600)
    q[4] -= c12 * 806400
    q[5] -= c12 * (-98304)
    q[6] -= c12 * 4096
    
    # Calculate xi^2 using the same G and Gx matrices
    # Note: The integral is over xi, and the weight is e^{-xi^2}.
    # The form is identical to the x case.
    norm_q_sq = q @ G @ q
    xi2 = (q @ Gx @ q) / norm_q_sq
    
    # The uncertainty product is sqrt(<x^2> * <xi^2>)
    # Note: The Fourier transform normalization might affect the constant.
    # Standard convention: F(f)(xi) = 1/sqrt(2pi) int f(x) e^{-ixxi} dx
    # Then <xi^2> is computed as above.
    # The product is sqrt(x2 * xi2).
    
    product = math.sqrt(x2 * xi2)
    
    return product

def _compute_grad(c0, c4, c8):
    """
    Compute the analytic gradient of the bound.
    """
    # We need d/dc [ sqrt( (p^T Gx p / p^T G p) * (q^T Gx q / q^T G q) ) ]
    # Let A = p^T Gx p, B = p^T G p, C = q^T Gx q, D = q^T G q
    # f = sqrt( (A/B) * (C/D) ) = (A C / (B D))^{1/4}
    # This is getting complicated. Let's use the chain rule on log(f) or just finite difference of the analytic function?
    # No, let's derive it properly.
    # Let u = A/B, v = C/D. f = sqrt(u v).
    # df/dc = (1/(2 sqrt(uv))) * (v du/dc + u dv/dc)
    
    # We need du/dc and dv/dc.
    # u = A/B. du/dc = (A' B - A B') / B^2
    # A = p^T Gx p. p depends on c.
    # p = M c + p0 (where c = [c0, c4, c8] and p12 is fixed by constraint? No, c12 is linear in c).
    # p is linear in c. Let p = L c.
    # Then A = c^T L^T Gx L c.
    # A' = 2 L^T Gx L c.
    # Similarly for B, C, D.
    
    # Construct L such that p = L c.
    # p[0] = 0
    # p[1] = -48 c4 - 1344 c8 - 32256 c12
    # ...
    # c12 = -(c0 + 12 c4 + 168 c8) / 1330
    
    # Let's just compute the gradient numerically using the analytic _compute_bound function.
    # It's fast enough and less error-prone than deriving the 4th root chain rule manually.
    # The previous numerical gradient was slow because _compute_bound was slow.
    # Now _compute_bound is fast (matrix multiplications).
    
    step = 1e-8
    f0 = _compute_bound(c0, c4, c8)
    grad = np.zeros(3)
    grad[0] = (_compute_bound(c0 + step, c4, c8) - _compute_bound(c0 - step, c4, c8)) / (2 * step)
    grad[1] = (_compute_bound(c0, c4 + step, c8) - _compute_bound(c0, c4 - step, c8)) / (2 * step)
    grad[2] = (_compute_bound(c0, c4 + step, c8) - _compute_bound(c0, c4 - step, c8)) / (2 * step)
    return grad

def _fixed_point_equations(x, t):
    """
    Fixed-point equations for homotopy continuation.
    """
    c0, c4, c8 = x
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    if t < 0.5:
        grad = _compute_grad(c0, c4, c8)
        return (1 - 2*t) * (x - seed) - 2*t * grad
    else:
        grad = _compute_grad(c0, c4, c8)
        return (1 - t) * (x - seed) - t * grad

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180s limit per task contract
    
    # Initial guess: Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    def _objective(x):
        return _compute_bound(x[0], x[1], x[2])
    
    def _grad(x):
        return _compute_grad(x[0], x[1], x[2])
    
    # 1. Local Optimization from Seed using SLSQP
    try:
        res_local = scipy.optimize.minimize(
            _objective,
            seed_coeffs,
            method='SLSQP',
            jac=_grad,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if res_local.success and res_local.fun < best_val:
            best_val = res_local.fun
            best_coeffs = res_local.x.copy()
    except Exception:
        pass
    
    # 2. Homotopy Continuation Fixed-Point Solve
    if time.time() < deadline - 60.0:
        try:
            num_steps = 20
            t_values = np.linspace(0.0, 1.0, num_steps)
            current_x = best_coeffs.copy()
            
            for t in t_values:
                if time.time() > deadline - 30.0:
                    break
                try:
                    sol = scipy.optimize.fsolve(
                        _fixed_point_equations,
                        current_x,
                        args=(t,),
                        full_output=True
                    )
                    x_sol, info, ier, msg = sol
                    if ier == 1:
                        current_x = x_sol
                        val = _compute_bound(current_x[0], current_x[1], current_x[2])
                        if val < best_val:
                            best_val = val
                            best_coeffs = current_x.copy()
                except Exception:
                    pass
            
            if time.time() < deadline - 20.0:
                try:
                    sol_final = scipy.optimize.fsolve(
                        _fixed_point_equations,
                        best_coeffs,
                        args=(1.0,),
                        full_output=True
                    )
                    x_final, info, ier, msg = sol_final
                    if ier == 1:
                        val_final = _compute_bound(x_final[0], x_final[1], x_final[2])
                        if val_final < best_val:
                            best_val = val_final
                            best_coeffs = x_final.copy()
                except Exception:
                    pass
        except Exception:
            pass
    
    # 3. Global Search using Differential Evolution
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    if time.time() < deadline - 40.0:
        try:
            de_res = scipy.optimize.differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                polish=False
            )
            
            if de_res.success or (de_res.fun < best_val):
                candidate = de_res.x
                val_candidate = de_res.fun
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_coeffs = candidate.copy()
        except Exception:
            pass
    
    # 4. Multi-start Local Optimization
    if time.time() < deadline - 15.0:
        np.random.seed(123)
        for _ in range(8):
            if time.time() > deadline - 10.0:
                break
            noise = np.random.normal(0, 0.5, 3)
            start_point = best_coeffs + noise
            try:
                res_restart = scipy.optimize.minimize(
                    _objective,
                    start_point,
                    method='SLSQP',
                    jac=_grad,
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                if res_restart.success and res_restart.fun < best_val:
                    best_val = res_restart.fun
                    best_coeffs = res_restart.x.copy()
            except Exception:
                pass

    # 5. Final High-Precision Polish
    if time.time() < deadline - 5.0:
        try:
            res_final = scipy.optimize.minimize(
                _objective,
                best_coeffs,
                method='SLSQP',
                jac=_grad,
                options={'maxiter': 2000, 'ftol': 1e-15}
            )
            if res_final.success and res_final.fun < best_val:
                best_val = res_final.fun
                best_coeffs = res_final.x.copy()
        except Exception:
            pass

    return best_coeffs
