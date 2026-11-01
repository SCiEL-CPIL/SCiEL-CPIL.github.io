import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

START_TIME = time.time()
TIME_LIMIT = 195.0

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Precompute moments of e^{-x^2}
    # I[n] = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
    # Using log to avoid overflow for large n
    sp = np.sqrt(np.pi)
    max_n = 24 # Need up to x^24 (from H12 * H12 * x^4)
    I = np.zeros(max_n + 1)
    for n in range(max_n + 1):
        if n == 0:
            I[n] = sp
        else:
            I[n] = I[n-1] * (2*n - 1) / 2.0

    def compute_bound_and_grad(c0, c4, c8):
        c12 = -(c0 + 3.0*c4 + 105.0*c8) / 945.0
        
        # Coefficients of P(x) = sum p_k x^{2k}
        # H0 = 1
        # H4 = x^4 - 6x^2 + 3
        # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
        # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
        
        p = np.zeros(7)
        p[0] = c0 + 3.0*c4 + 105.0*c8 + 945.0*c12
        p[1] = -6.0*c4 - 420.0*c8 - 7560.0*c12
        p[2] = c4 + 210.0*c8 + 13860.0*c12
        p[3] = -28.0*c8 - 6930.0*c12
        p[4] = c8 + 1155.0*c12
        p[5] = -66.0*c12
        p[6] = c12
        
        # Moments m_k = integral x^k |P|^2 e^{-x^2} dx
        # m_k = sum_{i,j} p_i p_j I[k+i+j]
        # We need m0, m2, m4
        m0 = 0.0
        m2 = 0.0
        m4 = 0.0
        
        # Precompute sums to speed up
        # s_k[i] = sum_j p_j I[k+i+j]
        # Then m_k = sum_i p_i s_k[i]
        
        # Since p is small (7 elements), direct loops are fine but we can optimize.
        
        # Let's compute s0, s2, s4 vectors
        # s_k[i] = sum_{j=0}^6 p_j I[k+i+j]
        
        s0 = np.zeros(7)
        s2 = np.zeros(7)
        s4 = np.zeros(7)
        
        for i in range(7):
            for j in range(7):
                val = p[j]
                if val == 0: continue
                idx0 = i + j
                idx2 = i + j + 1
                idx4 = i + j + 2
                if idx0 <= max_n: s0[i] += val * I[idx0]
                if idx2 <= max_n: s2[i] += val * I[idx2]
                if idx4 <= max_n: s4[i] += val * I[idx4]
                
        m0 = np.dot(p, s0)
        m2 = np.dot(p, s2)
        m4 = np.dot(p, s4)
        
        # Fourier side
        # Q(xi) = P(-i xi) = sum p_k (-1)^k xi^{2k}
        q = p * np.array([1, -1, 1, -1, 1, -1, 1])
        
        # n_k = 2*pi * integral xi^k |Q|^2 e^{-xi^2} dxi
        # Same structure as m_k but with q
        t0 = np.zeros(7)
        t2 = np.zeros(7)
        t4 = np.zeros(7)
        
        for i in range(7):
            for j in range(7):
                val = q[j]
                if val == 0: continue
                idx0 = i + j
                idx2 = i + j + 1
                idx4 = i + j + 2
                if idx0 <= max_n: t0[i] += val * I[idx0]
                if idx2 <= max_n: t2[i] += val * I[idx2]
                if idx4 <= max_n: t4[i] += val * I[idx4]
                
        n0 = 2.0 * np.pi * np.dot(q, t0)
        n2 = 2.0 * np.pi * np.dot(q, t2)
        n4 = 2.0 * np.pi * np.dot(q, t4)
        
        if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
            return 1e18, np.array([1e18, 1e18, 1e18])
            
        vf = m2 / m0
        vF = n2 / n0
        
        if vf <= 0 or vF <= 0:
            return 1e18, np.array([1e18, 1e18, 1e18])
            
        bound = np.sqrt(vf * vF)
        
        # Gradient
        # d(bound)/dc = (1/(2*bound)) * (vF * d(vf)/dc + vf * d(vF)/dc)
        # d(vf)/dc = (m2' m0 - m2 m0') / m0^2
        # d(vF)/dc = (n2' n0 - n2 n0') / n0^2
        
        # Derivatives of p with respect to c0, c4, c8
        # c12 = -(c0 + 3c4 + 105c8)/945
        # dp_k/dc = dp_k/dc_direct + dp_k/dc12 * dc12/dc
        
        # dc12/dc = [-1/945, -3/945, -105/945]
        dc12_dc = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
        
        # dp_k/dc_direct
        # p0: c0, 3c4, 105c8, 945c12 -> [1, 3, 105]
        # p1: -6c4, -420c8, -7560c12 -> [0, -6, -420]
        # p2: c4, 210c8, 13860c12 -> [0, 1, 210]
        # p3: -28c8, -6930c12 -> [0, 0, -28]
        # p4: c8, 1155c12 -> [0, 0, 1]
        # p5: -66c12 -> [0, 0, 0]
        # p6: c12 -> [0, 0, 0]
        
        dp_direct = np.array([
            [1.0, 3.0, 105.0],
            [0.0, -6.0, -420.0],
            [0.0, 1.0, 210.0],
            [0.0, 0.0, -28.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        # dp_k/dc12
        dp_c12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
        
        grad = np.zeros(3)
        
        for c in range(3):
            dp_dc = dp_direct[:, c] + dp_c12 * dc12_dc[c]
            
            # dm_k/dc = 2 * sum_i dp_dc[i] * s_k[i]
            dm0 = 2.0 * np.dot(dp_dc, s0)
            dm2 = 2.0 * np.dot(dp_dc, s2)
            dm4 = 2.0 * np.dot(dp_dc, s4)
            
            # dq_dc = dp_dc * (-1)^k
            dq_dc = dp_dc * np.array([1, -1, 1, -1, 1, -1, 1])
            
            # dn_k/dc = 2*pi * 2 * sum_i dq_dc[i] * t_k[i]
            dn0 = 2.0 * 2.0 * np.pi * np.dot(dq_dc, t0)
            dn2 = 2.0 * 2.0 * np.pi * np.dot(dq_dc, t2)
            dn4 = 2.0 * 2.0 * np.pi * np.dot(dq_dc, t4)
            
            d_vf = (dm2 * m0 - m2 * dm0) / (m0 * m0)
            d_vF = (dn2 * n0 - n2 * dn0) / (n0 * n0)
            
            grad[c] = (vF * d_vf + vf * d_vF) / (2.0 * bound)
            
        return bound, grad

    def objective_and_grad(x):
        return compute_bound_and_grad(x[0], x[1], x[2])
    
    def objective(x):
        return objective_and_grad(x)[0]

    # Initial guess from seed
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Strategy:
    # 1. Differential Evolution on (u, v) where u=c4/c0, v=c8/c0.
    #    This removes scale invariance.
    #    c0 is free, but bound is scale invariant?
    #    A(f) A(F) is scale invariant for the product?
    #    A(cf) = A(f). So yes, the bound depends only on the shape, i.e., ratios.
    #    So we can set c0 = 1 (or -1) and optimize u, v.
    #    Wait, c0 can be 0. But generically c0 != 0.
    #    Let's try DE on (u, v) with c0=1.
    
    def de_objective(params):
        u, v = params
        c0 = 1.0
        c4 = u * c0
        c8 = v * c0
        val, _ = compute_bound_and_grad(c0, c4, c8)
        return val

    # Bounds for u, v
    # From seed: c0=-1.13, c4=0.04, c8=0.000308
    # u = c4/c0 = -0.035
    # v = c8/c0 = -0.00027
    # Let's use reasonable bounds.
    bounds_de = [(-1.0, 1.0), (-0.01, 0.01)]
    
    best_val = np.inf
    best_coeffs = seed_coeffs.copy()
    
    # Check seed
    val, _ = compute_bound_and_grad(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_val = val
    
    # Run Differential Evolution
    if time.time() < TIME_LIMIT - 30:
        try:
            # Use a fixed seed for reproducibility
            np.random.seed(42)
            res_de = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-10,
                polish=False # We'll polish later
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                u, v = res_de.x
                # Reconstruct c0, c4, c8
                # We need to find the scale c0 that minimizes? No, scale invariant.
                # But the evaluator expects specific c0, c4, c8.
                # The score is the value of the bound.
                # We can return any scaling. Let's return the one from DE (c0=1).
                best_coeffs = np.array([1.0, u, v])
        except Exception:
            pass

    # Local Polish with L-BFGS-B on (c0, c4, c8)
    # Start from best_coeffs found by DE (or seed if DE failed)
    # Also start from seed
    
    starts = [best_coeffs, seed_coeffs]
    
    # Add some random perturbations around best_coeffs
    for _ in range(5):
        pert = np.random.normal(0, 0.1, size=3)
        starts.append(best_coeffs + pert)
        
    for start in starts:
        if time.time() > TIME_LIMIT - 20:
            break
        try:
            # Use bounds to keep things sane
            # c0 can be large/small, but let's constrain it to avoid numerical issues
            # Actually, since it's scale invariant, we can fix c0=1 and optimize c4, c8?
            # No, let's stick to 3D for robustness.
            bounds_lbfgs = [(-100.0, 100.0), (-10.0, 10.0), (-1.0, 1.0)]
            
            res = minimize(
                lambda x: objective_and_grad(x)[0],
                start,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=bounds_lbfgs,
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Final Polish with Newton or tight L-BFGS
    if time.time() < TIME_LIMIT - 10:
        try:
            res = minimize(
                lambda x: objective_and_grad(x)[0],
                best_coeffs,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=[(-1000.0, 1000.0), (-100.0, 100.0), (-10.0, 10.0)],
                options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return best_coeffs
