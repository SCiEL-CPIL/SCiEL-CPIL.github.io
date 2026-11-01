import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 195.0

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Precompute the norm and 4th-moment matrices exactly.
    # Q0 is diagonal: Q0[d,d] = sqrt(pi) * 2^d * d!
    # Q4: Q4[d_i, d_j] = int x^4 H_{d_i} H_{d_j} e^{-x^2} dx
    #     Using x^4 H_n expansion coefficients.
    
    def fact(n):
        r = 1
        for k in range(2, n + 1):
            r *= k
        return r
    
    # x^4 H_n = (1/16) H_{n+4} + (n+5/4) H_{n+2} + (7n^2/4 + 5n/2 + 1) H_n 
    #           + n(n-1)(n-2)(n-3)/16 H_{n-4}
    # (standard physicists' Hermite recurrence)
    
    def x4_coeffs(n):
        d = {}
        d[n + 4] = 1.0 / 16.0
        d[n + 2] = (n + 5.0 / 4.0)
        d[n] = (7.0 * n * n / 4.0) + (5.0 * n / 2.0) + 1.0
        if n >= 4:
            d[n - 4] = float(n * (n - 1) * (n - 2) * (n - 3)) / 16.0
        return d
    
    degrees = [0, 4, 8, 12]
    nd = len(degrees)
    deg_set = {d: i for i, d in enumerate(degrees)}
    
    Q0 = np.zeros((nd, nd))
    for i, d in enumerate(degrees):
        Q0[i, i] = np.sqrt(np.pi) * (2.0 ** d) * fact(d)
    
    Q4 = np.zeros((nd, nd))
    for i, di in enumerate(degrees):
        for m, a in x4_coeffs(di).items():
            if m in deg_set:
                j = deg_set[m]
                Q4[i, j] = a * Q0[j, j]
    Q4 = 0.5 * (Q4 + Q4.T)
    
    def c12_of(c0, c4, c8):
        return -(c0 + 3.0 * c4 + 105.0 * c8) / 945.0
    
    def obj_and_grad(x):
        if time.time() > TIME_LIMIT - 10:
            return 1e30, np.zeros(3)
        c0, c4, c8 = x
        c12 = c12_of(c0, c4, c8)
        c = np.array([c0, c4, c8, c12])
        N = float(c @ Q0 @ c)
        if N <= 1e-300:
            return 1e30, np.zeros(3)
        J = float(c @ Q4 @ c)
        S = J / (N * N)
        
        dN = 2.0 * (Q0 @ c)
        dJ = 2.0 * (Q4 @ c)
        dS = dJ / (N * N) - 2.0 * J * dN / (N * N * N)
        
        g = np.empty(3)
        g[0] = dS[0] + dS[3] * (-1.0 / 945.0)
        g[1] = dS[1] + dS[3] * (-3.0 / 945.0)
        g[2] = dS[2] + dS[3] * (-105.0 / 945.0)
        return S, g
    
    # Seed
    x_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val, _ = obj_and_grad(x_seed)
    best_x = x_seed.copy()
    
    rng = np.random.default_rng(12345)
    
    # A collection of starting points: seed + scaled perturbations + random
    starts = [x_seed.copy()]
    for scale in [1e-3, 1e-2, 5e-2, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(3):
            x0 = x_seed + scale * rng.standard_normal(3)
            # clamp to reasonable box to avoid absurd values
            x0[0] = np.clip(x0[0], -50.0, 50.0)
            x0[1] = np.clip(x0[1], -5.0, 5.0)
            x0[2] = np.clip(x0[2], -0.5, 0.5)
            starts.append(x0)
    # a few fully random starts
    for _ in range(6):
        x0 = np.array([
            rng.uniform(-50.0, 50.0),
            rng.uniform(-5.0, 5.0),
            rng.uniform(-0.5, 0.5),
        ])
        starts.append(x0)
    
    bounds = [(-50.0, 50.0), (-5.0, 5.0), (-0.5, 0.5)]
    
    def polish(x0, tol, maxiter, tag):
        nonlocal best_val, best_x
        if time.time() > TIME_LIMIT - 5:
            return
        try:
            res = minimize(
                lambda z: obj_and_grad(z)[0],
                x0,
                jac=lambda z: obj_and_grad(z)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol * 1e-2, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Pass 1: coarse multi-start
    for x0 in starts:
        if time.time() > TIME_LIMIT - 15:
            break
        try:
            res = minimize(
                lambda z: obj_and_grad(z)[0],
                x0,
                jac=lambda z: obj_and_grad(z)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Pass 2: tight polish from current best and from a few nearby points
    polish(best_x, 1e-15, 1500, 'polish1')
    if time.time() < TIME_LIMIT - 10:
        for scale in [1e-4, 1e-5, 1e-6]:
            x0 = best_x + scale * rng.standard_normal(3)
            polish(x0, 1e-15, 1500, f'polish_s{scale}')
            if time.time() > TIME_LIMIT - 10:
                break
    
    # Pass 3: ultra-tight final polish
    polish(best_x, 1e-18, 5000, 'final')

    # Pass 4: exact Newton polish using analytical Hessian of S(c) = J/N^2
    # S = J/N^2, grad_S = 2(Q4 c)/N^2 - 4J(Q0 c)/N^3
    # Hessian_S = 2Q4/N^2 - 4J(2Q0Q0/N^3 - 2(Q0 c)(Q0 c)^T/N^4)
    #            = 2Q4/N^2 - 8J Q0Q0/N^3 + 8J (Q0 c)(Q0 c)^T/N^4
    if time.time() < TIME_LIMIT - 15:
        try:
            for _ in range(200):
                if time.time() > TIME_LIMIT - 12:
                    break
                c0, c4, c8 = best_x
                c12 = c12_of(c0, c4, c8)
                c = np.array([c0, c4, c8, c12])
                N = float(c @ Q0 @ c)
                J = float(c @ Q4 @ c)
                S = J / (N * N)
                # Gradient in 4-d
                g4 = 2.0 * (Q4 @ c) / (N * N) - 4.0 * J * (Q0 @ c) / (N * N * N)
                # Hessian in 4-d
                H4 = 2.0 * Q4 / (N * N) - 8.0 * J * (Q0 @ Q0) / (N * N * N) + 8.0 * J * np.outer(Q0 @ c, Q0 @ c) / (N * N * N * N)
                # Project to 3-d free variables via constraint c12 = -(c0+3c4+105c8)/945
                # g3[i] = g4[i] + g4[3]*dc12/dc_i
                # H3[i,j] = H4[i,j] + H4[i,3]*dc12/dc_j + H4[3,j]*dc12/dc_i + H4[3,3]*dc12/dc_i*dc12/dc_j
                dc12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
                g3 = g4[:3] + g4[3] * dc12
                H3 = H4[:3,:3] + np.outer(g4[3]*dc12, dc12) + np.outer(dc12, g4[3]*dc12)
                # Wait, let me redo: H3[i,j] = H4[i,j] + H4[i,3]*dc12[j] + H4[3,j]*dc12[i] + H4[3,3]*dc12[i]*dc12[j]
                H3 = H4[:3,:3].copy()
                for i in range(3):
                    for j in range(3):
                        H3[i,j] += H4[i,3]*dc12[j] + H4[3,j]*dc12[i] + H4[3,3]*dc12[i]*dc12[j]
                # Symmetrize
                H3 = 0.5*(H3 + H3.T)
                # Solve H3 * delta = -g3
                try:
                    delta = np.linalg.solve(H3, -g3)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(H3, -g3, rcond=None)[0]
                # Damped step
                step = 1.0
                while step > 1e-12:
                    x_trial = best_x + step * delta
                    x_trial[0] = np.clip(x_trial[0], -50.0, 50.0)
                    x_trial[1] = np.clip(x_trial[1], -5.0, 5.0)
                    x_trial[2] = np.clip(x_trial[2], -0.5, 0.5)
                    v_trial, _ = obj_and_grad(x_trial)
                    if v_trial < S:
                        best_x = x_trial
                        S = v_trial
                        break
                    step *= 0.5
                if step <= 1e-12:
                    break
                if np.linalg.norm(delta) * step < 1e-16:
                    break
        except Exception:
            pass

    # Safety: re-evaluate objective at best_x to ensure consistency
    try:
        v, _ = obj_and_grad(best_x)
        if v < best_val:
            best_val = v
    except Exception:
        pass
    
    return best_x
