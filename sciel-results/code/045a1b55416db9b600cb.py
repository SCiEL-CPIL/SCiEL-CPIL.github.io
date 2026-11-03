import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def construct():
    t_start = time.time()
    deadline = 205.0  # 5s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint P(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Coefficients for powers x^0, x^2, ..., x^12 (index m -> power 2m)
    H0_arr = np.array([1.0])
    H4_arr = np.array([12.0, -48.0, 16.0])
    H8_arr = np.array([1680.0, -32256.0, 23040.0, -4608.0, 256.0])
    H12_arr = np.array([3991680.0, -12355200.0, 13271040.0, -5806080.0,
                        1032192.0, -98304.0, 4096.0])

    # Precompute ∫ x^{2k} e^{-πx²} dx for k=0..12  = (2k-1)!! / (2π)^k
    E_int = np.zeros(13)
    E_int[0] = 1.0
    for k in range(1, 13):
        df = 1.0
        for j in range(1, 2 * k, 2):
            df *= j
        E_int[k] = df / (2.0 * np.pi) ** k

    # Signs for Q: Q[m] = P[m] * (-1)^m
    signs = np.array([(-1.0) ** m for m in range(7)])

    # ---- Spherical reparameterization over the P(0)=0 nullspace ----
    # Constraint: a0 + 12 a4 + 1680 a8 + 665280 a12 = 0
    # Nullspace basis (a0, a4, a8, a12):
    v1 = np.array([0.0, 1.0, 0.0, -12.0 / 665280.0])
    v2 = np.array([0.0, 0.0, 1.0, -1680.0 / 665280.0])
    # Orthonormalize in R^4
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 - np.dot(v2, u1) * u1
    u2 = u2 / np.linalg.norm(u2)
    U = np.vstack([u1, u2])  # 2x4

    # Seed point for reference (lies in nullspace up to rounding)
    a_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    a12_seed = -(a_seed[0] * H0_0 + a_seed[1] * H4_0 + a_seed[2] * H8_0) / H12_0
    a_seed_full = np.array([a_seed[0], a_seed[1], a_seed[2], a12_seed])
    seed_r = np.dot(U, a_seed_full)  # [r1, r2]
    seed_norm = np.linalg.norm(seed_r)
    seed_theta = np.arctan2(seed_r[1], seed_r[0])

    def build_P(a):
        a0, a4, a8, a12 = a
        P = np.zeros(7)
        P[0] += a0 * H0_arr[0]
        for m in range(3):
            P[m] += a4 * H4_arr[m]
        for m in range(5):
            P[m] += a8 * H8_arr[m]
        for m in range(7):
            P[m] += a12 * H12_arr[m]
        return P

    def objective_from_a(a):
        P = build_P(a)
        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)
        return norm_sq * norm_sq_hat

    def objective_spherical(params):
        s, theta = params
        a = s * (np.cos(theta) * U[0] + np.sin(theta) * U[1])
        return objective_from_a(a)

    def objective_log(params):
        val = objective_from_a(params)
        if val <= 0:
            return 0.0
        return np.log(val)

    best_a = a_seed_full.copy()
    best_val = objective_from_a(a_seed_full)

    def consider(a, val):
        nonlocal best_a, best_val
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_a = a.copy()

    # ---- Algebraic critical point finding via polynomial system ----
    # The objective is a quartic polynomial in (a0, a4, a8) with a12 determined by constraint.
    # We find critical points by solving the gradient equations.
    if time.time() < deadline - 80:
        try:
            # Express a12 in terms of a0, a4, a8: a12 = -(a0 + 12*a4 + 1680*a8) / 665280
            # Substitute into objective to get f(a0, a4, a8)
            # Then find critical points: df/da0 = 0, df/da4 = 0, df/da8 = 0
            # These are cubic polynomials (since f is quartic, derivatives are cubic)
            
            # Build the polynomial coefficients for f(a0, a4, a8)
            # P[m] = a0*H0[m] + a4*H4[m] + a8*H8[m] + a12*H12[m]
            # where a12 = -(a0 + 12*a4 + 1680*a8) / 665280
            
            # Let c = 1/665280
            c = 1.0 / H12_0
            a12_expr = lambda a0, a4, a8: -(a0 + 12.0*a4 + 1680.0*a8) * c
            
            # P[m] as linear function of (a0, a4, a8):
            # P[m] = a0*(H0[m] - H12[m]*c) + a4*(H4[m] - 12*H12[m]*c) + a8*(H8[m] - 1680*H12[m]*c)
            P_coeff = np.zeros((7, 3))  # P[m] = sum_j P_coeff[m,j] * a_j
            for m in range(7):
                h0_m = H0_arr[m] if m < len(H0_arr) else 0.0
                h4_m = H4_arr[m] if m < len(H4_arr) else 0.0
                h8_m = H8_arr[m] if m < len(H8_arr) else 0.0
                h12_m = H12_arr[m] if m < len(H12_arr) else 0.0
                P_coeff[m, 0] = h0_m - h12_m * c
                P_coeff[m, 1] = h4_m - 12.0 * h12_m * c
                P_coeff[m, 2] = h8_m - 1680.0 * h12_m * c
            
            # norm_sq = sum_{i,j} P[i] P[j] E_int[i+j]
            # This is a quadratic form: a^T M a where M is 3x3
            # M[i,j] = sum_k P_coeff[k,i] * P_coeff[k,j] * E_int[k] ... wait, need to be careful
            # Actually norm_sq = sum_{p,q} P[p] P[q] E_int[p+q]
            # P[p] = sum_i P_coeff[p,i] * a_i
            # So norm_sq = sum_{p,q} (sum_i P_coeff[p,i] a_i) (sum_j P_coeff[q,j] a_j) E_int[p+q]
            # = sum_{i,j} a_i a_j (sum_{p,q} P_coeff[p,i] P_coeff[q,j] E_int[p+q])
            # Let M_norm[i,j] = sum_{p,q} P_coeff[p,i] P_coeff[q,j] E_int[p+q]
            
            M_norm = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    s = 0.0
                    for p in range(7):
                        for q in range(7):
                            s += P_coeff[p, i] * P_coeff[q, j] * E_int[p + q]
                    M_norm[i, j] = s
            
            # Similarly for norm_sq_hat with Q = P * signs
            # Q[m] = P[m] * (-1)^m
            # So Q_coeff[m,i] = P_coeff[m,i] * (-1)^m
            Q_coeff = P_coeff * signs[:, np.newaxis]
            
            M_hat = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    s = 0.0
                    for p in range(7):
                        for q in range(7):
                            s += Q_coeff[p, i] * Q_coeff[q, j] * E_int[p + q]
                    M_hat[i, j] = s
            
            # objective = a^T M_norm a * a^T M_hat a
            # This is a quartic. Its gradient:
            # d/da_k [a^T A a * a^T B a] = 2*(a^T A a)*A[:,k] + 2*(a^T B a)*B[:,k]
            # Setting to zero: (a^T A a)*A[:,k] + (a^T B a)*B[:,k] = 0 for k=0,1,2
            
            # Let t = a^T A a, u = a^T B a
            # Then: t*A[:,k] + u*B[:,k] = 0
            # This gives: t*A[:,k] = -u*B[:,k]
            # If A[:,k] and B[:,k] are linearly dependent, we can solve for t/u
            # From k=0: t*A[0,0] + u*B[0,0] = 0 => t/u = -B[0,0]/A[0,0] (if A[0,0] != 0)
            # From k=1: t*A[0,1] + u*B[0,1] = 0 => t/u = -B[0,1]/A[0,1] (if A[0,1] != 0)
            # These must be consistent, giving a condition on a.
            
            # Alternative approach: use the fact that at critical points,
            # the vectors A@a and B@a are linearly dependent (since both are proportional to the gradient direction)
            # Actually, from t*A[:,k] + u*B[:,k] = 0, we get that A@a and B@a are parallel (both equal to -lambda * something)
            # Wait: t*A[:,k] + u*B[:,k] = 0 means (t*A + u*B)[:,k] = 0 for all k, so t*A + u*B = 0 as a vector
            # So A@a and B@a are linearly dependent: there exists lambda such that A@a = -lambda * B@a (or similar)
            
            # Let me just use numerical root finding on the gradient
            def gradient_from_a3(a3):
                a = np.array([a3[0], a3[1], a3[2],
                              -(a3[0] * H0_0 + a3[1] * H4_0 + a3[2] * H8_0) / H12_0])
                P = build_P(a)
                P2 = np.convolve(P, P)
                norm_sq = np.dot(P2, E_int)
                Q = P * signs
                Q2 = np.convolve(Q, Q)
                norm_sq_hat = np.dot(Q2, E_int)
                
                # Numerical gradient
                eps = 1e-8
                grad = np.zeros(3)
                for i in range(3):
                    a_plus = a3.copy()
                    a_plus[i] += eps
                    val_plus = objective_from_a_3(a_plus)
                    a_minus = a3.copy()
                    a_minus[i] -= eps
                    val_minus = objective_from_a_3(a_minus)
                    grad[i] = (val_plus - val_minus) / (2 * eps)
                return grad
            
            # Find critical points by solving gradient = 0 using fsolve-like approach
            # Start from multiple initial points
            initial_points = [
                a_seed.copy(),
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 0.0, -1.0]),
                np.array([0.5, 0.5, 0.0]),
                np.array([0.0, 0.5, 0.5]),
                np.array([0.5, 0.0, 0.5]),
            ]
            
            from scipy.optimize import fsolve
            for init_pt in initial_points:
                if time.time() > deadline - 20:
                    break
                try:
                    sol, info, ier, msg = fsolve(gradient_from_a3, init_pt, full_output=True, maxfev=1000)
                    if ier == 1 and np.all(np.isfinite(sol)):
                        val = objective_from_a_3(sol)
                        if np.isfinite(val) and val < best_val:
                            best_val = val
                            best_a = np.array([sol[0], sol[1], sol[2],
                                               -(sol[0] * H0_0 + sol[1] * H4_0 + sol[2] * H8_0) / H12_0])
                except Exception:
                    pass
        except Exception:
            pass

    # ---- Global search in (s, theta) via Differential Evolution ----
    if time.time() < deadline - 70:
        try:
            s_lo, s_hi = 0.0, 5.0
            bounds = [(s_lo, s_hi), (-np.pi, np.pi)]
            res = differential_evolution(
                objective_spherical,
                bounds=bounds,
                seed=12345,
                maxiter=120,
                tol=1e-10,
                popsize=18,
                polish=True,
                init='sobol',
                updating='deferred',
            )
            if res.fun < best_val:
                best_val = res.fun
                best_a = (res.x[0] * (np.cos(res.x[1]) * U[0]
                                       + np.sin(res.x[1]) * U[1])).copy()
        except Exception:
            pass

    # ---- Local polish in 3D (a0, a4, a8) with exact a12 ----
    a3 = best_a[:3].copy()

    def try_local(a3_start, method, opts, tag):
        nonlocal best_a, best_val
        if time.time() > deadline - 10:
            return
        try:
            r = minimize(objective_from_a_3, a3_start, method=method, options=opts)
            if r.fun < best_val:
                best_val = r.fun
                best_a = np.array([r.x[0], r.x[1], r.x[2],
                                   -(r.x[0] * H0_0 + r.x[1] * H4_0 + r.x[2] * H8_0) / H12_0])
        except Exception:
            pass

    def objective_from_a_3(a3):
        a = np.array([a3[0], a3[1], a3[2],
                      -(a3[0] * H0_0 + a3[1] * H4_0 + a3[2] * H8_0) / H12_0])
        return objective_from_a(a)

    # Nelder-Mead polish
    try_local(a3, 'Nelder-Mead',
              {'maxiter': 40000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}, 'nm1')
    # BFGS polish
    try_local(a3, 'BFGS', {'maxiter': 20000, 'gtol': 1e-18, 'disp': False}, 'bfgs')
    # L-BFGS-B polish
    try_local(a3, 'L-BFGS-B', {'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-18}, 'lbfgs')
    # Powell polish
    try_local(a3, 'Powell', {'maxiter': 20000, 'xtol': 1e-15, 'ftol': 1e-18}, 'powell')

    # ---- Final high-precision Nelder-Mead on best point ----
    if time.time() < deadline - 5:
        try:
            a3_final = best_a[:3].copy()
            r = minimize(objective_from_a_3, a3_final, method='Nelder-Mead',
                         options={'maxiter': 120000, 'xatol': 1e-18, 'fatol': 1e-20,
                                  'adaptive': True})
            if r.fun < best_val:
                best_val = r.fun
                best_a = np.array([r.x[0], r.x[1], r.x[2],
                                   -(r.x[0] * H0_0 + r.x[1] * H4_0 + r.x[2] * H8_0) / H12_0])
        except Exception:
            pass

    # Fallback to seed if anything went wrong
    if not np.all(np.isfinite(best_a)) or best_val > objective_from_a(a_seed_full):
        best_a = a_seed_full

    return np.array([best_a[0], best_a[1], best_a[2]])
