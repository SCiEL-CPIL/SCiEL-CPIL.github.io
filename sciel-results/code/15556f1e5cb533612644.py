import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time


def construct():
    t_start = time.time()
    deadline = 208.0  # 2s safety margin from 210s

    def now():
        return time.time()

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

    # ---- Build quadratic-form matrices for the 3 free coefficients ----
    # a12 = -(a0 + 12 a4 + 1680 a8) / H12_0
    c = 1.0 / H12_0
    # P[m] = sum_j P_coeff[m,j] * a_j  (a = [a0, a4, a8])
    P_coeff = np.zeros((7, 3))
    for m in range(7):
        h0_m = H0_arr[m] if m < len(H0_arr) else 0.0
        h4_m = H4_arr[m] if m < len(H4_arr) else 0.0
        h8_m = H8_arr[m] if m < len(H8_arr) else 0.0
        h12_m = H12_arr[m] if m < len(H12_arr) else 0.0
        P_coeff[m, 0] = h0_m - h12_m * c
        P_coeff[m, 1] = h4_m - 12.0 * h12_m * c
        P_coeff[m, 2] = h8_m - 1680.0 * h12_m * c

    Q_coeff = P_coeff * signs[:, np.newaxis]

    # M_norm[i,j] = sum_{p,q} P_coeff[p,i] P_coeff[q,j] E_int[p+q]
    M_norm = np.zeros((3, 3))
    M_hat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            s = 0.0
            sh = 0.0
            for p in range(7):
                for q in range(7):
                    e = E_int[p + q]
                    s += P_coeff[p, i] * P_coeff[q, j] * e
                    sh += Q_coeff[p, i] * Q_coeff[q, j] * e
            M_norm[i, j] = s
            M_hat[i, j] = sh

    # Symmetrize (numerical safety)
    M_norm = 0.5 * (M_norm + M_norm.T)
    M_hat = 0.5 * (M_hat + M_hat.T)

    def a_full(a3):
        a12 = -(a3[0] * H0_0 + a3[1] * H4_0 + a3[2] * H8_0) * c
        return np.array([a3[0], a3[1], a3[2], a12])

    def objective_from_a3(a3):
        t = float(a3 @ M_norm @ a3)
        u = float(a3 @ M_hat @ a3)
        return t * u

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

    # ---- Seed point ----
    a_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_a3 = a_seed.copy()
    best_val = objective_from_a3(a_seed)

    def consider(a3, val):
        nonlocal best_a3, best_val
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_a3 = np.array(a3, dtype=float)

    # ---- Algebraic global solve: gradient = 0 is a cubic system ----
    # grad_k = 2 t M_norm[:,k] + 2 u M_hat[:,k] = 0, t=a.MN.a, u=a.MH.a
    # => t M_norm[:,k] + u M_hat[:,k] = 0 for k=0,1,2
    # Eliminate t,u: 3x3 matrix G(a) = [t*MN[:,k] + u*MH[:,k]]_{k rows} = 0
    # Each entry is cubic. Solve via resultants.
    try:
        if now() < deadline - 15:
            # Build symbolic-free cubic coefficients for each G entry.
            # G[k, l] = t * MN[l, k] + u * MH[l, k]  (row l, col k) -> but we need
            # the 3 equations for k=0,1,2:  F_k = t*MN[:,k] + u*MH[:,k] = 0 (vector)
            # We have 3 vector equations (9 scalar) but only 3 independent combos.
            # Use the 3 scalar equations: for each k, F_k · (something) ... 
            # Simpler: the 3 equations t*MN[:,k]+u*MH[:,k]=0 each give 3 components.
            # Pick 3 independent scalar equations to form a 3x3 system = 0.
            # Use rows l=0,1,2 with the k-equation structure:
            # Define E_k = t*MN[:,k] + u*MH[:,k]; require E_0=E_1=E_2=0 (vectors).
            # Take one scalar from each: E_k[0] for k=0,1,2? Not guaranteed independent.
            # Robust: take the 3 equations E_k[k] (diagonal-ish) as the system.
            def cubic_coeffs(MN_col, MH_col):
                # returns coeffs of polynomial in (a0,a4,a8): t*MN_col + u*MH_col
                # t = a^T MN a (quadratic), u = a^T MH a (quadratic)
                # product is quartic in a; but we treat it as polynomial.
                # We'll build via expansion using monomials.
                # Represent polynomial as dict monomial->coeff, monomial=(e0,e1,e2)
                def quad(M):
                    d = {}
                    for i in range(3):
                        for j in range(3):
                            m = [0, 0, 0]
                            m[i] += 1
                            m[j] += 1
                            key = tuple(m)
                            d[key] = d.get(key, 0.0) + 0.5 * (M[i, j] + M[j, i])
                    return d
                def add(d, other, scale):
                    for k, v in other.items():
                        d[k] = d.get(k, 0.0) + scale * v
                    return d
                def mul(d1, d2):
                    out = {}
                    for k1, v1 in d1.items():
                        for k2, v2 in d2.items():
                            key = (k1[0] + k2[0], k1[1] + k2[1], k1[2] + k2[2])
                            out[key] = out.get(key, 0.0) + v1 * v2
                    return out
                qt = quad(M_norm)
                qu = quad(M_hat)
                # t*MN_col: for component l, MN[l,k]*t ; but we want t*MN_col[k]?? 
                # E_k[l] = t*MN[l,k] + u*MH[l,k]. For fixed k, vector over l.
                # We'll build the scalar equation E_k[k] = t*MN[k,k] + u*MH[k,k]
                # = MN[k,k]*t + MH[k,k]*u
                res = {}
                add(res, mul(qt, {tuple([0,0,0]): MN_col}), 1.0) if False else None
                # t * MN_col where MN_col is a scalar (the [k,k] entry)
                # E_k[k] = MN[k,k] * t + MH[k,k] * u
                # t is quadratic, u is quadratic -> E_k[k] is quadratic! Not cubic.
                # Wait: t and u are quadratic in a. E_k[k] = MN[k,k]*t + MH[k,k]*u is quadratic.
                # So each scalar equation is quadratic, not cubic. 
                # But we have 9 scalar equations (3k x 3l) each quadratic.
                # Actually the full condition E_k=0 (vector) gives 3 quadratic eqs per k.
                # Total 9 quadratic equations in 3 unknowns -> overdetermined, but
                # the true condition is that t*MN[:,k]+u*MH[:,k]=0 for all k, i.e.
                # the 3x3 matrix [t*MN + u*MH] has zero columns -> rank condition.
                # Equivalent: there exist t,u not both zero s.t. t*MN[:,k]+u*MH[:,k]=0.
                # This means columns of MN and MH are linearly dependent (t/u fixed)
                # AND a lies in the intersection of kernels.
                # Simpler robust route: minimize |t*MN[:,k]+u*MH[:,k]|^2 over a and ratio.
                # Let r = u/t (or handle t=0). Then condition: (MN[:,k] + r MH[:,k])·... 
                # Actually we need t*MN[:,k] + u*MH[:,k]=0 => MN[:,k]+r MH[:,k]=0 (if t!=0)
                # for all k => the 3x3 matrix MN + r*MH = 0 => r must make it zero,
                # which only happens if MN and MH are proportional. Not generally true.
                # So the correct statement: at critical point, for the specific a,
                # the vectors t*MN@a and u*MH@a are both zero? No.
                # Re-derive: grad = 2 t (MN@a) + 2 u (MH@a) = 0  (vector, 3 components)
                # => t*(MN@a) = -u*(MH@a). So MN@a and MH@a are parallel.
                # Let MN@a = w, MH@a = v. Condition: w and v parallel => w x v = 0 (3 cubic eqs)
                # OR equivalently the 2x3 matrix [w; v] has rank 1 => all 2x2 minors = 0.
                # w = MN@a (linear in a), v = MH@a (linear in a).
                # 2x2 minors of [[w0,w1,w2],[v0,v1,v2]] are w_i v_j - w_j v_i (quadratic in a).
                # So 3 quadratic equations! Great.
                # w_i v_j - w_j v_i = 0 for (i,j) in {(0,1),(0,2),(1,2)}
                # Each is quadratic in a. Solve 3 quadratic equations.
                # Use two of them to eliminate, get a resultant.
                # Let's build the 3 quadratics as coefficient arrays and use np.roots via
                # eliminating one variable.
                def lin_coeffs(M):
                    # returns 3 arrays of length 3: M@a components as linear in a
                    return M  # (M@a)[i] = sum_j M[i,j] a_j
                # w_i = sum_j MN[i,j] a_j ; v_i = sum_j MH[i,j] a_j
                # minor(i,j) = w_i v_j - w_j v_i = sum_{p,q} (MN[i,p]MH[j,q] - MN[j,p]MH[i,q]) a_p a_q
                def minor_coeffs(i, j):
                    A = np.zeros((3, 3))
                    for p in range(3):
                        for q in range(3):
                            A[p, q] = MN[i, p] * MH[j, q] - MN[j, p] * MH[i, q]
                    return 0.5 * (A + A.T)
                m01 = minor_coeffs(0, 1)
                m02 = minor_coeffs(0, 2)
                m12 = minor_coeffs(1, 2)
                # Solve the system of 3 quadratics. Use approach: treat as
                # find a where all three vanish. Use grid of a2, then solve 2x2
                # system of quadratics in (a0,a1) for each a2 via resultant.
                # Simpler: use scipy fsolve on the 3 quadratics from many starts,
                # plus a coarse grid, then polish. This is robust enough.
                def quad_vec(a3):
                    w = M_norm @ a3
                    v = M_hat @ a3
                    return np.array([w[0]*v[1]-w[1]*v[0],
                                     w[0]*v[2]-w[2]*v[0],
                                     w[1]*v[2]-w[2]*v[1]])
                from scipy.optimize import fsolve
                starts = [a_seed.copy(),
                          np.zeros(3),
                          np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]),
                          np.array([-1,0,0]), np.array([0,-1,0]), np.array([0,0,-1]),
                          np.array([1,1,0]), np.array([0,1,1]), np.array([1,0,1]),
                          np.array([-1,-1,0]), np.array([0,-1,-1]), np.array([-1,0,-1])]
                for s in starts:
                    if now() > deadline - 12:
                        break
                    try:
                        sol, info, ier, msg = fsolve(quad_vec, s, full_output=True,
                                                     maxfev=2000, xtol=1e-14)
                        if np.all(np.isfinite(sol)):
                            # check residual
                            if np.max(np.abs(quad_vec(sol))) < 1e-8:
                                val = objective_from_a3(sol)
                                consider(sol, val)
                    except Exception:
                        pass
                # Also a coarse random set
                rng = np.random.default_rng(7)
                for _ in range(200):
                    if now() > deadline - 12:
                        break
                    s = rng.normal(0, 1, 3)
                    try:
                        sol, info, ier, msg = fsolve(quad_vec, s, full_output=True,
                                                     maxfev=1000, xtol=1e-14)
                        if np.all(np.isfinite(sol)):
                            if np.max(np.abs(quad_vec(sol))) < 1e-8:
                                val = objective_from_a3(sol)
                                consider(sol, val)
                    except Exception:
                        pass
    except Exception:
        pass

    # ---- Differential Evolution over the 2D nullspace (s, theta) ----
    if now() < deadline - 30:
        try:
            v1 = np.array([0.0, 1.0, 0.0, -12.0 / H12_0])
            v2 = np.array([0.0, 0.0, 1.0, -1680.0 / H12_0])
            u1 = v1 / np.linalg.norm(v1)
            u2 = v2 - np.dot(v2, u1) * u1
            u2 = u2 / np.linalg.norm(u2)
            U = np.vstack([u1, u2])

            def obj_sph(params):
                s, theta = params
                a = s * (np.cos(theta) * U[0] + np.sin(theta) * U[1])
                return objective_from_a(a)

            res = differential_evolution(
                obj_sph,
                bounds=[(0.0, 5.0), (-np.pi, np.pi)],
                seed=12345, maxiter=150, tol=1e-11, popsize=20,
                polish=True, init='sobol', updating='deferred',
            )
            if res.fun < best_val:
                a = res.x[0] * (np.cos(res.x[1]) * U[0] + np.sin(res.x[1]) * U[1])
                consider(a[:3], res.fun)
        except Exception:
            pass

    # ---- Local polish in 3D ----
    def try_local(a3_start, method, opts):
        nonlocal best_a3, best_val
        if now() > deadline - 8:
            return
        try:
            r = minimize(objective_from_a3, a3_start, method=method, options=opts)
            if np.isfinite(r.fun) and r.fun < best_val:
                best_val = r.fun
                best_a3 = np.array(r.x, dtype=float)
        except Exception:
            pass

    a3 = best_a3.copy()
    try_local(a3, 'Nelder-Mead',
              {'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
    try_local(a3, 'Powell', {'maxiter': 30000, 'xtol': 1e-16, 'ftol': 1e-18})
    try_local(a3, 'L-BFGS-B', {'maxiter': 30000, 'ftol': 1e-19, 'gtol': 1e-19})

    # Final high-precision polish
    if now() < deadline - 4:
        try:
            r = minimize(objective_from_a3, best_a3.copy(), method='Nelder-Mead',
                         options={'maxiter': 200000, 'xatol': 1e-19, 'fatol': 1e-21,
                                  'adaptive': True})
            if np.isfinite(r.fun) and r.fun < best_val:
                best_val = r.fun
                best_a3 = np.array(r.x, dtype=float)
        except Exception:
            pass

    # Fallback to seed
    if not np.all(np.isfinite(best_a3)) or best_val > objective_from_a3(a_seed):
        best_a3 = a_seed.copy()

    return np.array([best_a3[0], best_a3[1], best_a3[2]])
