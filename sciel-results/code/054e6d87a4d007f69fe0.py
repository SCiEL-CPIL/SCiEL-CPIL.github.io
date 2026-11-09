"""Compact dark-energy growth-index fit (CPL cosmologies).
SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    return list(_best_coeffs())

def _best_coeffs():
    t0 = time.time()
    deadline = t0 + 82.0
    c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    try:
        c = _fit(deadline)
    except Exception:
        pass
    return c

def _fit(deadline):
    # Dense reference grid
    om0 = np.array([0.20, 0.24, 0.26, 0.27, 0.28, 0.30, 0.32, 0.35, 0.40])
    w0  = np.array([-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
    wa  = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    a   = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    
    om0 = om0[(om0 > 0.15) & (om0 < 0.45)]
    w0  = w0[(w0 > -1.1) & (w0 < 0.3)]
    wa  = wa[(wa > -0.5) & (wa < 0.5)]
    
    A = a[1:]  # a > 0
    N = len(A)
    M = len(om0) * len(w0) * len(wa)
    
    # Vectorised model grid
    Om0_g, W0_g, Wa_g, A_g = np.meshgrid(om0, w0, wa, A, indexing='ij')
    Om0_g = Om0_g.ravel(); W0_g = W0_g.ravel()
    Wa_g  = Wa_g.ravel();  A_g  = A_g.ravel()
    Npts = len(A_g)
    
    # Precompute per-(a) coefficients for the ODE: y'' = p(a) y' + q(a) y
    P = np.empty(N)
    Qm = np.empty((M, N))
    for i, ai in enumerate(A):
        P[i] = 1.5 / ai
        Qm[:, i] = 0.5 / ai * (3.0 * om0 * ai ** -3.0 + 1.0)
    
    def ref_y_vec():
        """Exact GR linear growth factor on the a-grid via the 2nd-order ODE (vectorised over models)."""
        Y = np.empty((M, N))
        Y[:, 0] = 1.0
        Y[:, 1] = 1.0 + (P[0] + Qm[:, 0] / P[0]) * (A[1] - A[0])
        for i in range(1, N - 1):
            h = A[i + 1] - A[i]
            Y[:, i + 1] = (2.0 * Y[:, i] - Y[:, i - 1]
                           + h * h * (P[i] * (Y[:, i] - Y[:, i - 1]) / h + Qm[:, i] * Y[:, i]))
        return Y
    
    refF = ref_y_vec()
    
    # Compute reference f from ODE: f_ref(a) = a * d ln D / d ln a = a * D'(a)/D(a)
    # We need D'(a). From the ODE: y'' = P y' + Q y.
    # Let's solve for y' as well. Actually, we can estimate D' from the ODE solution.
    # The ODE is: D'' = P D' + Q D
    # We have D at each point. We need D'.
    # From the finite difference scheme:
    # D[i+1] = 2 D[i] - D[i-1] + h^2 (P[i] * (D[i] - D[i-1])/h + Q[i] * D[i])
    # Let's denote v[i] = D'(a_i)
    # Then D[i+1] ≈ D[i] + h v[i] + 0.5 h^2 D''[i]
    # D''[i] = P[i] v[i] + Q[i] D[i]
    # So D[i+1] ≈ D[i] + h v[i] + 0.5 h^2 (P[i] v[i] + Q[i] D[i])
    # = D[i] + v[i] (h + 0.5 h^2 P[i]) + 0.5 h^2 Q[i] D[i]
    # We can solve for v[i]:
    # v[i] = (D[i+1] - D[i] - 0.5 h^2 Q[i] D[i]) / (h + 0.5 h^2 P[i])
    
    f_ref = np.empty((M, N))
    for i in range(N - 1):
        h = A[i + 1] - A[i]
        denom = h + 0.5 * h * h * P[i]
        num = refF[:, i + 1] - refF[:, i] - 0.5 * h * h * Qm[:, i] * refF[:, i]
        v_i = num / denom  # This is D'(a_i)
        f_ref[:, i] = A[i] * v_i / refF[:, i]
    # Last point: use backward difference or copy previous
    h = A[N-1] - A[N-2]
    denom = h + 0.5 * h * h * P[N-2]
    num = refF[:, N-1] - refF[:, N-2] - 0.5 * h * h * Qm[:, N-2] * refF[:, N-2]
    v_last = num / denom
    f_ref[:, N-1] = A[N-1] * v_last / refF[:, N-1]
    
    # Least-squares initial guess
    def _lsq_initial():
        a_sub = np.linspace(0.10, 0.95, 12)
        o_sub = np.linspace(0.20, 0.40, 5)
        w_sub = np.linspace(-0.8, 0.2, 5)
        wa_sub = np.linspace(-0.4, 0.4, 5)
        rows = []
        for o in o_sub:
            for w in w_sub:
                for ww in wa_sub:
                    for ai in a_sub:
                        u = 1.0 + w + ww * (1.0 - ai)
                        if u <= 1e-8:
                            continue
                        om_a = o * ai**3 / (o * ai**3 + (1.0 - o) * ai**(3.0*(1.0+w+ww)))
                        g_ref = (5.0/7.0) * (om_a / u) ** 0.38
                        q = 1.0 - o
                        rows.append([1.0, q, q*q, u, ww*ai, u*q, g_ref])
        X = np.array(rows)
        Xf = X[:, :6]
        y = X[:, 6]
        try:
            sol, *_ = np.linalg.lstsq(Xf, y, rcond=None)
        except Exception:
            sol = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        sol = np.clip(sol, -10.0, 10.0)
        return sol
    
    x0 = _lsq_initial()
    
    # Vectorised gamma / f / D evaluation
    def _gamma(c):
        u = 1.0 + W0_g + Wa_g * (1.0 - A_g)
        q = 1.0 - Om0_g
        g = (c[0] + c[1]*q + c[2]*q*q + c[3]*u
             + c[4]*Wa_g*A_g + c[5]*u*q)
        return np.clip(g, 0.1, 1.2)
    
    def _f(c):
        g = _gamma(c)
        return Om0_g ** g
    
    log_refF = np.log(refF)
    
    def _D_err(c):
        f = _f(c)
        f_mat = f.reshape(M, N)
        h = np.diff(A)
        trap = 0.5 * (f_mat[:, :-1] + f_mat[:, 1:]) * h
        cum = np.cumsum(trap, axis=1)
        
        errs = []
        for j in range(N):
            I_row = np.zeros((M, N))
            if j == 0:
                for i in range(1, N):
                    I_row[:, i] = cum[:, i-1]
            else:
                for i in range(j+1, N):
                    I_row[:, i] = cum[:, i-1] - cum[:, j-1]
            
            log_R = log_refF - log_refF[:, j:j+1]
            mask = np.zeros((M, N), dtype=bool)
            mask[:, j:] = True
            x = -(I_row + log_R)
            e = np.where(np.abs(x) < 0.5, np.expm1(x), np.exp(x) - 1.0)
            err = np.abs(e)
            errs.append(err[mask])
        return np.concatenate(errs)
    
    def _f_err(c):
        f = _f(c).reshape(M, N)
        return np.abs(f - f_ref) / f_ref
    
    def objective(c):
        fe = _f_err(c)
        de = _D_err(c)
        vals = np.concatenate([fe.ravel(), de.ravel()])
        m = vals.max()
        if m < 1e-15:
            return 0.0
        s = np.logaddexp.reduce(np.log(vals + 1e-15))
        return float(s)
    
    # Optimise with multiple restarts
    best = np.clip(x0, -10.0, 10.0)
    best_f = objective(best)
    x0 = best.copy()
    rng = np.random.default_rng(42)
    restarts = 0
    while time.time() < deadline - 10.0:
        res = minimize(objective, x0, method="Nelder-Mead",
                       options=dict(xatol=1e-5, fatol=1e-8, maxiter=400, adaptive=True))
        if res.fun < best_f:
            best_f = res.fun; best = np.clip(res.x, -10.0, 10.0); x0 = best.copy()
        else:
            improved = False
            for _ in range(5):
                cand = best + 0.05 * rng.standard_normal(6)
                cand = np.clip(cand, -10.0, 10.0)
                fc = objective(cand)
                if fc < best_f:
                    best_f = fc; best = cand; x0 = cand.copy(); improved = True
            if not improved:
                restarts += 1
                if restarts >= 3:
                    break
                x0 = best + 0.2 * rng.standard_normal(6)
                x0 = np.clip(x0, -10.0, 10.0)
    
    if time.time() < deadline - 3.0:
        res = minimize(objective, best, method="Nelder-Mead",
                       options=dict(xatol=1e-8, fatol=1e-10, maxiter=600, adaptive=True))
        if res.fun < best_f:
            best = np.clip(res.x, -10.0, 10.0)
    
    best = np.clip(best, -10.0, 10.0)
    return list(best)
# EVOLVE-BLOCK-END
