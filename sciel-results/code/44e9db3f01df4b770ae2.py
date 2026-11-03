"""Model-and-optimize approach: analytic gradient + Riemannian projection on the unit sphere."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _hermite_poly(n, x):
    """Physicist's Hermite polynomial via recurrence."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x, dtype=float)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def construct():
    t0 = time.time()
    deadline = 195.0  # safety margin vs 210s kill

    def elapsed():
        return time.time() - t0

    # ---------- Grid & basis (higher resolution than seed) ----------
    N = 8192
    x = np.linspace(-25.0, 25.0, N)
    dx = x[1] - x[0]
    freqs = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-0.5 * x * x)

    # Fold in g(0)=0 constraint:  c12 = -(c0 + 12 c4 + 1680 c8)/40320
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss

    # ---------- Precompute Gram matrices ----------
    def gram(w):
        G = np.empty((3, 3), dtype=float)
        for i in range(3):
            for j in range(i, 3):
                G[i, j] = G[j, i] = float(np.sum(w * v[i] * v[j]) * dx)
        return G

    v = (v0, v4, v8)
    M = gram(np.ones(N))
    Mx = gram(x * x)

    Gf = np.array([np.fft.fftshift(np.fft.fft(np.fft.ifftshift(vi))) * dx for vi in v])
    Mf = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(i, 3):
            Mf[i, j] = Mf[j, i] = float(np.sum(freqs * freqs * np.real(Gf[i] * np.conj(Gf[j]))) * dx)

    C = np.array([
        float(np.sum(M[i, 0] * M[j, 0] + M[i, 1] * M[j, 1] + M[i, 2] * M[j, 2]) * dx)
        for i in range(3)
    ] for j in range(3))
    # Build C correctly (nested)
    C = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            C[i, j] = float(np.sum(M[i, :] * M[j, :]) * dx)

    # Symmetrize for numerical safety
    M = 0.5 * (M + M.T)
    Mx = 0.5 * (Mx + Mx.T)
    Mf = 0.5 * (Mf + Mf.T)
    C = 0.5 * (C + C.T)

    # ---------- Objective & analytic gradient ----------
    def objective(c):
        """Return (value, grad) for normalized c. Minimize value * sqrt(M) * sqrt(F)."""
        nrm = float(np.dot(c, c))
        if nrm <= 1e-30:
            return np.inf, np.zeros(3)
        cn = c / np.sqrt(nrm)
        Mv = float(cn @ M @ cn)
        Fv = float(cn @ Mf @ cn)
        if Mv <= 1e-30 or Fv <= 1e-30:
            return np.inf, np.zeros(3)
        Cn = float(cn @ C @ cn)
        val = Cn * np.sqrt(Mv * Fv)
        gM = 2.0 * (M @ cn)
        gF = 2.0 * (Mf @ cn)
        gC = 2.0 * (C @ cn)
        gcn = gC * np.sqrt(Mv * Fv) + Cn * (gM * np.sqrt(Fv) + gF * np.sqrt(Mv)) / (2.0 * np.sqrt(Mv * Fv))
        grad = (gcn - float(np.dot(gcn, cn)) * cn) / (nrm ** 1.5)
        return val, grad

    # ---------- Generalized-eigenvalue direct solve ----------
    # For a fixed Cn = t, minimize Mv*Fv subject to c^T M c = Mv, c^T Mf c = Fv,
    # c^T C c = t, ||c||=1.  Lagrange multipliers give a generalized eigenproblem
    # (M - lam1*C) v = lam2 * Mf v  (with c^T M c fixed).  Sweep lam1 to cover the
    # Cn range, keeping the candidate with the smallest resulting objective.
    def eig_solve():
        """Return (best_c, best_val) from the eigenvalue sweep."""
        if elapsed() > deadline - 10.0:
            return None, np.inf
        # Determine a sensible lam1 sweep range from the diagonal of M and C
        c_diag = np.diag(C)
        m_diag = np.diag(M)
        # Cn for unit-norm c lies in [min eigen of C, max eigen of C]
        try:
            c_eig = np.linalg.eigvalsh(C)
            c_lo, c_hi = float(c_eig[0]), float(c_eig[-1])
        except Exception:
            c_lo, c_hi = float(np.min(c_diag)), float(np.max(c_diag))
        # Expand range a bit
        span = max(c_hi - c_lo, 1e-6)
        c_lo -= 0.2 * span
        c_hi += 0.2 * span
        n_sweep = 60
        best_c_local = None
        best_val_local = np.inf
        for k in range(n_sweep):
            if elapsed() > deadline - 5.0:
                break
            lam1 = c_lo + (c_hi - c_lo) * k / max(n_sweep - 1, 1)
            A = M - lam1 * C
            try:
                # Solve generalized eigenproblem A v = mu * Mf v
                # i.e. Mf^{-1} A v = mu v  -> use eigh with b=Mf
                mu, vecs = np.linalg.eigh(A, Mf)
            except Exception:
                continue
            # For each eigenvector, compute the objective
            for idx in range(vecs.shape[1]):
                v = vecs[:, idx]
                nrm_v = float(np.dot(v, v))
                if nrm_v <= 1e-30:
                    continue
                vn = v / np.sqrt(nrm_v)
                Mv_v = float(vn @ M @ vn)
                Fv_v = float(vn @ Mf @ vn)
                Cn_v = float(vn @ C @ vn)
                if Mv_v <= 1e-30 or Fv_v <= 1e-30 or Cn_v <= 1e-30:
                    continue
                val_v = Cn_v * np.sqrt(Mv_v * Fv_v)
                if np.isfinite(val_v) and val_v < best_val_local:
                    best_val_local = val_v
                    best_c_local = vn.copy()
        return best_c_local, best_val_local

    def objective(c):
        """Return (value, grad) for normalized c. Minimize value * sqrt(M) * sqrt(F)."""
        nrm = float(np.dot(c, c))
        if nrm <= 1e-30:
            return np.inf, np.zeros(3)
        cn = c / np.sqrt(nrm)
        Mv = float(cn @ M @ cn)
        Fv = float(cn @ Mf @ cn)
        if Mv <= 1e-30 or Fv <= 1e-30:
            return np.inf, np.zeros(3)
        Cn = float(cn @ C @ cn)
        val = Cn * np.sqrt(Mv * Fv)
        gM = 2.0 * (M @ cn)
        gF = 2.0 * (Mf @ cn)
        gC = 2.0 * (C @ cn)
        gcn = gC * np.sqrt(Mv * Fv) + Cn * (gM * np.sqrt(Fv) + gF * np.sqrt(Mv)) / (2.0 * np.sqrt(Mv * Fv))
        grad = (gcn - float(np.dot(gcn, cn)) * cn) / (nrm ** 1.5)
        return val, grad

    def obj_only(c):
        val, _ = objective(c)
        return val

    def eig_solve():
        """Generalized-eigenvalue sweep: for fixed Cn, minimize Mv*Fv via eigh."""
        if elapsed() > deadline - 10.0:
            return None, np.inf
        try:
            c_eig = np.linalg.eigvalsh(C)
            c_lo, c_hi = float(c_eig[0]), float(c_eig[-1])
        except Exception:
            c_lo, c_hi = float(np.min(np.diag(C))), float(np.max(np.diag(C)))
        span = max(c_hi - c_lo, 1e-6)
        c_lo -= 0.3 * span
        c_hi += 0.3 * span
        n_sweep = 80
        best_c_local = None
        best_val_local = np.inf
        for k in range(n_sweep):
            if elapsed() > deadline - 5.0:
                break
            lam1 = c_lo + (c_hi - c_lo) * k / max(n_sweep - 1, 1)
            A = M - lam1 * C
            try:
                mu, vecs = np.linalg.eigh(A, Mf)
            except Exception:
                continue
            for idx in range(vecs.shape[1]):
                v = vecs[:, idx]
                nrm_v = float(np.dot(v, v))
                if nrm_v <= 1e-30:
                    continue
                vn = v / np.sqrt(nrm_v)
                Mv_v = float(vn @ M @ vn)
                Fv_v = float(vn @ Mf @ vn)
                Cn_v = float(vn @ C @ vn)
                if Mv_v <= 1e-30 or Fv_v <= 1e-30 or Cn_v <= 1e-30:
                    continue
                val_v = Cn_v * np.sqrt(Mv_v * Fv_v)
                if np.isfinite(val_v) and val_v < best_val_local:
                    best_val_local = val_v
                    best_c_local = vn.copy()
        return best_c_local, best_val_local

    # ---------- Deterministic multi-start ----------
    rng = np.random.RandomState(12345)

    def project_to_sphere(c):
        n = np.linalg.norm(c)
        if n < 1e-12:
            return c
        return c / n

    # Starting points: seed solution + random directions
    starts = []
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    starts.append(c_seed / np.linalg.norm(c_seed))
    # coordinate axis directions
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        starts.append(e)
    # random directions
    for _ in range(14):
        d = rng.randn(3)
        starts.append(d / np.linalg.norm(d))

    best_c = None
    best_val = np.inf

    # --- Try generalized eigenvalue solve first ---
    eig_c, eig_val = eig_solve()
    if eig_c is not None and np.isfinite(eig_val):
        best_c = eig_c.copy()
        best_val = eig_val
        # Also add as a starting point for gradient refinement
        starts.append(project_to_sphere(eig_c))

    def try_refine(c0):
        nonlocal best_c, best_val
        if elapsed() > deadline:
            return
        c0 = project_to_sphere(c0)
        # Riemannian gradient descent with backtracking
        c = c0.copy()
        val, grad = objective(c)
        if not np.isfinite(val):
            return
        lr = 0.5
        for it in range(400):
            if elapsed() > deadline:
                break
            # project gradient to tangent space
            gproj = grad - float(np.dot(grad, c)) * c
            gnorm = np.linalg.norm(gproj)
            if gnorm < 1e-13:
                break
            dstep = -gproj / gnorm
            # backtracking line search
            alpha = lr
            improved = False
            for _ in range(20):
                cand = project_to_sphere(c + alpha * dstep)
                cand_val, _ = objective(cand)
                if np.isfinite(cand_val) and cand_val < val:
                    c = cand
                    val = cand_val
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                lr *= 0.7
                if lr < 1e-8:
                    break
        # polish with L-BFGS-B in Cartesian (constrained by renormalizing in wrapper)
        if elapsed() < deadline - 3.0:
            try:
                def cart(c):
                    return obj_only(c)
                res = minimize(
                    cart, x0=c,
                    jac=lambda c: objective(c)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 40}
                )
                if np.isfinite(res.fun) and res.fun < val:
                    c = project_to_sphere(res.x)
                    val, _ = objective(c)
            except Exception:
                pass
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_c = c.copy()

    for s in starts:
        try_refine(s)
        if elapsed() > deadline - 5.0:
            break

    if best_c is None:
        # Fallback to seed
        c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        best_c = c_seed / np.linalg.norm(c_seed)
    else:
        best_c = project_to_sphere(best_c)

    # --- Final high-precision polish pass ---
    if elapsed() < deadline - 2.0:
        try:
            def cart(c):
                return obj_only(c)
            res = minimize(
                cart, x0=best_c,
                jac=lambda c: objective(c)[1],
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 60}
            )
            if np.isfinite(res.fun):
                polished = project_to_sphere(res.x)
                pval, _ = objective(polished)
                if np.isfinite(pval) and pval < best_val:
                    best_c = polished
                    best_val = pval
        except Exception:
            pass

    return np.array([best_c[0], best_c[1], best_c[2]])

# EVOLVE-BLOCK-END
