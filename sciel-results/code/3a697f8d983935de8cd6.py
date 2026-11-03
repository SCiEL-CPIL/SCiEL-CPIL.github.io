"""Model-and-optimize approach: generalized eigenvalue + iterative refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh


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

    # ---------- Grid & basis ----------
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

    C = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            C[i, j] = float(np.sum(M[i, :] * M[j, :]) * dx)

    # Symmetrize for numerical safety
    M = 0.5 * (M + M.T)
    Mx = 0.5 * (Mx + Mx.T)
    Mf = 0.5 * (Mf + Mf.T)
    C = 0.5 * (C + C.T)

    # ---------- Objective ----------
    def compute_bound(c):
        """Compute the upper bound C4 for normalized coefficient vector c."""
        nrm = float(np.dot(c, c))
        if nrm <= 1e-30:
            return np.inf
        cn = c / np.sqrt(nrm)
        Mv = float(cn @ M @ cn)
        Fv = float(cn @ Mf @ cn)
        Cn = float(cn @ C @ cn)
        if Mv <= 1e-30 or Fv <= 1e-30:
            return np.inf
        return Cn * np.sqrt(Mv * Fv)

    # ---------- Generalized eigenvalue approach ----------
    # We want to minimize f(c) = (c^T C c) * sqrt((c^T M c) * (c^T Mf c))
    # subject to c^T c = 1.
    #
    # Strategy: Use an iterative scheme where we alternate between:
    # 1. Fix the "effective" mass matrix and solve for the optimal c
    # 2. Update the effective matrices based on the current c
    #
    # A more direct approach: minimize the log of the objective, which is:
    # log(f) = log(c^T C c) + 0.5*log(c^T M c) + 0.5*log(c^T Mf c)
    #
    # We can use a generalized eigenvalue formulation by considering:
    # Minimize c^T C c subject to c^T (alpha*M + beta*Mf) c = 1
    # where alpha and beta are chosen to balance the M and Mf terms.
    #
    # Actually, let's use a simpler approach: since we're in 3D, we can
    # parameterize the sphere and use a direct optimization with multiple
    # starting points, but use the eigenvalue structure to guide starts.

    # First, find the eigenvectors of the "average" problem
    # Minimize c^T C c subject to c^T M c = 1 and c^T Mf c = 1
    # This is a constrained problem; we can use Lagrange multipliers.

    # Simpler: use the fact that in 3D, we can enumerate critical points.
    # The gradient of f(c) = Cn*sqrt(Mv*Fv) on the sphere is:
    # grad = (gC*sqrt(Mv*Fv) + Cn*(gM*sqrt(Fv) + gF*sqrt(Mv))/(2*sqrt(Mv*Fv)))
    # projected to tangent space.

    def objective_grad(c):
        nrm = float(np.dot(c, c))
        if nrm <= 1e-30:
            return np.inf, np.zeros(3)
        cn = c / np.sqrt(nrm)
        Mv = float(cn @ M @ cn)
        Fv = float(cn @ Mf @ cn)
        Cn = float(cn @ C @ cn)
        if Mv <= 1e-30 or Fv <= 1e-30:
            return np.inf, np.zeros(3)
        val = Cn * np.sqrt(Mv * Fv)
        gM = 2.0 * (M @ cn)
        gF = 2.0 * (Mf @ cn)
        gC = 2.0 * (C @ cn)
        gcn = gC * np.sqrt(Mv * Fv) + Cn * (gM * np.sqrt(Fv) + gF * np.sqrt(Mv)) / (2.0 * np.sqrt(Mv * Fv))
        grad = (gcn - float(np.dot(gcn, cn)) * cn) / (nrm ** 1.5)
        return val, grad

    # ---------- Multi-start with eigenvalue-guided initial points ----------
    rng = np.random.RandomState(12345)

    def project_to_sphere(c):
        n = np.linalg.norm(c)
        if n < 1e-12:
            return c
        return c / n

    # Generate starting points from eigenvectors of C, M, Mf
    starts = []

    # Eigenvectors of C
    try:
        eigvals_C, eigvecs_C = eigh(C)
        for i in range(3):
            starts.append(project_to_sphere(eigvecs_C[:, i]))
            starts.append(project_to_sphere(-eigvecs_C[:, i]))
    except Exception:
        pass

    # Eigenvectors of M
    try:
        eigvals_M, eigvecs_M = eigh(M)
        for i in range(3):
            starts.append(project_to_sphere(eigvecs_M[:, i]))
            starts.append(project_to_sphere(-eigvecs_M[:, i]))
    except Exception:
        pass

    # Eigenvectors of Mf
    try:
        eigvals_Mf, eigvecs_Mf = eigh(Mf)
        for i in range(3):
            starts.append(project_to_sphere(eigvecs_Mf[:, i]))
            starts.append(project_to_sphere(-eigvecs_Mf[:, i]))
    except Exception:
        pass

    # Coordinate axes
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        starts.append(e)
        starts.append(-e)

    # Seed solution
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    starts.append(project_to_sphere(c_seed))
    starts.append(project_to_sphere(-c_seed))

    # Random directions
    for _ in range(20):
        d = rng.randn(3)
        starts.append(project_to_sphere(d))

    # Remove duplicates (approximately)
    unique_starts = []
    for s in starts:
        is_dup = False
        for u in unique_starts:
            if abs(np.dot(s, u)) > 0.999:
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(s)
    starts = unique_starts

    best_c = None
    best_val = np.inf

    def try_refine(c0):
        nonlocal best_c, best_val
        if elapsed() > deadline:
            return
        c = project_to_sphere(c0)
        val, grad = objective_grad(c)
        if not np.isfinite(val):
            return

        # Riemannian gradient descent with backtracking
        lr = 0.5
        for it in range(500):
            if elapsed() > deadline:
                break
            gproj = grad - float(np.dot(grad, c)) * c
            gnorm = np.linalg.norm(gproj)
            if gnorm < 1e-14:
                break
            dstep = -gproj / gnorm
            alpha = lr
            improved = False
            for _ in range(30):
                cand = project_to_sphere(c + alpha * dstep)
                cand_val, _ = objective_grad(cand)
                if np.isfinite(cand_val) and cand_val < val:
                    c = cand
                    val = cand_val
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                lr *= 0.7
                if lr < 1e-10:
                    break

        # Polish with L-BFGS-B
        if elapsed() < deadline - 3.0:
            try:
                def cart(c):
                    return compute_bound(c)
                def cart_grad(c):
                    _, g = objective_grad(c)
                    return g
                res = minimize(
                    cart, x0=c,
                    jac=cart_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50}
                )
                if np.isfinite(res.fun) and res.fun < val:
                    c = project_to_sphere(res.x)
                    val, _ = objective_grad(c)
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
        c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        best_c = project_to_sphere(c_seed)
    else:
        best_c = project_to_sphere(best_c)

    return np.array([best_c[0], best_c[1], best_c[2]])

# EVOLVE-BLOCK-END
