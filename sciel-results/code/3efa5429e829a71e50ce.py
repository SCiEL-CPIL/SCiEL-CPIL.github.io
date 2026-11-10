# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize as scipy_minimize
import time

def _precompute_basis():
    k = 10
    n_theta = 70
    n_phi = 130
    dim = 21

    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    n_pts = n_theta * n_phi

    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)

    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) *
                       float(factorial(k - m) / factorial(k + m)))

    B = np.zeros((n_pts, dim))
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    sqrt2 = np.sqrt(2.0)

    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)
    for m in range(1, 11):
        val = sqrt2 * N[m] * P[m, :]
        cos_t = np.cos(m * phi)
        sin_t = np.sin(m * phi)
        B[:, 10 + m] = (val[:, None] * cos_t[None, :]).ravel()
        B[:, 10 - m] = (val[:, None] * sin_t[None, :]).ravel()

    return B, weights

B_G, W_G = _precompute_basis()

# Precompute the 4-index tensor S for L4 norm
# S is 441x441, S[21*i+j, 21*k+l] = sum_p W_p B_p[i] B_p[j] B_p[k] B_p[l]
def _precompute_S():
    N_pts, dim = B_G.shape
    B_reshaped = B_G[:, :, None] * B_G[:, None, :]
    Z = B_reshaped.reshape(N_pts, dim * dim)
    S = Z.T @ (W_G[:, None] * Z)
    return S

S_G = _precompute_S()

def _compute_ratio(c):
    # Use the tensor-based computation for speed
    norm2_sq = np.dot(c, c)
    if norm2_sq < 1e-30:
        return np.inf
    u = (c[:, None] * c[None, :]).ravel()
    norm4_4 = u @ S_G @ u
    if norm4_4 < 1e-30:
        return np.inf
    norm4 = norm4_4 ** 0.25
    norm2 = np.sqrt(norm2_sq)
    return norm4 / norm2

def _objective_and_grad(c):
    norm2_sq = np.dot(c, c)
    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c)
    
    u = (c[:, None] * c[None, :]).ravel()
    norm4_4 = u @ S_G @ u
    if norm4_4 < 1e-30:
        return np.inf, np.zeros_like(c)
    
    norm4 = norm4_4 ** 0.25
    norm2 = np.sqrt(norm2_sq)
    R = norm4 / norm2
    
    V = S_G @ u
    V_reshaped = V.reshape(21, 21)
    grad_Q = 4.0 * (V_reshaped @ c)
    
    grad_R = grad_Q / (4.0 * norm4**3 * norm2**2) - (norm4 * c) / (norm2**3)
    
    return R, grad_R

def _normalize_unit(c):
    n2 = np.dot(c, c)
    if n2 < 1e-30:
        return c
    return c / np.sqrt(n2)

def _project_onto_sphere(c, r=1.0):
    n = np.linalg.norm(c)
    if n < 1e-15:
        c = np.random.randn(len(c))
        return c / np.linalg.norm(c) * r
    return c / n * r

def _init_pool(dim):
    inits = []
    # zonal
    c = np.zeros(dim); c[10] = 1.0; inits.append(c)
    # all ones
    inits.append(np.ones(dim))
    # single m
    for i in range(dim):
        c = np.zeros(dim); c[i] = 1.0; inits.append(c)
    # low-m bias
    for b in [0.3, 0.5, 0.7]:
        c = np.zeros(dim)
        for i in range(dim):
            c[i] = b ** abs(i - 10)
        inits.append(c)
    # cosine/sine patterns in m
    for fr in np.arange(0.2, 3.0, 0.2):
        c = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c[i] = np.cos(fr * m)
        inits.append(c)
        c2 = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c2[i] = np.sin(fr * m)
        inits.append(c2)
    # random normals
    for s in range(40):
        np.random.seed(s * 12345 + 67890)
        inits.append(np.random.randn(dim))
    # low-m biased random
    for s in range(30):
        np.random.seed(s * 98765 + 43210)
        c = np.random.randn(dim)
        for i in range(dim):
            c[i] *= 0.5 ** abs(i - 10)
        inits.append(c)
    # sparse random
    for s in range(30):
        np.random.seed(s * 24680 + 13579)
        c = np.zeros(dim)
        nnz = np.random.randint(1, 8)
        idx = np.random.choice(dim, nnz, replace=False)
        c[idx] = np.random.randn(nnz)
        inits.append(c)
    # two-peak
    for p1 in range(-10, 11, 3):
        for p2 in range(p1 + 2, 11, 3):
            c = np.zeros(dim)
            for i in range(dim):
                m = i - 10
                c[i] = np.exp(-(m - p1)**2 / 3.0) + np.exp(-(m - p2)**2 / 3.0)
            inits.append(c)
    # anti-symmetric patterns
    for s in range(15):
        np.random.seed(s * 55555 + 11111)
        c = np.random.randn(dim)
        for i in range(dim):
            m = i - 10
            if m < 0:
                c[i] *= -1
        inits.append(c)
    # paired m patterns (m and -m)
    for s in range(15):
        np.random.seed(s * 77777 + 33333)
        c = np.zeros(dim)
        for m in range(1, 11):
            val = np.random.randn()
            c[10 + m] = val
            c[10 - m] = val
        inits.append(c)
    # comb patterns
    for freq in [2, 3, 4, 5]:
        c = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c[i] = 1.0 if (m % freq == 0) else 0.0
        inits.append(c)
    return inits

def _lgbfsb(c0, deadline, maxfun=3000, ftol=1e-18, gtol=1e-16):
    def obj_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        cn = _normalize_unit(c)
        val, grad = _objective_and_grad(cn)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad
    try:
        res = scipy_minimize(obj_grad, c0, jac=True, method='L-BFGS-B',
                             options={'maxiter': 2000, 'maxfun': maxfun,
                                      'ftol': ftol, 'gtol': gtol})
        if res.fun < 1e9:
            return res.fun, _normalize_unit(res.x)
    except Exception:
        pass
    return np.inf, c0

def _nelder_mead(func, x0, deadline, maxiter=2000, tol=1e-15):
    n = len(x0)
    x0 = _project_onto_sphere(x0)
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy(); xi[i] += 0.05
        simplex.append(_project_onto_sphere(xi))
    fvals = [func(v) for v in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    best_val = min(fvals)
    best_pt = simplex[np.argmin(fvals)].copy()
    for it in range(maxiter):
        if time.time() > deadline - 3:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < tol * (1 + abs(fvals[0])):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = _project_onto_sphere(centroid + alpha * (centroid - simplex[-1]))
        fr = func(xr)
        if fr < fvals[0]:
            xe = _project_onto_sphere(centroid + gamma * (centroid - simplex[-1]))
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-1]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            if fr < fvals[-1]:
                xc = _project_onto_sphere(centroid + rho * (simplex[-1] - centroid))
            else:
                xc = _project_onto_sphere(centroid - rho * (simplex[-1] - centroid))
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = _project_onto_sphere(simplex[0] + sigma * (simplex[i] - simplex[0]))
                    fvals[i] = func(simplex[i])
        cur = min(fvals)
        if cur < best_val:
            best_val = cur
            best_pt = simplex[np.argmin(fvals)].copy()
    return best_val, best_pt

def _grad_descent_polish(c, deadline, max_iter=2000):
    c = _project_onto_sphere(c.copy())
    cur = _compute_ratio(c)
    step = 0.1
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
        val, grad = _objective_and_grad(c)
        if not np.isfinite(val):
            break
        grad_t = grad - (grad @ c) * c
        gnorm = np.linalg.norm(grad_t)
        if gnorm < 1e-15:
            break
        grad_t = grad_t / gnorm
        improved = False
        for _bt in range(50):
            c_try = _project_onto_sphere(c - step * grad_t)
            vt = _compute_ratio(c_try)
            if vt < cur - 1e-18:
                c = c_try; cur = vt; improved = True
                step = min(step * 1.5, 0.3)
                break
            step *= 0.5
        if not improved:
            step *= 0.5
            if step < 1e-12:
                break
    return cur, c

def _local_restarts(c_best, n_restarts, deadline, maxfun=500):
    """Generate perturbations around the best solution and optimize each."""
    best_val = _compute_ratio(c_best)
    best_c = c_best.copy()
    dim = len(c_best)
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        # Create a perturbation
        np.random.seed(42 + i * 1000)
        perturbation = np.random.randn(dim) * 0.1
        c_pert = _project_onto_sphere(c_best + perturbation)
        # Optimize from this perturbation
        vo, co = _lgbfsb(c_pert, deadline, maxfun=maxfun)
        if vo < best_val:
            best_val = vo
            best_c = co.copy()
    return best_val, best_c

def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 600.0
    dim = 21

    def objective(c):
        if not np.all(np.isfinite(c)):
            return 1e10
        return _compute_ratio(_normalize_unit(c))

    best_c = None
    best_val = np.inf

    inits = _init_pool(dim)
    init_sorted = []
    for c_init in inits:
        cn = _project_onto_sphere(c_init)
        v = _compute_ratio(cn)
        init_sorted.append((v, cn))
        if v < best_val:
            best_val = v
            best_c = cn.copy()
    init_sorted.sort(key=lambda x: x[0])

    # Phase 1: L-BFGS-B from best inits
    for v, cs in init_sorted[:40]:
        if time.time() > deadline - 150:
            break
        try:
            vo, co = _lgbfsb(cs, deadline, maxfun=2500)
            if vo < best_val:
                best_val = vo
                best_c = co.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead from a few starts
    for v, cs in init_sorted[:12]:
        if time.time() > deadline - 100:
            break
        try:
            vo, co = _nelder_mead(objective, cs, deadline, maxiter=1500)
            if vo < best_val:
                best_val = vo
                best_c = co.copy()
        except Exception:
            pass

    # Phase 3: gradient polish
    if time.time() < deadline - 40:
        vp, cp = _grad_descent_polish(best_c, deadline, max_iter=3000)
        if vp < best_val:
            best_val = vp
            best_c = cp.copy()
        # L-BFGS-B re-polish
        if time.time() < deadline - 20:
            try:
                vo, co = _lgbfsb(best_c, deadline, maxfun=3000, ftol=1e-20, gtol=1e-18)
                if vo < best_val:
                    best_val = vo
                    best_c = co.copy()
            except Exception:
                pass
        # final grad polish
        if time.time() < deadline - 8:
            vp, cp = _grad_descent_polish(best_c, deadline, max_iter=2000)
            if vp < best_val:
                best_val = vp
                best_c = cp.copy()
    
    # Phase 4: Local restarts around the best solution
    if time.time() < deadline - 15:
        vr, cr = _local_restarts(best_c, 8, deadline, maxfun=400)
        if vr < best_val:
            best_val = vr
            best_c = cr.copy()
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            vo, co = _lgbfsb(best_c, deadline, maxfun=4000, ftol=1e-22, gtol=1e-20)
            if vo < best_val:
                best_val = vo
                best_c = co.copy()
        except Exception:
            pass
        vp, cp = _grad_descent_polish(best_c, deadline, max_iter=3000)
        if vp < best_val:
            best_val = vp
            best_c = cp.copy()

    coeffs = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.ones(dim)
    mc = np.max(np.abs(coeffs))
    if mc < 1e-3:
        coeffs = coeffs / mc * 1.0 if mc > 0 else np.ones(dim)
    elif mc > 1e5:
        coeffs = coeffs / mc * 1.0
    return coeffs

# EVOLVE-BLOCK-END
