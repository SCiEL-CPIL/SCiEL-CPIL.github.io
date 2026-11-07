import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET = N / 2.0
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0


def _objective(h):
    """Exact objective: max over lags of the full cross-correlation h . (1-h)."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return float(np.max(corr))


def _project(h):
    """Project onto {0<=h<=1, sum(h)=N/2} by clip + iterative redistribution."""
    h = np.asarray(h, dtype=float).copy()
    for _ in range(200):
        h = np.clip(h, 0.0, 1.0)
        diff = TARGET - h.sum()
        if abs(diff) < 1e-13:
            break
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if not np.any(mask):
            break
        h[mask] += diff / mask.sum()
    h = np.clip(h, 0.0, 1.0)
    # final exact fix on any non-saturated entries
    diff = TARGET - h.sum()
    if abs(diff) > 1e-13:
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.any(mask):
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    return h


def _correlate_full(h, g):
    """Full cross-correlation, matching np.correlate(h, g, mode='full')."""
    return np.correlate(h, g, mode='full')


def _analytic_max_grad(h):
    """Exact sub-gradient of max_lag corr(h,1-h) w.r.t. h."""
    one_h = 1.0 - h
    corr = _correlate_full(h, one_h)
    k = int(np.argmax(corr))
    offset = k - (N - 1)
    d = np.zeros(N)
    # term 1: h[i] * (1-h[i-offset])  ->  d/dh[i] = (1-h[i-offset])
    idx1 = np.arange(N) - offset
    m1 = (idx1 >= 0) & (idx1 < N)
    d[m1] += (1.0 - h[idx1[m1]])
    # term 2: h[i+offset] * (1-h[i])  ->  d/dh[i] = -h[i+offset]
    idx2 = np.arange(N) + offset
    m2 = (idx2 >= 0) & (idx2 < N)
    d[m2] += -h[idx2[m2]]
    return d


def _smooth_max(h, beta):
    """Smooth (log-sum-exp) approximation of max(corr)."""
    corr = _correlate_full(h, 1.0 - h)
    m = np.max(corr)
    shifted = beta * (corr - m)
    return m + np.log(np.sum(np.exp(shifted))) / beta


def _smooth_max_grad(h, beta):
    """Analytic gradient of the smooth max."""
    corr = _correlate_full(h, 1.0 - h)
    m = np.max(corr)
    shifted = beta * (corr - m)
    w = np.exp(shifted)
    w /= w.sum()
    d = np.zeros(N)
    for k in range(2 * N - 1):
        if w[k] <= 1e-12:
            continue
        offset = k - (N - 1)
        idx1 = np.arange(N) - offset
        m1 = (idx1 >= 0) & (idx1 < N)
        d[m1] += w[k] * (1.0 - h[idx1[m1]])
        idx2 = np.arange(N) + offset
        m2 = (idx2 >= 0) & (idx2 < N)
        d[m2] += w[k] * (-h[idx2[m2]])
    return d


def _gd_exact(h0, max_iter=4000, lr0=0.05, deadline=None):
    """Projected gradient descent using the exact max sub-gradient."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _objective(h)
    lr = lr0
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        d = _analytic_max_grad(h)
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            break
        h_new = h - lr * d
        h_new = _project(h_new)
        val = _objective(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
            h = h_new
            lr = min(lr * 1.02, 0.2)
        else:
            lr *= 0.9
            if lr < 1e-5:
                break
    return best_h, best_val


def _gd_smooth(h0, max_iter=2000, lr0=0.02, deadline=None):
    """Projected gradient descent using the smooth max (escapes plateaus)."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _objective(h)
    lr = lr0
    beta = 8.0
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        d = _smooth_max_grad(h, beta)
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            break
        h_new = h - lr * d
        h_new = _project(h_new)
        val = _objective(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
            h = h_new
        else:
            lr *= 0.95
            beta *= 1.05
            if lr < 1e-6:
                break
    return best_h, best_val


def _build_poisson_solver(num_sources):
    """Build a discrete Poisson solver: h = base + G @ s.
    
    We solve -laplacian(h) = s (sparse source) on a 1D grid with Neumann BCs.
    The solution is h = G @ s + c, where G is the Green's function matrix.
    We'll compute G numerically by solving the linear system.
    
    Returns: (G_matrix, num_sources) where G is (N, num_sources)
    """
    # Build the 1D discrete Laplacian with Neumann BCs
    # -laplacian: L[i,i] = 2 (interior), L[i,i-1] = -1, L[i,i+1] = -1
    # With Neumann BCs, the matrix is singular (constant null space).
    # We fix the gauge by requiring sum(h) = 0 for the particular solution.
    
    # Actually, let's use a simpler approach: 
    # h[i] = sum_j G[i,j] * s[j]
    # where G is computed from the Green's function of the 1D Poisson eq.
    
    # For 1D with Neumann BCs, the Green's function has a known form.
    # But let's just solve numerically: for each source basis vector e_k,
    # solve L h = e_k with the constraint sum(h)=0.
    
    # Build L (N x N)
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = 2.0
        if i > 0:
            L[i, i-1] = -1.0
        if i < N-1:
            L[i, i+1] = -1.0
    # Neumann BCs: the above is already Neumann-like (no boundary modification needed
    # for the standard 1D Laplacian with this discretization)
    
    # To make L invertible, we add a small regularization or fix the gauge.
    # Let's fix the gauge by setting the last component to 0, or use pseudoinverse.
    # Use SVD-based pseudoinverse for robustness.
    
    # Actually, for efficiency, let's precompute G by solving for each source.
    # But num_sources can be up to ~20, and N=100, so this is fine.
    
    # Use the fact that L has a 1D structure. Let's just use np.linalg.pinv.
    L_pinv = np.linalg.pinv(L)
    
    # G[i, k] = L_pinv[i, k] for k in range(num_sources)
    # But we want sparse sources, so s has non-zero entries at chosen positions.
    # Let's say sources are placed at evenly spaced positions.
    
    source_positions = np.linspace(0, N-1, num_sources).astype(int)
    
    # G[i, k] = L_pinv[i, source_positions[k]]
    G = L_pinv[:, source_positions]
    
    return G, source_positions


def _h_from_sources(s, G, source_positions):
    """Compute h from sparse source vector s using the Poisson Green's function."""
    # s is a vector of length num_sources
    # h_raw = G @ s
    h_raw = G @ s
    # Center: h = h_raw + c, where c is chosen so that sum(h) = TARGET
    c = (TARGET - h_raw.sum()) / N
    h = h_raw + c
    return h


def _slsqp_on_sources(G, source_positions, s0, deadline, max_iter=200):
    """Optimize the sparse source vector s using SLSQP."""
    num_sources = len(s0)
    
    def obj_and_grad(s_vec):
        h = _h_from_sources(s_vec, G, source_positions)
        h = _project(h)
        val = _objective(h)
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros(num_sources)
        for i in range(num_sources):
            sp = s_vec.copy()
            sp[i] += eps
            hp = _project(_h_from_sources(sp, G, source_positions))
            grad[i] = (_objective(hp) - val) / eps
        return val, grad
    
    # Bounds for s: keep them moderate to avoid extreme values
    bounds = [(-1.0, 1.0)] * num_sources
    
    best_s = s0.copy()
    best_val = _objective(_project(_h_from_sources(s0, G, source_positions)))
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        val, grad = obj_and_grad(best_s)
        
        if val < best_val:
            best_val = val
        
        # Use SLSQP to find a step
        try:
            res = minimize(
                lambda sv: _objective(_project(_h_from_sources(sv, G, source_positions))),
                best_s,
                jac=lambda sv: obj_and_grad(sv)[1],
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-8}
            )
            if res.success or res.fun < best_val:
                h_new = _project(_h_from_sources(res.x, G, source_positions))
                v_new = _objective(h_new)
                if v_new < best_val:
                    best_val = v_new
                    best_s = res.x.copy()
        except Exception:
            pass
        
        # Check if converged
        if np.linalg.norm(grad) < 1e-8:
            break
    
    h_best = _project(_h_from_sources(best_s, G, source_positions))
    return h_best, best_val


def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)

    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)  # seed = 0.5

    rng = np.random.RandomState(100)
    starts = [best_h.copy()]

    # random perturbations of varying scale
    for scale in [0.05, 0.1, 0.2, 0.3, 0.4]:
        for _ in range(2):
            hp = 0.5 + rng.randn(N) * scale
            starts.append(_project(hp))

    # sine waves
    x = np.arange(N)
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for amp in [0.2, 0.4, 0.45]:
            hs = 0.5 + amp * np.sin(2 * np.pi * freq * x / N)
            starts.append(_project(hs))

    # alternating two-level
    for a in [0.2, 0.3, 0.4, 0.45]:
        ha = np.where(x % 2 == 0, a, 1.0 - a)
        starts.append(_project(ha))

    # block patterns
    for bs in [10, 20, 25]:
        hb = np.zeros(N)
        hb[::2 * bs][:bs] = 1.0
        for i in range(0, N, 2 * bs):
            hb[i:i + bs] = 1.0
        starts.append(_project(hb))

    # linear ramp
    starts.append(_project(np.linspace(0.0, 1.0, N)))

    # === NEW: Poisson-source-based starts ===
    # Build Poisson solvers with different numbers of sources
    poisson_configs = []
    for num_src in [4, 6, 8, 10]:
        if time.time() > deadline - 10:
            break
        G, src_pos = _build_poisson_solver(num_src)
        poisson_configs.append((G, src_pos, num_src))
    
    # For each Poisson config, create several random source initializations
    for G, src_pos, num_src in poisson_configs:
        if time.time() > deadline - 10:
            break
        for _ in range(3):
            s_init = rng.randn(num_src) * 0.1
            h_poisson = _project(_h_from_sources(s_init, G, src_pos))
            starts.append(h_poisson)

    # refine each start: exact GD then smooth GD polish
    for h0 in starts:
        if time.time() > deadline - 3:
            break
        h1, v1 = _gd_exact(h0, max_iter=4000, lr0=0.03, deadline=deadline)
        if v1 < best_val:
            best_val = v1
            best_h = h1.copy()
        # smooth polish from the best-so-far and from this exact-GD result
        h2, v2 = _gd_smooth(h1, max_iter=1500, lr0=0.01, deadline=deadline)
        if v2 < best_val:
            best_val = v2
            best_h = h2.copy()

    # === NEW: SLSQP optimization on Poisson sources ===
    for G, src_pos, num_src in poisson_configs:
        if time.time() > deadline - 5:
            break
        # Initialize from current best_h: project best_h onto the Poisson subspace
        # We want s such that G @ s + c ≈ best_h, i.e., G @ s ≈ best_h - mean(best_h)
        # Use least squares to find s
        target_h = best_h - best_h.mean()
        try:
            s_init, _, _, _ = np.linalg.lstsq(G, target_h, rcond=None)
        except Exception:
            s_init = rng.randn(num_src) * 0.1
        
        # Clip s to reasonable bounds
        s_init = np.clip(s_init, -1.0, 1.0)
        
        h_pois, v_pois = _slsqp_on_sources(G, src_pos, s_init, deadline, max_iter=50)
        if v_pois < best_val:
            best_val = v_pois
            best_h = h_pois.copy()
        
        # Also try from a random initialization
        if time.time() > deadline - 5:
            break
        s_rand = rng.randn(num_src) * 0.2
        h_pois2, v_pois2 = _slsqp_on_sources(G, src_pos, s_rand, deadline, max_iter=50)
        if v_pois2 < best_val:
            best_val = v_pois2
            best_h = h_pois2.copy()

    # final global polish on the best point
    if time.time() < deadline - 1:
        h3, v3 = _gd_smooth(best_h, max_iter=1500, lr0=0.005, deadline=deadline)
        if v3 < best_val:
            best_val = v3
            best_h = h3.copy()

    best_h = _project(best_h)
    # Final high-precision polish: alternate exact and smooth GD with tiny steps
    if time.time() < deadline - 0.5:
        for _ in range(3):
            if time.time() > deadline - 0.2:
                break
            h4, v4 = _gd_exact(best_h, max_iter=500, lr0=0.01, deadline=deadline)
            if v4 < best_val:
                best_val = v4
                best_h = h4.copy()
            if time.time() > deadline - 0.2:
                break
            h5, v5 = _gd_smooth(best_h, max_iter=500, lr0=0.005, deadline=deadline)
            if v5 < best_val:
                best_val = v5
                best_h = h5.copy()

    best_h = _project(best_h)
    return best_h
