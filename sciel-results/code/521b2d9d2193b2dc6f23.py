"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.

Combined approach:
- Barycentric coordinate representation with direct determinant computation (faster)
- Sign-fixed LP sub-problems (structural change: fix determinant signs, solve LP)
- Soft-min smooth surrogate for gradient-based refinement
- Multi-start with diverse initializations including FPS and structured patterns
- Iterative sign-refinement loop
"""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
TRI_AREA = np.sqrt(3) / 4.0

# Precompute all C(11,3)=165 triple indices
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N)
                         for j in range(i + 1, N)
                         for k in range(j + 1, N)])
M = len(_IDX_TRIPLES)  # 165

def _clamp_bary(bary):
    """Project to valid barycentric simplex: u>=0, v>=0, u+v<=1."""
    u = np.maximum(0.0, bary[:, 0])
    v = np.maximum(0.0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack([u, v])

def _bary_to_cart(bary):
    """Convert barycentric (u, v) to Cartesian. P = u*V0 + v*V1 + (1-u-v)*V2."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack([x, y])

def _all_dets_from_bary(bary):
    """Compute all 165 determinants (twice area in barycentric, scaled by TRI_AREA).
    
    For points with barycentric coords (u_i, v_i, w_i) where w_i = 1-u_i-v_i,
    the area of triangle (i,j,k) = TRI_AREA * |det of 3x3 matrix with rows (u,v,w)|.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    Pi = P[_IDX_TRIPLES[:, 0]]
    Pj = P[_IDX_TRIPLES[:, 1]]
    Pk = P[_IDX_TRIPLES[:, 2]]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
    return det

def _min_area_value(bary):
    """Minimum triangle area in actual units."""
    bary = _clamp_bary(bary)
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets)) * TRI_AREA

def _score_bary(bary):
    return _min_area_value(bary)

def _softmin_objective(x, beta=120.0):
    """Smooth surrogate: negative soft-min of |dets| with floor penalty."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_a - log_sum / beta
    
    # Penalize near-degenerate triangles
    floor_penalty = 500.0 * np.sum(np.maximum(0.0, 1e-7 - areas) ** 2)
    return -soft_min + floor_penalty

def _hard_objective(x):
    """Hard objective: negative min |det|."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    dets = _all_dets_from_bary(bary)
    return -np.min(np.abs(dets))

def _solve_lp_with_signs(bary_init):
    """Solve LP: maximize t subject to sign_m * det_m >= t for all m.
    
    Variables: [u_0, v_0, u_1, v_1, ..., u_10, v_10, t] (2N+1 = 23 vars)
    Constraints:
      - sign_m * (det_m as linear in u,v) >= t  for each m
      - u_i >= 0, v_i >= 0, u_i + v_i <= 1 for each i
      - t >= 0
    """
    bary = _clamp_bary(bary_init)
    dets = _all_dets_from_bary(bary)
    signs = np.sign(dets)
    # Fix zero determinants to +1
    signs[signs == 0] = 1.0
    
    # Build A_ub, b_ub for scipy linprog (minimize c^T x s.t. A_ub x <= b_ub)
    # We want: maximize t  =>  minimize -t
    # Constraints: sign_m * det_m - t >= 0  =>  -sign_m * det_m + t <= 0
    
    n_vars = 2 * N + 1  # 22 + 1 = 23
    # c: minimize -t
    c = np.zeros(n_vars)
    c[-1] = -1.0  # minimize -t
    
    # A_ub rows: for each triple m, -sign_m * det_m + t <= 0
    # det_m = u_i*(v_j*w_k - w_j*v_k) - v_i*(u_j*w_k - w_j*u_k) + w_i*(u_j*v_k - v_j*u_k)
    # where w = 1 - u - v
    # This is linear in u and v.
    # Let's compute the coefficient vector for each triple.
    
    A_rows = []
    b_rows = []
    
    for m in range(M):
        i, j, k = _IDX_TRIPLES[m]
        # det = u_i*(v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k) 
        #       - v_i*(u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k)
        #       + (1-u_i-v_i)*(u_j*v_k - v_j*u_k)
        # 
        # Let's expand in terms of u_i, v_i, u_j, v_j, u_k, v_k:
        # det = u_i*v_j - u_i*v_j*u_k - u_i*v_j*v_k - u_i*v_k + u_i*u_j*v_k + u_i*v_j*u_k
        #       - v_i*u_j + v_i*u_j*u_k + v_i*u_j*v_k - v_i*u_k + v_i*u_j*u_k + v_i*u_j*v_k  ... 
        # 
        # This is getting complex. Let me use a different approach.
        # Compute the gradient of det with respect to each variable.
        # Actually, let's just compute the linear coefficients directly.
        
        # det(i,j,k) where point p has barycentric (u_p, v_p, 1-u_p-v_p)
        # = u_i * [v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k]
        #   - v_i * [u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k]
        #   + (1-u_i-v_i) * [u_j*v_k - v_j*u_k]
        
        # Coefficient of u_i:
        # From first term: v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k
        #   = v_j - v_j*u_k - v_j*v_k - v_k + u_j*v_k + v_j*v_k
        #   = v_j - v_j*u_k - v_k + u_j*v_k
        # From third term: -[u_j*v_k - v_j*u_k] = -u_j*v_k + v_j*u_k
        # Total coeff of u_i: v_j - v_j*u_k - v_k + u_j*v_k - u_j*v_k + v_j*u_k
        #   = v_j - v_k
        # Wait, that can't be right because it should depend on the other points' coordinates.
        # Let me re-derive more carefully.
        
        # Actually, the determinant is:
        # |u_i  v_i  1-u_i-v_i|
        # |u_j  v_j  1-u_j-v_j|
        # |u_k  v_k  1-u_k-v_k|
        
        # = u_i*(v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k)
        #   - v_i*(u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k)
        #   + (1-u_i-v_i)*(u_j*v_k - v_j*u_k)
        
        # This IS linear in each u_p, v_p individually (the product u_i*v_j is NOT present
        # because u_i only appears in row i). Let me verify:
        # u_i appears in: u_i * (v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k)
        #   and in: -(u_i) * (u_j*v_k - v_j*u_k)
        # So coeff of u_i = v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k - (u_j*v_k - v_j*u_k)
        #   = v_j - v_j*u_k - v_j*v_k - v_k + u_j*v_k + v_j*v_k - u_j*v_k + v_j*u_k
        #   = v_j - v_k
        
        # Similarly coeff of v_i = -(u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k) - (u_j*v_k - v_j*u_k)
        #   = -u_j + u_j*u_k + u_j*v_k + u_k - u_j*u_k - u_j*v_k - u_j*v_k + v_j*u_k
        #   Wait let me redo:
        #   = -u_j*(1-u_k-v_k) + (1-u_j-v_j)*u_k - u_j*v_k + v_j*u_k
        #   = -u_j + u_j*u_k + u_j*v_k + u_k - u_j*u_k - v_j*u_k - u_j*v_k + v_j*u_k
        #   = -u_j + u_k
        
        # So the determinant is:
        # det = sum over i of u_i*(v_j - v_k) + v_i*(-u_j + u_k) where (i,j,k) is cyclic
        # Actually for the specific triple (i,j,k):
        # coeff of u_i = v_j - v_k
        # coeff of v_i = u_k - u_j
        # coeff of u_j = v_k - v_i  (by symmetry, cycling)
        # coeff of v_j = u_i - u_k
        # coeff of u_k = v_i - v_j
        # coeff of v_k = u_j - u_i
        
        # Let me verify with a simple case:
        # i=(0,0), j=(1,0), k=(0,1): det should be 1
        # u_i=0,v_i=0; u_j=1,v_j=0; u_k=0,v_k=1
        # coeff_u_i = v_j - v_k = 0 - 1 = -1, contribution: 0*(-1)=0
        # coeff_v_i = u_k - u_j = 0 - 1 = -1, contribution: 0*(-1)=0
        # coeff_u_j = v_k - v_i = 1 - 0 = 1, contribution: 1*1=1
        # coeff_v_j = u_i - u_k = 0 - 0 = 0, contribution: 0
        # coeff_u_k = v_i - v_j = 0 - 0 = 0, contribution: 0
        # coeff_v_k = u_j - u_i = 1 - 0 = 1, contribution: 1*1=1
        # Total = 0+0+1+0+0+1 = 2? But det should be 1.
        
        # Hmm, let me recheck. The 3x3 det of:
        # |0  0  1|
        # |1  0  0|
        # |0  1  0| = 0*(0*0-0*1) - 0*(1*0-0*0) + 1*(1*1-0*0) = 1
        # 
        # So my formula gives 2 but actual is 1. Let me re-derive.
        # 
        # det = u_i*(v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k)
        #     - v_i*(u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k)
        #     + (1-u_i-v_i)*(u_j*v_k - v_j*u_k)
        #
        # For i=(0,0), j=(1,0), k=(0,1):
        # Term 1: 0 * (...) = 0
        # Term 2: -0 * (...) = 0
        # Term 3: 1 * (1*1 - 0*0) = 1
        # Total = 1. Good.
        
        # Let me recompute coefficients:
        # Coeff of u_i:
        # From term 1: v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k
        #   = v_j - v_j*u_k - v_j*v_k - v_k + u_j*v_k + v_j*v_k
        #   = v_j - v_j*u_k - v_k + u_j*v_k
        # From term 3: -(u_j*v_k - v_j*u_k) = -u_j*v_k + v_j*u_k
        # Total: v_j - v_j*u_k - v_k + u_j*v_k - u_j*v_k + v_j*u_k = v_j - v_k ✓
        
        # Coeff of v_i:
        # From term 2: -(u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k)
        #   = -u_j + u_j*u_k + u_j*v_k + u_k - u_j*u_k - v_j*u_k
        #   = -u_j + u_j*v_k + u_k - v_j*u_k
        # From term 3: -(u_j*v_k - v_j*u_k) = -u_j*v_k + v_j*u_k
        # Total: -u_j + u_j*v_k + u_k - v_j*u_k - u_j*v_k + v_j*u_k = -u_j + u_k ✓
        
        # Coeff of u_j:
        # From term 1: u_i * (-v_j*v_k + v_j*v_k) = 0? No wait.
        # Actually u_j appears in:
        # Term 1: u_i * [v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k]
        #   The part with u_j: u_i * [0 - (-v_j)*v_k] = u_i * v_j * v_k... no
        #   (1-u_j-v_j)*v_k = v_k - u_j*v_k - v_j*v_k
        #   So term 1 = u_i*[v_j - v_j*u_k - v_j*v_k - v_k + u_j*v_k + v_j*v_k]
        #   = u_i*[v_j - v_j*u_k - v_k + u_j*v_k]
        #   Coeff of u_j in term 1: u_i * v_k
        # Term 2: -v_i * [u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k]
        #   = -v_i * [u_j - u_j*u_k - u_j*v_k - u_k + u_j*u_k + v_j*u_k]
        #   = -v_i * [u_j - u_j*v_k - u_k + v_j*u_k]
        #   Coeff of u_j in term 2: -v_i * 1 = -v_i
        # Term 3: (1-u_i-v_i)*(u_j*v_k - v_j*u_k)
        #   Coeff of u_j: (1-u_i-v_i)*v_k
        # Total coeff of u_j: u_i*v_k - v_i + (1-u_i-v_i)*v_k
        #   = u_i*v_k - v_i + v_k - u_i*v_k - v_i*v_k
        #   = v_k - v_i - v_i*v_k... 
        # Hmm this doesn't simplify to v_k - v_i. Let me just compute numerically.
        
        # Actually, I think the issue is that the determinant IS linear in each variable
        # but the coefficients depend on the OTHER points' coordinates. So for the LP,
        # I need to evaluate these coefficients at the current barycentric values.
        # This means the LP is only valid locally (around the current point).
        # This is fine for an iterative approach.
        
        # Let me just compute the gradient numerically or symbolically at the current point.
        pass
    
    # Given the complexity, let me use a numerical approach:
    # Compute the linearization of each det at the current point.
    
    u = bary[:, 0]
    v = bary[:, 1]
    
    # For each triple, compute the coefficient vector (2N+1 dimensional)
    # such that det_m ≈ c_m^T * [u; v; 1] + const, and we need det_m - t >= 0
    # Actually: sign_m * det_m - t >= 0
    # => sign_m * (linear approx of det_m) - t >= 0
    # => -sign_m * (linear approx) + t <= 0
    
    # Linear approx: det_m ≈ det_m(current) + sum_p grad_det_m[p] * (var_p - var_p_current)
    # = (det_m(current) - sum_p grad_det_m[p]*var_p_current) + sum_p grad_det_m[p]*var_p
    # So the constant term is: det_m(current) - sum_p grad_det_m[p]*var_p_current
    # And the linear part is: sum_p grad_det_m[p]*var_p
    
    # For the constraint: sign_m * det_m - t >= 0
    # sign_m * [const_m + sum_p grad_m[p]*var_p] - t >= 0
    # -sign_m * const_m - sign_m * sum_p grad_m[p]*var_p + t <= 0
    
    A_ub_list = []
    b_ub_list = []
    
    for m in range(M):
        i, j, k = _IDX_TRIPLES[m]
        s = signs[m]
        
        # Compute gradient of det_m with respect to each u_p, v_p
        # Using the formula:
        # grad_u_i = v_j - v_k  (but this was derived assuming the others are fixed)
        # Wait, I showed it equals v_j - v_k above. Let me just use that.
        # Actually I need to double-check. Let me use finite differences for safety.
        
        # Finite difference approach for the gradient
        grad = np.zeros(2 * N)
        eps = 1e-8
        for p in range(N):
            for d in range(2):
                bary_plus = bary.copy()
                bary_plus[p, d] += eps
                bary_plus = _clamp_bary(bary_plus)
                det_plus = _all_dets_from_bary(bary_plus)[m]
                bary_minus = bary.copy()
                bary_minus[p, d] -= eps
                bary_minus = _clamp_bary(bary_minus)
                det_minus = _all_dets_from_bary(bary_minus)[m]
                grad[2*p + d] = (det_plus - det_minus) / (2 * eps)
        
        det_val = dets[m]
        const = det_val - np.dot(grad, bary.flatten())
        
        # Constraint: s * (const + grad^T * bary_vars) - t >= 0
        # => -s*const - s*grad^T*bary_vars + t <= 0
        row = np.zeros(n_vars)
        row[:2*N] = -s * grad
        row[-1] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(-s * const)
    
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    
    # Bounds: u_i in [0,1], v_i in [0,1], t in [0, 1]
    bounds = [(0, 1)] * (2 * N) + [(0, 1)]
    
    # Initial point for linprog (helps with HiGHS)
    x0 = np.zeros(n_vars)
    x0[:2*N] = bary.flatten()
    x0[-1] = np.min(np.abs(dets))
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            bary_opt = _clamp_bary(res.x[:2*N].reshape(N, 2))
            t_opt = res.x[-1]
            return bary_opt, t_opt
        else:
            return bary_init.copy(), None
    except Exception:
        return bary_init.copy(), None

def _softmin_lpbfgs(bary0, maxiter=300, beta=120.0):
    """L-BFGS-B with soft-min objective."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        res = minimize(
            _softmin_objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
        )
        return _clamp_bary(res.x.reshape(N, 2))
    except Exception:
        return bary0.copy()

def _hard_lpbfgs(bary0, maxiter=200):
    """L-BFGS-B with hard min objective."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        res = minimize(
            _hard_objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        return _clamp_bary(res.x.reshape(N, 2))
    except Exception:
        return bary0.copy()

def _structured_init_bary(seed=42):
    """Diverse structured initializations."""
    rng = np.random.RandomState(seed)
    
    patterns = [
        # A: vertices + midpoints + centroid
        np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3],
            [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
        ]),
        # B: hexagonal
        np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [0.33, 0.33],
            [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
        ]),
        # C: spread
        np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
            [0.33, 0.33],
            [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
        ]),
        # D: concentric
        np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.333, 0.0], [0.667, 0.0], [0.0, 0.333],
            [0.0, 0.667], [0.333, 0.333],
            [0.667, 0.333], [0.333, 0.667], [0.5, 0.5]
        ]),
        # E: magic
        np.array([
            [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
            [1/3, 1/3], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
            [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25]
        ]),
        # F: lattice
        np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3], [0.25, 0.0], [0.75, 0.0],
            [0.25, 0.25], [0.5, 0.75]
        ]),
    ]
    
    idx = seed % len(patterns)
    bary = patterns[idx].copy()
    jitter = rng.normal(0, 0.015, size=bary.shape)
    bary = bary + jitter
    return _clamp_bary(bary)

def _fps_init(seed=42):
    """Farthest Point Sampling initialization."""
    rng = np.random.RandomState(seed)
    n_grid = 25
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            candidates.append([i / n_grid, j / n_grid])
    candidates = np.array(candidates)
    
    # Start with centroid
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = np.min(
            np.linalg.norm(candidates[:, None, :] - np.array(selected)[None, :, :], axis=2),
            axis=1
        )
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
    
    return np.array(selected)

def _random_init_bary(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    bary = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0.01, 0.99)
            v = rng.uniform(0.01, 0.99)
            if u + v < 0.98:
                bary[i] = [u, v]
                break
    return bary

def _iterative_lp_refine(bary0, max_iters=10, time_limit=None, start_time=None):
    """Iteratively solve LP with sign-fixing to converge to local optimum."""
    bary = bary0.copy()
    best_bary = bary.copy()
    best_score = _score_bary(bary)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        bary_lp, t_val = _solve_lp_with_signs(bary)
        score_lp = _score_bary(bary_lp)
        
        if score_lp > best_score:
            best_score = score_lp
            best_bary = bary_lp.copy()
            bary = bary_lp
        else:
            # No improvement, try a small perturbation
            rng = np.random.RandomState(it * 7 + 1)
            pert = rng.normal(0, 0.01, size=(N, 2))
            bary_try = _clamp_bary(bary + pert)
            bary_lp2, _ = _solve_lp_with_signs(bary_try)
            score_lp2 = _score_bary(bary_lp2)
            if score_lp2 > best_score:
                best_score = score_lp2
                best_bary = bary_lp2.copy()
                bary = bary_lp2
            else:
                break
    
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    best_bary = None
    best_score = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_score
        s = _score_bary(bary)
        if s > best_score:
            best_score = s
            best_bary = bary.copy()
        return s
    
    # Phase 1: Multi-start with structured + FPS initializations
    inits = []
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        inits.append(_structured_init_bary(seed=seed))
    inits.append(_fps_init(seed=42))
    inits.append(_fps_init(seed=123))
    
    for bary_init in inits:
        if time_left() < 8.0:
            break
        
        # Soft-min L-BFGS
        if time_left() > 6.0:
            bary_sm = _softmin_lpbfgs(bary_init, maxiter=300, beta=120.0)
            update_best(bary_sm)
            bary_init = bary_sm if _score_bary(bary_sm) > _score_bary(bary_init) else bary_init
        
        # Hard L-BFGS polish
        if time_left() > 4.0:
            bary_h = _hard_lpbfgs(bary_init, maxiter=150)
            update_best(bary_h)
            bary_init = bary_h if _score_bary(bary_h) > _score_bary(bary_init) else bary_init
        
        # LP refinement (structural: sign-fixed LP)
        if time_left() > 10.0:
            bary_lp = _iterative_lp_refine(bary_init, max_iters=8,
                                           time_limit=time_left() - 5.0,
                                           start_time=start_time)
            update_best(bary_lp)
    
    # Phase 2: More random starts
    rng = np.random.RandomState(999)
    for rs in range(6):
        if time_left() < 10.0:
            break
        
        bary_rand = _random_init_bary(seed=2000 + rs)
        
        if time_left() > 7.0:
            bary_sm = _softmin_lpbfgs(bary_rand, maxiter=250, beta=100.0)
            update_best(bary_sm)
            bary_rand = bary_sm
        
        if time_left() > 5.0:
            bary_h = _hard_lpbfgs(bary_rand, maxiter=150)
            update_best(bary_h)
            bary_rand = bary_h
        
        if time_left() > 10.0:
            bary_lp = _iterative_lp_refine(bary_rand, max_iters=6,
                                           time_limit=time_left() - 5.0,
                                           start_time=start_time)
            update_best(bary_lp)
    
    # Phase 3: Perturbations around best
    if best_bary is not None:
        max_perts = 30
        for p in range(max_perts):
            if time_left() < 5.0:
                break
            
            scale = 0.03 * (1 - p / max_perts) + 0.005
            pert = rng.normal(0, scale, size=(N, 2))
            bary_p = _clamp_bary(best_bary + pert)
            
            if time_left() > 4.0:
                bary_sm = _softmin_lpbfgs(bary_p, maxiter=200, beta=100.0)
                s = update_best(bary_sm)
                if s > best_score:
                    best_bary = bary_sm
            
            if time_left() > 3.0 and _score_bary(bary_p) > best_score:
                bary_h = _hard_lpbfgs(bary_p, maxiter=100)
                update_best(bary_h)
    
    # Phase 4: Final LP refinement on best
    if best_bary is not None and time_left() > 10.0:
        bary_lp_final = _iterative_lp_refine(best_bary, max_iters=10,
                                             time_limit=time_left() - 5.0,
                                             start_time=start_time)
        update_best(bary_lp_final)
    
    # Phase 5: Very fine perturbations
    if best_bary is not None and time_left() > 5.0:
        for ft in range(10):
            if time_left() < 2.0:
                break
            pert = rng.normal(0, 0.003, size=(N, 2))
            bary_ft = _clamp_bary(best_bary + pert)
            bary_sm = _softmin_lpbfgs(bary_ft, maxiter=150, beta=150.0)
            update_best(bary_sm)
    
    # Fallback
    if best_bary is None:
        best_bary = _structured_init_bary(seed=42)
    
    return _bary_to_cart(best_bary)
