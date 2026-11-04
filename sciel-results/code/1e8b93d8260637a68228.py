"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]

# Fixed normal directions for half-plane representation
_HALFPLANE_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)
_NORMALS = np.column_stack([np.cos(_HALFPLANE_ANGLES), np.sin(_HALFPLANE_ANGLES)])

# Basis functions for learnable support function expansion
# h(θ) = a0 + Σ_k [a_k cos(kθ) + b_k sin(kθ)]
_N_BASIS = 4  # number of Fourier modes

def _build_basis_matrix():
    """Build the basis matrix B where h = B @ params."""
    theta = _HALFPLANE_ANGLES
    B = np.ones((N, 1))
    for k in range(1, _N_BASIS + 1):
        B = np.column_stack([B, np.cos(k * theta), np.sin(k * theta)])
    return B

_BASIS_MATRIX = _build_basis_matrix()
_N_PARAMS = _BASIS_MATRIX.shape[1]  # 1 + 2*_N_BASIS = 9

def _support_to_points(h):
    """Convert support function values h[i] at fixed normals to polygon vertices.
    Vertex i is intersection of half-planes i and (i+1)%N."""
    # h[i] = n_i · v_i where n_i is the outward normal
    # v_i satisfies: n_i · v_i = h[i] and n_{i+1} · v_i = h[(i+1)%N]
    pts = np.zeros((N, 2))
    for i in range(N):
        n1 = _NORMALS[i]
        n2 = _NORMALS[(i + 1) % N]
        h1 = h[i]
        h2 = h[(i + 1) % N]
        # Solve: [n1; n2] @ v = [h1; h2]
        A = np.array([n1, n2])
        b = np.array([h1, h2])
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            pts[i] = h1 * n1  # fallback
        else:
            pts[i, 0] = (h1 * A[1, 1] - h2 * A[0, 1]) / det
            pts[i, 1] = (A[0, 0] * h2 - A[1, 0] * h1) / det
    return pts

def _params_to_points(params):
    """Convert parameter vector to points via support function."""
    h = _BASIS_MATRIX @ params
    # Ensure h is positive (valid support function)
    h = np.maximum(h, 0.01)
    return _support_to_points(h)

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    i, j, k = _TRIPLETS.T
    a = pts[i]
    b = pts[j]
    c = pts[k]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _soft_min_objective(x, beta=50.0):
    """Negative soft-min of (min triangle area / hull area) for minimization.
    x: (2N,) flattened point coordinates."""
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _exact_objective(x):
    """Negative exact score for minimization (non-smooth)."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _soft_min_support_objective(params, beta=50.0):
    """Negative soft-min objective in support-function parameter space."""
    pts = _params_to_points(params)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _exact_support_objective(params):
    """Negative exact score in support-function parameter space."""
    pts = _params_to_points(params)
    return -_score(pts)

def _halton_sequence(n, base):
    """Generate first n Halton sequence values with given base."""
    result = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        idx = i
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        result.append(r)
    return np.array(result)

def _halton_points_2d(n, seed=0):
    """Generate n 2D points using Halton sequences (bases 2 and 3)."""
    rng = np.random.RandomState(seed)
    offset = rng.randint(1, 1000)
    h2 = _halton_sequence(n, 2)
    h3 = _halton_sequence(n, 3)
    pts = np.column_stack([h2[offset - 1:offset - 1 + n] if offset - 1 + n <= len(h2) else h2[:n],
                           h3[offset - 1:offset - 1 + n] if offset - 1 + n <= len(h3) else h3[:n]])
    return pts

def _concentric_ring_init(seed=0, n_inner=5, n_outer=8):
    """Generate points on two concentric rings for diverse initialization."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    # Inner ring
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_in = 0.4 + rng.uniform(-0.1, 0.1)
    pts[:n_inner, 0] = r_in * np.cos(angles_in)
    pts[:n_inner, 1] = r_in * np.sin(angles_in)
    # Outer ring
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_out = 1.0 + rng.uniform(-0.1, 0.1)
    pts[n_inner:, 0] = r_out * np.cos(angles_out)
    pts[n_inner:, 1] = r_out * np.sin(angles_out)
    return pts

def _regular_polygon_init(seed=0):
    """Regular polygon with small perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi)
    r = 1.0 + rng.normal(0, 0.05, N)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts

def _regular_support_params(seed=0):
    """Generate support parameters for a near-regular polygon."""
    rng = np.random.RandomState(seed)
    params = np.zeros(_N_PARAMS)
    params[0] = 1.0  # base radius
    for k in range(1, _N_BASIS + 1):
        params[2 * k - 1] = rng.normal(0, 0.05)  # cos coefficient
        params[2 * k] = rng.normal(0, 0.05)      # sin coefficient
    return params

def _nelder_mead_polish(pts, max_iter=500, tol=1e-10):
    """Polish using Nelder-Mead on the exact (non-smooth) objective."""
    x0 = pts.flatten()
    try:
        result = minimize(
            _exact_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts

def _nelder_mead_polish_support(params, max_iter=500, tol=1e-10):
    """Polish using Nelder-Mead in support-function parameter space."""
    try:
        result = minimize(
            _exact_support_objective,
            params,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x
    except Exception:
        return params

def _differential_evolution_step(pts, rng, pop_size=20, n_iter=100):
    """Differential evolution step to explore the landscape."""
    best = pts.copy()
    best_score = _score(pts)
    population = [pts.copy() for _ in range(pop_size)]
    
    for i in range(pop_size - 1):
        # Add perturbation
        noise_scale = 0.1
        population[i+1] = pts + rng.normal(0, noise_scale, (N, 2))
    
    for _ in range(n_iter):
        new_best = None
        new_best_score = best_score
        for i in range(pop_size - 2):
            a = population[rng.randint(0, pop_size-1)]
            b = population[rng.randint(0, pop_size-1)]
            c = population[rng.randint(0, pop_size-1)]
            mutant = a + 0.8 * (b - c)
            mutant = np.clip(mutant, -2, 2)
            score = _score(mutant)
            if score > new_best_score:
                new_best_score = score
                new_best = mutant.copy()
            if score > _score(population[i+1]):
                population[i+1] = mutant.copy()
        if new_best is not None:
            best = new_best
            best_score = new_best_score
            population[0] = best.copy()
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_score = 0.0
    best_pts = _regular_polygon_init(0)
    
    # Phase 1: Multi-start with soft-min L-BFGS-B on full 2D coordinates
    init_strategies = [
        _regular_polygon_init,
        _concentric_ring_init,
        lambda s: _halton_points_2d(N, seed=s) * 0.8,
    ]
    
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 40:
            break
        
        strategy_idx = i % len(init_strategies)
        seed = i
        try:
            pts0 = init_strategies[strategy_idx](seed)
            x0 = pts0.flatten()
            
            result = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 2: Support-function parameter space optimization
    # This enforces convexity structurally and allows asymmetric adjustments
    n_support_starts = 10
    for i in range(n_support_starts):
        if time.time() > deadline - 35:
            break
        
        seed = i + 50
        try:
            params0 = _regular_support_params(seed)
            
            result = minimize(
                _soft_min_support_objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            params = result.x
            pts = _params_to_points(params)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 3: Differential evolution to explore landscape
    if time.time() < deadline - 25:
        rng = np.random.RandomState(42)
        evolved_pts, evolved_score = _differential_evolution_step(best_pts, rng, pop_size=15, n_iter=80)
        if evolved_score > best_score:
            best_score = evolved_score
            best_pts = evolved_pts.copy()
    
    # Phase 4: Nelder-Mead polish on the best solution in Cartesian space
    if time.time() < deadline - 10:
        polished = _nelder_mead_polish(best_pts, max_iter=800)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()
    
    # Phase 5: Second round of differential evolution + polish if time permits
    if time.time() < deadline - 15:
        rng2 = np.random.RandomState(123)
        evolved_pts2, evolved_score2 = _differential_evolution_step(best_pts, rng2, pop_size=12, n_iter=60)
        if evolved_score2 > best_score:
            best_score = evolved_score2
            best_pts = evolved_pts2.copy()
        
        if time.time() < deadline - 5:
            polished2 = _nelder_mead_polish(best_pts, max_iter=500)
            score2 = _score(polished2)
            if score2 > best_score:
                best_score = score2
                best_pts = polished2.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
