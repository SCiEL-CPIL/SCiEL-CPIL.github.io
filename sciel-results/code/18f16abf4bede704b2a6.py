"""
Strategy: Informed geometric construction + single L-BFGS relaxation + gradient
polishing (simulated annealing on the analytic gradient), with a small number of
diverse geometric fallbacks if time remains.

Key differences from the current program:
  - No basin-hopping loop with repeated L-BFGS calls (which is the dominant
    cost in the current program).
  - Seeds are *deterministic* geometric constructions (Mackay icosahedral
    shells, FCC corner patches, random compact with min-distance), not random
    molten balls.
  - One L-BFGS relaxation per seed (cheap, ~1-3 s each for n<=80).
  - If time remains, a short gradient-based simulated-annealing polish that
    uses sutton_chen_energy_gradient directly (no L-BFGS overhead) to escape
    shallow local minima.
  - Deadline-guarded at 325 s.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_T0 = time.time()
_DEADLINE = 325.0

def _time_left():
    return _DEADLINE - (time.time() - _T0)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _icosahedron_vertices():
    """Return 12 icosahedron vertices on the unit sphere."""
    phi = (1 + 5 ** 0.5) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _icosahedron_shell_points(shell_radius, n_points, rng):
    """
    Generate n_points on an icosahedral shell of given radius.
    We distribute points on the 20 triangular faces of the icosahedron
    (barycentric coordinates) to get a roughly uniform distribution on the
    surface, then scale to the desired radius.
    """
    phi = (1 + 5 ** 0.5) / 2
    # 12 vertices (same as above)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # 20 faces of the icosahedron (indices into verts)
    faces = [
        (0, 1, 4), (0, 4, 5), (0, 5, 9), (0, 9, 8), (0, 8, 10),
        (1, 3, 4), (3, 2, 4), (2, 6, 4), (6, 7, 4), (7, 5, 4),
        (1, 0, 6), (0, 10, 6), (10, 8, 6), (8, 9, 6), (9, 11, 6),
        (3, 11, 2), (2, 7, 11), (7, 5, 11), (5, 9, 11), (9, 8, 11)
    ]
    # Normalize vertices to unit sphere
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    # Distribute points roughly evenly across faces
    pts_per_face = max(1, n_points // 20)
    extra = n_points - pts_per_face * 20
    
    face_idx = 0
    for face in faces:
        count = pts_per_face + (1 if face_idx < extra else 0)
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        for _ in range(count):
            # Random barycentric coordinates
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            p = (1 - u - v) * v0 + u * v1 + v * v2
            p = p / np.linalg.norm(p)
            points.append(p * shell_radius)
        face_idx += 1
    
    return np.array(points[:n_points])


def _generate_mackay_icosahedral(n, rng):
    """
    Mackay-style icosahedral cluster:
      - 1 central atom
      - 12 atoms on first shell (icosahedron vertices), radius r1
      - 30 atoms on second shell (face centers), radius r2
      - remaining atoms on subsequent shells
    
    For Sutton-Chen, the equilibrium nearest-neighbor distance is ~1.0.
    Icosahedron edge length = 2*r1*sin(pi/5) ≈ 1.176*r1, so r1 ≈ 0.85 for
    edge ≈ 1.0. We use r1 = 0.85, r2 ≈ 1.55 (second shell in Mackay 55-atom).
    """
    pts = [np.array([0.0, 0.0, 0.0])]
    
    # First shell: 12 icosahedron vertices
    r1 = 0.85
    if n >= 13:
        v = _icosahedron_vertices()
        pts.extend([v[i] * r1 for i in range(12)])
    
    # Second shell: 30 points (face centers of icosahedron), radius ~1.55
    r2 = 1.55
    remaining = n - len(pts)
    if remaining > 0:
        shell2 = _icosahedron_shell_points(r2, min(30, remaining), rng)
        pts.extend(shell2)
        remaining = n - len(pts)
    
    # Third shell and beyond: random points on sphere of increasing radius
    shell_idx = 3
    while remaining > 0:
        r_shell = 0.85 * shell_idx ** 0.8  # rough scaling
        count = min(remaining, 60)
        shell_pts = _icosahedron_shell_points(r_shell, count, rng)
        pts.extend(shell_pts)
        remaining = n - len(pts)
        shell_idx += 1
    
    return np.array(pts[:n])


def _generate_fcc_patch(n, rng):
    """
    Generate an FCC-like patch: place atoms on FCC lattice points within a
    sphere, centered at origin. Small random deformation to break symmetry.
    """
    if n <= 1:
        return np.zeros((n, 3))
    
    a = 1.0  # lattice constant
    fcc_pts = []
    # Generate FCC points in a box large enough
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                fcc_pts.append(np.array([i * a, j * a, k * a]))
                fcc_pts.append(np.array([(i + 0.5) * a, (j + 0.5) * a, k * a]))
                fcc_pts.append(np.array([(i + 0.5) * a, j * a, (k + 0.5) * a]))
                fcc_pts.append(np.array([i * a, (j + 0.5) * a, (k + 0.5) * a]))
    
    fcc_pts = np.array(fcc_pts)
    fcc_pts -= fcc_pts.mean(axis=0)
    
    # Sort by distance from origin, take closest n
    dists = np.linalg.norm(fcc_pts, axis=1)
    idx = np.argsort(dists)
    fcc_pts = fcc_pts[idx[:n]]
    
    # Small random deformation
    deformation = rng.normal(0, 0.1, fcc_pts.shape)
    fcc_pts += deformation
    
    return fcc_pts


def _generate_random_compact(n, rng):
    """
    Random compact seed with minimum-distance enforcement.
    Place atoms one by one in a sphere, ensuring min distance >= 0.6.
    """
    if n <= 1:
        return np.zeros((n, 3))
    
    r = 1.2 * n ** (1.0 / 3.0)
    pts = []
    for _ in range(n):
        placed = False
        for _attempt in range(300):
            p = rng.uniform(-r, r, 3)
            if np.linalg.norm(p) > r:
                continue
            if not pts:
                placed = True
                break
            d = np.linalg.norm(np.array(pts) - p, axis=1)
            if np.min(d) >= 0.6:
                placed = True
                break
        pts.append(p)
    
    return np.array(pts)


def _generate_decahedral(n, rng):
    """
    Decahedral (twinned FCC) seed: 5 FCC twins meeting at a center.
    Approximate by placing atoms on a deformed icosahedron with FCC-like
    internal structure. For simplicity, use a distorted icosahedral shell
    with slightly compressed axes.
    """
    base = _generate_mackay_icosahedral(n, rng)
    # Apply a slight anisotropic compression to mimic decahedral twinning
    compression = np.diag([1.0, 0.95, 0.95])
    base = base @ compression.T
    # Small random perturbation
    base += rng.normal(0, 0.05, base.shape)
    return base


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(pos, n):
    """Check validity: finite, shape, min distance >= 0.35."""
    if pos is None or len(pos) != n:
        return False
    if not np.all(np.isfinite(pos)):
        return False
    if n < 2:
        return True
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.min(d) >= 0.35


# ---------------------------------------------------------------------------
# Gradient-based simulated annealing polish
# ---------------------------------------------------------------------------

def _gradient_anneal(pos, potential, kind, rng, max_iters=50, t_start=0.3, t_end=0.05):
    """
    Simple gradient-based simulated annealing:
      - At each step, compute energy and gradient via sutton_chen_energy_gradient.
      - Propose a move: p_new = p - lr * grad + noise
      - Accept if energy decreases, or with probability exp(-dE/T).
      - No L-BFGS call; uses analytic gradient directly.
    """
    n = len(pos)
    try:
        cur_e, cur_grad = sutton_chen_energy_gradient(pos, kind)
    except:
        return float('inf'), pos
    
    if not np.isfinite(cur_e):
        return float('inf'), pos
    
    best_e = cur_e
    best_p = pos.copy()
    cur_p = pos.copy()
    
    t = t_start
    lr = 0.01  # learning rate for gradient step
    noise_scale = 0.15
    
    for i in range(max_iters):
        if _time_left() < 5.0:
            break
        
        # Adaptive learning rate: decrease over time
        lr_i = lr * (1.0 + 0.5 * np.cos(np.pi * i / max_iters))
        noise_i = noise_scale * (t_start / t) ** 0.5
        
        # Propose move
        grad_norm = np.linalg.norm(cur_grad)
        if grad_norm > 1e-10:
            grad_step = cur_grad / grad_norm * lr_i
        else:
            grad_step = np.zeros_like(cur_grad)
        
        noise = rng.normal(0, noise_i, cur_p.shape)
        proposed = cur_p - grad_step + noise
        
        # Evaluate proposed
        try:
            new_e, new_grad = sutton_chen_energy_gradient(proposed, kind)
        except:
            continue
        
        if not np.isfinite(new_e):
            continue
        
        de = new_e - cur_e
        if de < 0 or rng.random() < np.exp(-de / t):
            cur_e = new_e
            cur_grad = new_grad
            cur_p = proposed
            if new_e < best_e:
                best_e = new_e
                best_p = proposed.copy()
        
        # Cool
        t = max(t_end, t * 0.92)
    
    return best_e, best_p


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    best_energy = float('inf')
    best_pos = None
    
    # ------------------------------------------------------------------
    # Phase 1: Deterministic geometric seeds, one L-BFGS relaxation each
    # ------------------------------------------------------------------
    # Allocate ~60% of time to Phase 1 (seeds + relaxations)
    phase1_budget = 0.6 * _DEADLINE
    phase1_deadline = _T0 + phase1_budget
    
    seed_generators = [
        ("mackay", _generate_mackay_icosahedral),
        ("fcc", _generate_fcc_patch),
        ("decahedral", _generate_decahedral),
        ("random_compact", _generate_random_compact),
    ]
    
    rng_base = np.random.default_rng(42)
    
    for name, gen_func in seed_generators:
        if time.time() > phase1_deadline:
            break
        if _time_left() < 10.0:
            break
        
        # Generate up to 3 variants of each seed type
        for variant in range(3):
            if time.time() > phase1_deadline:
                break
            if _time_left() < 8.0:
                break
            
            rng = np.random.default_rng(1000 * (seed_generators.index((name, gen_func)) + 1) + variant)
            try:
                seed = gen_func(n, rng)
            except:
                continue
            
            if seed is None or len(seed) != n or not np.all(np.isfinite(seed)):
                continue
            
            # One L-BFGS relaxation
            try:
                e, pos = relax(seed, potential, kind)
            except:
                continue
            
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_pos = pos
    
    # ------------------------------------------------------------------
    # Phase 2: Gradient-based simulated annealing polish from best
    # ------------------------------------------------------------------
    if best_pos is not None and _time_left() > 15.0:
        remaining_iters = max(20, int((_time_left() - 10.0) / 0.3))
        ga_rng = np.random.default_rng(9999)
        try:
            e, pos = _gradient_anneal(best_pos, potential, kind, ga_rng,
                                       max_iters=remaining_iters,
                                       t_start=0.3, t_end=0.05)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_pos = pos
        except:
            pass
    
    # ------------------------------------------------------------------
    # Phase 3: If time remains, try a few more diverse starts
    # ------------------------------------------------------------------
    if _time_left() > 20.0:
        extra_rng = np.random.default_rng(7777)
        for i in range(3):
            if _time_left() < 8.0:
                break
            try:
                seed = _generate_random_compact(n, extra_rng)
                e, pos = relax(seed, potential, kind)
                if np.isfinite(e) and e < best_energy:
                    best_energy = e
                    best_pos = pos
            except:
                pass
    
    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    if best_pos is None or not _validate(best_pos, n):
        # Last resort: linear chain (will be high energy but valid)
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos
    
    return best_pos
# EVOLVE-BLOCK-END
