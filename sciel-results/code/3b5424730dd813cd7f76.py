# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _compute_covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere using hull facets."""
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        
        for simplex in hull.simplices:
            # Get the facet vertices
            facet_pts = pts[simplex]
            
            # Compute the circumcenter of the facet (in 3D)
            # For a triangle, the circumcenter is equidistant from all three vertices
            a, b, c = facet_pts
            
            # Solve for circumcenter using linear algebra
            # |x - a|^2 = |x - b|^2 = |x - c|^2
            # 2x·(b-a) = |b|^2 - |a|^2
            # 2x·(c-a) = |c|^2 - |a|^2
            # Also x is in the plane of the triangle
            
            ab = b - a
            ac = c - a
            aa = np.dot(a, a)
            ba = np.dot(b, a)
            ca = np.dot(c, a)
            
            # The circumcenter lies in the plane: x = a + s*ab + t*ac
            # |x-a|^2 = |x-b|^2 => 2*s*(ab·ab) + 2*t*(ab·ac) = 2*(ba - aa)
            # |x-a|^2 = |x-c|^2 => 2*s*(ac·ab) + 2*t*(ac·ac) = 2*(ca - aa)
            
            ab_ab = np.dot(ab, ab)
            ab_ac = np.dot(ab, ac)
            ac_ac = np.dot(ac, ac)
            
            rhs1 = ba - aa
            rhs2 = ca - aa
            
            # Solve 2x2 system
            det = ab_ab * ac_ac - ab_ac * ab_ac
            if abs(det) < 1e-12:
                continue
            
            s = (rhs1 * ac_ac - ab_ac * rhs2) / det
            t = (ab_ab * rhs2 - ab_ac * rhs1) / det
            
            circumcenter = a + s * ab + t * ac
            
            # The angular radius is the angle between the circumcenter and any vertex
            # Normalize circumcenter to get the direction
            cc_norm = np.linalg.norm(circumcenter)
            if cc_norm < 1e-10:
                continue
            cc_dir = circumcenter / cc_norm
            
            # Angular distance from cc_dir to a vertex
            dot_val = np.clip(np.dot(cc_dir, a), -1, 1)
            angular_radius = np.arccos(dot_val)
            
            # Only count if the cap is empty (circumcenter direction is outside the hull)
            # Check if cc_dir is on the outward side of the facet
            # The outward normal of the facet
            normal = np.cross(ab, ac)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-10:
                continue
            normal_dir = normal / normal_norm
            
            # Check if cc_dir has positive dot with outward normal (outward side)
            if np.dot(cc_dir, normal_dir) > 0:
                max_radius = max(max_radius, angular_radius)
        
        return np.degrees(max_radius)
    except Exception:
        # Fallback to grid method
        n_theta = 180
        n_phi = 360
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        TH, PH = np.meshgrid(thetas, phis, indexing='ij')
        Y = np.zeros((n_theta, n_phi, 3))
        Y[:, :, 0] = np.sin(TH) * np.cos(PH)
        Y[:, :, 1] = np.sin(TH) * np.sin(PH)
        Y[:, :, 2] = np.cos(TH)
        Y = Y.reshape(-1, 3)
        
        dots = Y @ pts.T
        min_dots = np.max(dots, axis=1)
        max_angular = np.arccos(np.clip(min_dots, -1, 1))
        return np.degrees(np.max(max_angular))


def _fibonacci_sphere(n, rng):
    """Generate a Fibonacci sphere initialization with slight random perturbation."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    golden = np.pi * (1.0 + np.sqrt(5.0))
    phi = golden * i + rng.random(n) * 0.1
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _simulate(pts, max_steps=30000, power=6, seed_dt=0.02):
    """Run symplectic integration with adaptive cooling."""
    N = len(pts)
    pts = _normalize(pts.copy())
    vel = np.zeros((N, 3))
    
    for step in range(max_steps):
        t = step / max_steps
        
        # Adaptive step size with exponential decay
        dt = seed_dt * np.exp(-3.0 * t) + 0.00005
        
        # Damping schedule: start with high damping, decrease
        damping = 0.95 - 0.15 * t
        
        # --- Vectorized Force Calculation ---
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist2, 1.0)
        
        # Force magnitude for 1/r^power potential
        inv_dist = 1.0 / np.sqrt(dist2)
        f_mag = power * inv_dist ** (power + 1)
        
        # Zero out self-forces
        np.fill_diagonal(f_mag, 0.0)
        
        # Total force
        forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Project to tangent plane
        dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
        forces -= dot_fp * pts
        
        # Velocity Verlet
        vel_half = vel + 0.5 * dt * forces
        dot_vp = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half -= dot_vp * pts
        
        pts_new = pts + dt * vel_half
        pts_new = _normalize(pts_new)
        
        # Recalculate forces
        diff_new = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dist2_new = np.sum(diff_new ** 2, axis=2)
        np.fill_diagonal(dist2_new, 1.0)
        inv_dist_new = 1.0 / np.sqrt(dist2_new)
        f_mag_new = power * inv_dist_new ** (power + 1)
        np.fill_diagonal(f_mag_new, 0.0)
        forces_new = np.sum(f_mag_new[:, :, np.newaxis] * diff_new, axis=1)
        
        dot_fp_new = np.sum(forces_new * pts_new, axis=1, keepdims=True)
        forces_new -= dot_fp_new * pts_new
        
        vel_new = vel_half + 0.5 * dt * forces_new
        vel_new *= damping
        
        dot_vp_new = np.sum(vel_new * pts_new, axis=1, keepdims=True)
        vel_new -= dot_vp_new * pts_new
        
        pts = pts_new
        vel = vel_new
    
    return _normalize(pts)


def _local_polish(pts, max_iter=2000):
    """Local optimization to polish the configuration."""
    N = len(pts)
    pts = _normalize(pts.copy())
    
    for _ in range(max_iter):
        # Compute forces with softer potential
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist2, 1.0)
        
        inv_dist = 1.0 / np.sqrt(dist2)
        f_mag = 4.0 * inv_dist ** 5
        np.fill_diagonal(f_mag, 0.0)
        
        forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Project to tangent
        dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
        forces -= dot_fp * pts
        
        # Small step update
        step_size = 0.001
        pts_new = pts + step_size * forces
        pts_new = _normalize(pts_new)
        
        # Check if improvement
        if np.allclose(pts, pts_new, atol=1e-10):
            break
        
        pts = pts_new
    
    return _normalize(pts)


def construct():
    """
    Minimize a repulsive potential on the unit sphere using symplectic integration
    with multiple restarts and local polishing.
    """
    N = 50
    best_pts = None
    best_radius = np.inf
    
    # Try multiple initializations
    for trial in range(3):
        rng = np.random.RandomState(42 + trial)
        
        if trial == 0:
            # Fibonacci sphere
            pts_init = _fibonacci_sphere(N, rng)
        elif trial == 1:
            # Random initialization
            pts_init = rng.randn(N, 3)
            pts_init = _normalize(pts_init)
        else:
            # Perturbed Fibonacci
            pts_init = _fibonacci_sphere(N, rng)
            pts_init += rng.randn(N, 3) * 0.1
            pts_init = _normalize(pts_init)
        
        # Run simulation
        pts = _simulate(pts_init, max_steps=25000, power=6, seed_dt=0.02)
        
        # Local polish
        pts = _local_polish(pts, max_iter=1000)
        
        # Evaluate
        radius = _compute_covering_radius(pts)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
