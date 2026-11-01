"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
def _energy_fsum(pts):
    """Compute logarithmic energy using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e30
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Compute energy and gradient on the sphere (Euclidean gradient)."""
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            dvec = pts[i] - pts[j]
            d = np.linalg.norm(dvec)
            if d < 1e-15:
                return 1e30, grad
            # d/dx_i log(1/|x_i - x_j|) = -1/|x_i-x_j| * (x_i-x_j)/|x_i-x_j|^2 = -(x_i-x_j)/|x_i-x_j|^3
            # Wait: log(1/d) = -log(d), d = |x_i - x_j|
            # d(-log d)/d x_i = -(1/d) * (x_i - x_j)/d = -(x_i - x_j)/d^2
            term = -1.0 / (d * d)
            grad[i] += term * dvec
            grad[j] -= term * dvec
            energy += math.log(1.0 / d)
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _sphere_grad(pts, euc_grad):
    """Project Euclidean gradient onto the sphere tangent space."""
    # Tangent projection: grad_t = grad - (grad . x) x
    dots = np.sum(pts * euc_grad, axis=1, keepdims=True)
    return euc_grad - dots * pts


def _riemannian_cg(pts, max_iter=500, tol=1e-12, time_limit=100):
    """Riemannian Conjugate Gradient on the sphere."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    f, g = _energy_and_grad(pts)
    g_t = _sphere_grad(pts, g)
    
    # Initial search direction: negative gradient
    d = -g_t
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-15:
        return pts, f
    
    beta = 0.0
    best_pts = pts.copy()
    best_f = f
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Line search on the sphere
        # Use backtracking to find a step size
        step = 0.1
        f_cur = f
        g_cur = g
        g_t_cur = g_t
        
        for ls_iter in range(30):
            # Move along geodesic
            # For small steps, we can use exponential map approximation
            # exp_p(d) = p * cos(||d||) + (d/||d||) * sin(||d||)
            d_norm_ls = np.linalg.norm(d)
            if d_norm_ls < 1e-15:
                break
            
            # Scale direction to step size
            d_scaled = d * (step / d_norm_ls)
            d_scaled_norm = np.linalg.norm(d_scaled)
            
            if d_scaled_norm < 1e-15:
                new_pts = pts.copy()
            else:
                cos_t = np.cos(d_scaled_norm)
                sin_t = np.sin(d_scaled_norm) / d_scaled_norm
                new_pts = pts * cos_t + d_scaled * sin_t
                new_pts = _project_to_sphere(new_pts)
            
            f_new, g_new = _energy_and_grad(new_pts)
            
            if not np.isfinite(f_new) or f_new > f_cur + 1e10:
                step *= 0.5
                continue
            
            # Check Armijo condition
            if f_new <= f_cur + 1e-4 * step * np.sum(g_t_cur * d):
                # Accept step
                pts = new_pts
                f = f_new
                g = g_new
                g_t = _sphere_grad(pts, g)
                break
            else:
                step *= 0.5
        else:
            # Line search failed
            break
        
        if f < best_f:
            best_f = f
            best_pts = pts.copy()
        
        # Check convergence
        g_t_norm = np.linalg.norm(g_t)
        if g_t_norm < tol:
            break
        
        # Update direction (Fletcher-Reeves)
        beta_new = np.sum(g_t * g_t) / (np.sum(g_cur * g_cur) + 1e-30)
        d = -g_t + beta_new * d
        d = _sphere_grad(pts, d)  # Project to tangent space
        d = _project_to_sphere(d * 100)  # This is wrong, let's just keep it
        # Actually, just project the direction
        d = d - np.sum(pts * d, axis=1, keepdims=True) * pts
        
        g_cur = g
        g_t_cur = g_t
    
    return best_pts, best_f


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Offset golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.3
    r = np.sqrt(1.0 - z * z)
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 3. Random configuration (seeded for reproducibility)
    rng = np.random.RandomState(42)
    pts_rand = rng.randn(N, 3)
    configs.append(_project_to_sphere(pts_rand))
    
    # 4. Another random configuration
    rng2 = np.random.RandomState(123)
    pts_rand2 = rng2.randn(N, 3)
    configs.append(_project_to_sphere(pts_rand2))
    
    # 5. Icosahedron-based subdivision
    # Start with icosahedron vertices
    phi_ico = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [0, 1, phi_ico], [0, -1, phi_ico], [0, 1, -phi_ico], [0, -1, -phi_ico],
        [1, phi_ico, 0], [-1, phi_ico, 0], [1, -phi_ico, 0], [-1, -phi_ico, 0],
        [phi_ico, 0, 1], [-phi_ico, 0, 1], [phi_ico, 0, -1], [-phi_ico, 0, -1],
        [1, 0, -phi_ico], [-1, 0, -phi_ico], [1, 0, phi_ico], [-1, 0, phi_ico]
    ])
    ico_verts = _project_to_sphere(ico_verts)
    
    # Create 100 points by placing points on the icosahedron faces
    # We'll use a simple approach: distribute 100 points using barycentric coordinates
    # on the faces of the icosahedron
    ico_faces = [
        [0, 1, 15], [0, 15, 14], [0, 14, 11], [0, 11, 10], [0, 10, 15],
        [1, 5, 15], [1, 15, 14], [1, 14, 2], [1, 2, 11], [1, 11, 5],
        [15, 5, 13], [15, 13, 12], [15, 12, 14], [14, 12, 3], [14, 3, 2],
        [11, 2, 8], [11, 8, 7], [11, 7, 10], [10, 7, 9], [10, 9, 15],
        [5, 1, 11], [5, 11, 7], [5, 7, 13], [13, 7, 9], [13, 9, 15],
        [2, 14, 12], [2, 12, 6], [2, 6, 8], [8, 6, 4], [8, 4, 7],
        [9, 7, 4], [9, 4, 3], [9, 3, 15], [12, 13, 9], [12, 9, 3],
        [6, 12, 9], [6, 9, 4], [4, 6, 8]
    ]
    
    # Distribute 100 points among 20 faces (5 per face)
    pts_ico = []
    for face in ico_faces:
        v0, v1, v2 = ico_verts[face[0]], ico_verts[face[1]], ico_verts[face[2]]
        # Place 5 points per face using a small grid
        for k in range(5):
            # Barycentric coordinates
            u = 0.5 + 0.3 * np.cos(2 * np.pi * k / 5)
            v = 0.5 + 0.3 * np.sin(2 * np.pi * k / 5)
            w = 1.0 - u - v
            if w < 0:
                u, v, w = 0.5, 0.25, 0.25
            pt = u * v0 + v * v1 + w * v2
            pts_ico.append(_project_to_sphere(pt.reshape(1, 3)).flatten())
    configs.append(np.array(pts_ico))
    
    # Optimize each configuration
    best_pts = None
    best_energy = float('inf')
    
    for idx, cfg in enumerate(configs):
        if time.time() > deadline - 20:
            break
        
        # Run Riemannian CG
        time_left = deadline - time.time() - 10
        if time_left < 5:
            break
        
        pts_opt, energy = _riemannian_cg(cfg, max_iter=300, tol=1e-12, time_limit=time_left)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt
    
    # If we still have time, do a final polish with more iterations
    if best_pts is not None and time.time() < deadline - 15:
        time_left = deadline - time.time() - 5
        if time_left > 10:
            pts_polish, energy_polish = _riemannian_cg(best_pts, max_iter=500, tol=1e-14, time_limit=time_left)
            if energy_polish < best_energy:
                best_pts = pts_polish
                best_energy = energy_polish
    
    if best_pts is None:
        best_pts = configs[0]
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-9) or not np.all(np.isfinite(best_pts)):
        # Fallback to golden spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
# EVOLVE-BLOCK-END
