# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses repulsion-based optimization on the sphere (Tammes problem approach).
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 610  # safety margin before 630s kill

    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def project_to_sphere(pts):
        """Project points onto unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return pts / norms

    def compute_repulsion(pts):
        """Compute repulsion forces for points on sphere.
        Uses inverse-square repulsion to maximize minimum distance."""
        dists = pdist(pts, metric='euclidean')
        n_dists = len(dists)
        # Convert pdist to full matrix
        i_idx, j_idx = np.triu_indices(n, k=1)
        dist_matrix = np.zeros((n, n))
        dist_matrix[i_idx, j_idx] = dists
        dist_matrix[j_idx, i_idx] = dists
        
        # Repulsion: force proportional to 1/d^2 for each pair
        # Only consider pairs with small distance to focus on minimum
        repulsion = np.zeros_like(pts)
        for i in range(n):
            for j in range(i+1, n):
                d_ij = dist_matrix[i, j]
                if d_ij < 1e-10:
                    # Random perturbation for coincident points
                    diff = np.random.randn(d)
                    diff /= np.linalg.norm(diff)
                    repulsion[i] += diff
                    repulsion[j] -= diff
                else:
                    # Weight repulsion more heavily for close pairs
                    weight = 1.0 / (d_ij * d_ij + 1e-10)
                    diff = pts[i] - pts[j]
                    diff /= (np.linalg.norm(diff) + 1e-10)
                    repulsion[i] += weight * diff
                    repulsion[j] -= weight * diff
        
        return repulsion

    def repulsion_optimize(pts, max_iters=2000, lr_initial=0.1, lr_decay=0.999):
        """Optimize points on sphere using repulsion forces."""
        pts = project_to_sphere(pts.copy())
        best_pts = pts.copy()
        best_score = get_score(pts)
        lr = lr_initial
        
        for it in range(max_iters):
            elapsed = time.time() - start_time
            if elapsed > time_limit - 10:
                break
            
            # Compute repulsion
            rep = compute_repulsion(pts)
            
            # Project gradient to tangent plane of sphere
            # Tangent: remove radial component
            radial = np.sum(pts * rep, axis=1, keepdims=True)
            tangent_rep = rep - radial * pts
            
            # Update
            pts = pts + lr * tangent_rep
            pts = project_to_sphere(pts)
            
            # Evaluate
            score = get_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
            
            # Decay learning rate
            lr *= lr_decay
            if lr < 1e-8:
                break
        
        return best_pts, best_score

    def simulated_annealing(pts, max_iters=5000, T_initial=0.5, T_final=0.01):
        """Simulated annealing on sphere to escape local minima."""
        pts = project_to_sphere(pts.copy())
        best_pts = pts.copy()
        best_score = get_score(pts)
        
        T = T_initial
        for it in range(max_iters):
            elapsed = time.time() - start_time
            if elapsed > time_limit - 10:
                break
            
            T = T_initial * (T_final / T_initial) ** (it / max_iters)
            
            # Random perturbation
            n_perturb = np.random.randint(1, 4)  # Perturb 1-3 points
            indices = np.random.choice(n, n_perturb, replace=False)
            new_pts = pts.copy()
            for idx in indices:
                # Random direction on sphere
                random_dir = np.random.randn(d)
                random_dir /= np.linalg.norm(random_dir)
                # Small step
                step = T * np.random.randn()
                new_pts[idx] = pts[idx] + step * random_dir
                new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
            
            new_score = get_score(new_pts)
            delta = new_score - best_score
            
            # Accept if better or with probability exp(delta/T)
            if delta > 0 or np.random.random() < np.exp(delta / (T + 1e-10)):
                pts = new_pts
                if new_score > best_score:
                    best_score = new_score
                    best_pts = pts.copy()
        
        return best_pts, best_score

    def polish(pts, max_iters=3000):
        """Final polishing with very small steps."""
        pts = project_to_sphere(pts.copy())
        best_pts = pts.copy()
        best_score = get_score(pts)
        
        for it in range(max_iters):
            elapsed = time.time() - start_time
            if elapsed > time_limit - 5:
                break
            
            # Small random perturbations
            idx = np.random.randint(n)
            random_dir = np.random.randn(d)
            random_dir /= np.linalg.norm(random_dir)
            step = 0.01 * np.random.randn()
            new_pts = pts.copy()
            new_pts[idx] = pts[idx] + step * random_dir
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
            
            new_score = get_score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts
            else:
                # Try opposite direction
                new_pts2 = pts.copy()
                new_pts2[idx] = pts[idx] - step * random_dir
                new_pts2[idx] = new_pts2[idx] / np.linalg.norm(new_pts2[idx])
                new_score2 = get_score(new_pts2)
                if new_score2 > best_score:
                    best_score = new_score2
                    best_pts = new_pts2.copy()
                    pts = new_pts2
        
        return best_pts, best_score

    # Generate multiple initializations
    initializations = []
    
    # 1. Icosahedron + 2 poles
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    base_points = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    initializations.append(project_to_sphere(base_points))
    
    # 2. Fibonacci sphere
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_pts = np.zeros((n, 3))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        radius = np.sqrt(1 - z * z)
        theta = golden_angle * i
        fib_pts[i] = [radius * np.cos(theta), radius * np.sin(theta), z]
    initializations.append(project_to_sphere(fib_pts))
    
    # 3. Random initializations
    for seed in range(30):
        np.random.seed(seed)
        rand_points = np.random.randn(n, d)
        rand_points = rand_points / np.linalg.norm(rand_points, axis=1, keepdims=True)
        initializations.append(rand_points)
    
    # Optimize each initialization
    best_points = initializations[0].copy()
    best_score = get_score(best_points)
    
    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            break
        
        try:
            # Phase 1: Repulsion optimization
            opt_pts, opt_score = repulsion_optimize(init_pts, max_iters=1500)
            
            if opt_score > best_score:
                best_points = opt_pts.copy()
                best_score = opt_score
            
            # Phase 2: Simulated annealing
            elapsed = time.time() - start_time
            if elapsed < time_limit - 20:
                sa_pts, sa_score = simulated_annealing(opt_pts, max_iters=3000)
                if sa_score > best_score:
                    best_points = sa_pts.copy()
                    best_score = sa_score
            
            # Phase 3: Polish
            elapsed = time.time() - start_time
            if elapsed < time_limit - 10:
                polish_pts, polish_score = polish(best_points, max_iters=2000)
                if polish_score > best_score:
                    best_points = polish_pts.copy()
                    best_score = polish_score
                    
        except Exception:
            pass

    # Final validity check
    if not np.all(np.isfinite(best_points)):
        best_points = initializations[0].copy()
    
    return best_points

# EVOLVE-BLOCK-END
