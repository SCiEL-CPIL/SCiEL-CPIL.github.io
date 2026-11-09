import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 13

def construct():
    # Helper to compute score
    def compute_score(pts):
        try:
            hull = Delaunay(pts).convex_hull
            hull_pts = pts[hull]
            x = hull_pts[:, 0]
            y = hull_pts[:, 1]
            hull_area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            
            # Compute all triangle areas
            # For efficiency, we only need the minimum. 
            # But for optimization, we need the soft-min.
            # Here we just need the raw score for selection.
            # We can use Delaunay triangles as a proxy for the minimum area 
            # if the points are in general position and well-distributed.
            # Actually, the minimum area over ALL triples is <= min area over Delaunay triangles.
            # The task says "maximize the minimum area over all triples".
            # The Delaunay triangulation minimizes the maximum edge length, but does it guarantee the minimum triangle area is maximized?
            # Not necessarily. However, for well-distributed points, the smallest triangles are usually part of the Delaunay triangulation.
            # To be safe, we should compute the minimum area over all C(13,3) = 286 triples.
            
            # 286 triples is small enough to compute explicitly.
            min_area = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    for k in range(j+1, N):
                        p1 = pts[i]
                        p2 = pts[j]
                        p3 = pts[k]
                        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                        area = 0.5 * abs(cross)
                        if area < min_area:
                            min_area = area
                            
            if hull_area == 0:
                return 0.0
            return min_area / hull_area
        except Exception:
            return 0.0

    # Helper to run optimization with annealing
    def optimize_with_annealing(x0, k_start=5.0, k_end=50.0, n_steps=10, max_iter_per_step=50):
        current_x = x0.copy()
        
        for step in range(n_steps):
            # Linearly interpolate k
            t = step / max(1, n_steps - 1)
            k_val = k_start + (k_end - k_start) * t
            
            def obj(x):
                pts = x.reshape(N, 2)
                try:
                    tri = Delaunay(pts)
                    simplices = tri.simplices
                except Exception:
                    return 1e10, np.zeros_like(x)
                
                areas = np.zeros(len(simplices))
                grads = np.zeros_like(pts)
                
                for i, tri_idx in enumerate(simplices):
                    p1 = pts[tri_idx[0]]
                    p2 = pts[tri_idx[1]]
                    p3 = pts[tri_idx[2]]
                    
                    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                    area = 0.5 * abs(cross)
                    areas[i] = area
                    
                    # Gradient of signed area
                    # g1 = 0.5 * perp(p3 - p2)
                    # g2 = 0.5 * perp(p1 - p3)
                    # g3 = 0.5 * perp(p2 - p1)
                    
                    v32 = p3 - p2
                    v13 = p1 - p3
                    v21 = p2 - p1
                    
                    g1 = 0.5 * np.array([-v32[1], v32[0]])
                    g2 = 0.5 * np.array([-v13[1], v13[0]])
                    g3 = 0.5 * np.array([-v21[1], v21[0]])
                    
                    sign = 1.0 if cross >= 0 else -1.0
                    
                    grads[tri_idx[0]] += sign * g1
                    grads[tri_idx[1]] += sign * g2
                    grads[tri_idx[2]] += sign * g3
                    
                # Soft min
                # f = -1/k * log(sum(exp(-k * A)))
                # We minimize -soft_min = 1/k * log(sum(exp(-k * A)))
                # Gradient: -1/S * sum(exp(-k*A_i) * grad_A_i)
                
                # To avoid overflow, subtract max area? No, exp(-k*A) is small for large A.
                # If A is large, exp(-k*A) is ~0.
                # If A is small, exp(-k*A) is ~1.
                # So we don't need to shift by max. We might need to shift if k*A is very large negative?
                # No, A >= 0. So -k*A <= 0. exp(-k*A) <= 1. No overflow.
                
                exp_vals = np.exp(-k_val * areas)
                S = np.sum(exp_vals)
                
                if S == 0:
                    return 1e10, np.zeros_like(x)
                    
                weights = exp_vals / S
                
                grad_obj = -np.sum(weights[:, np.newaxis] * grads, axis=0)
                
                val = (1.0 / k_val) * np.log(S)
                
                return val, grad_obj.flatten()

            res = minimize(
                lambda x: obj(x)[0],
                current_x,
                jac=lambda x: obj(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_step, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            current_x = res.x
            
        return current_x

    # Initialization strategies
    best_score = -np.inf
    best_points = None
    
    rng = np.random.RandomState(123)
    
    # Strategy 1: Triangular Lattice Subset
    # Generate a grid of triangular lattice points
    # Side length s = 1
    # Points: (i + 0.5*j, j*sqrt(3)/2)
    # We want 13 points. 
    # A 4x4 grid has 16 points. We can take the first 13 or a central cluster.
    # Let's take a cluster around the origin.
    lattice_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    
    # Select 13 points closest to the centroid? Or just the first 13?
    # Let's compute centroid and pick 13 closest.
    centroid = np.mean(lattice_points, axis=0)
    dists = np.linalg.norm(lattice_points - centroid, axis=1)
    idx = np.argsort(dists)[:13]
    init1 = lattice_points[idx]
    
    # Strategy 2: Random points in a disk, then repelled
    # Start with random points in unit disk
    init2 = np.zeros((N, 2))
    for _ in range(50): # Repulsion iterations
        # Compute pairwise distances
        diff = init2[:, np.newaxis, :] - init2[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-10)
        forces = np.zeros_like(init2)
        for i in range(N):
            for j in range(N):
                if i != j:
                    f = (init2[i] - init2[j]) / (dists[i, j]**2)
                    forces[i] += f
        init2 += 0.01 * forces
        # Project back to unit disk? Or just let them spread.
        # We want them to form a mesh, so spreading is good.
        
    # Normalize init2 to roughly unit area
    try:
        hull = Delaunay(init2).convex_hull
        hull_pts = init2[hull]
        x = hull_pts[:, 0]
        y = hull_pts[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area > 0:
            init2 = init2 / np.sqrt(area)
    except:
        pass

    # Strategy 3: Perturbed Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = 1.0 + rng.uniform(-0.2, 0.2, N)
    init3 = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    candidates = [init1, init2, init3]
    
    for init_pts in candidates:
        # Try a few random perturbations for each candidate
        for _ in range(3):
            perturbed = init_pts + rng.uniform(-0.05, 0.05, (N, 2))
            result = optimize_with_annealing(perturbed.flatten())
            result_pts = result.reshape(N, 2)
            
            # Normalize
            try:
                hull = Delaunay(result_pts).convex_hull
                hull_pts = result_pts[hull]
                x = hull_pts[:, 0]
                y = hull_pts[:, 1]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
                if area > 0:
                    result_pts = result_pts / np.sqrt(area)
            except:
                continue
                
            score = compute_score(result_pts)
            if score > best_score:
                best_score = score
                best_points = result_pts
                
    return best_points
