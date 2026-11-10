import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D using Nelder-Mead optimization 
    from multiple symmetric initializations.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 620  # Leave 10s margin
    
    def get_score(pts):
        """
        Calculates the metric (d_min / d_max)^2.
        Returns a value to be maximized.
        """
        # Calculate squared pairwise distances
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        
        if d_max_sq < 1e-12:
            return 0.0
        
        # Metric is (d_min / d_max)^2 = d_min_sq / d_max_sq
        return d_min_sq / d_max_sq

    def get_negative_score(pts):
        """
        Nelder-Mead minimizes. We want to maximize score.
        So we minimize -score.
        """
        return -get_score(pts)

    # Initialize with high-symmetry structures
    phi = (1 + np.sqrt(5)) / 2
    # Standard Icosahedron vertices
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms

    # Generate a set of diverse initial configurations
    # 1. Icosahedron + 2 points at center (will be pushed out by optimizer)
    # 2. Icosahedron + 2 points at opposite poles (distorted)
    # 3. Icosahedron + 2 points at random locations on sphere
    # 4. Regular Tetrahedron-based lattice? (Maybe too sparse)
    # 5. Random Gaussian (as fallback)

    initializations = []
    
    # Init 1: Icosahedron + 2 near center
    p1 = ico_verts.copy()
    p1 = np.vstack([p1, [[0.1, 0, 0]], [[-0.1, 0, 0]]])
    initializations.append(p1)
    
    # Init 2: Icosahedron + 2 at North/South poles (0, 1, 0) and (0, -1, 0)
    # Note: Icosahedron already has points near poles, so this creates clusters.
    p2 = ico_verts.copy()
    p2 = np.vstack([p2, [[0, 1.1, 0]], [[0, -1.1, 0]]])
    initializations.append(p2)

    # Init 3: Icosahedron + 2 random on sphere
    p3 = ico_verts.copy()
    r1 = np.random.randn(3); r1 /= np.linalg.norm(r1)
    r2 = np.random.randn(3); r2 /= np.linalg.norm(r2)
    p3 = np.vstack([p3, [r1], [r2]])
    initializations.append(p3)
    
    # Init 4: Distorted Icosahedron (scale one pair of vertices)
    p4 = ico_verts.copy()
    p4[0] *= 1.2
    p4[1] *= 1.2
    p4 = np.vstack([p4, [[0.2, 0.2, 0.2]], [[-0.2, -0.2, -0.2]]])
    initializations.append(p4)
    
    # Init 5: Random
    p5 = np.random.randn(n, d)
    initializations.append(p5)

    best_points = initializations[0].copy()
    best_score = get_score(best_points)

    # We will run Nelder-Mead (scipy.optimize.minimize) for each initialization
    # within the time budget.
    
    # Parameters for Nelder-Mead
    # maxiter is per run. We need to manage total time.
    # We can run multiple optimizations in parallel? No, standard Python is single-threaded.
    # We will run them sequentially, breaking if time runs out.
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
            
        try:
            # Flatten for optimizer
            x0 = init_pts.flatten()
            
            # Use scipy.optimize.minimize with Nelder-Mead
            # We minimize -score
            from scipy.optimize import minimize
            
            # Set a reasonable maxiter for this specific run.
            # If we have 5 inits, we can spend ~100s each.
            # Nelder-Mead is fast per iteration but might need many.
            # Let's set maxiter=2000 for each, but check time inside?
            # Minimize doesn't have a time limit callback easily.
            # We'll rely on maxiter and the outer loop break.
            
            res = minimize(
                get_negative_score, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.success or res.fun < 0: # res.fun is -score
                current_pts = res.x.reshape(n, d)
                current_score = get_score(current_pts)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts
                    
        except Exception as e:
            # If optimization fails, skip
            pass
            
        # Check time
        if time.time() > deadline:
            break

    return best_points

# Fallback if scipy.optimize is not imported correctly or fails
# The above code imports minimize inside the loop to ensure it's available.
# Let's ensure the import is at the top for efficiency and safety.
import scipy.optimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D using Nelder-Mead optimization 
    from multiple symmetric initializations.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 620  # Leave 10s margin
    
    def get_score(pts):
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        if d_max_sq < 1e-12:
            return 0.0
        return d_min_sq / d_max_sq

    def get_negative_score(pts):
        return -get_score(pts)

    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms

    initializations = []
    
    p1 = ico_verts.copy()
    p1 = np.vstack([p1, [[0.1, 0, 0]], [[-0.1, 0, 0]]])
    initializations.append(p1)
    
    p2 = ico_verts.copy()
    p2 = np.vstack([p2, [[0, 1.1, 0]], [[0, -1.1, 0]]])
    initializations.append(p2)

    p3 = ico_verts.copy()
    r1 = np.random.randn(3); r1 /= np.linalg.norm(r1)
    r2 = np.random.randn(3); r2 /= np.linalg.norm(r2)
    p3 = np.vstack([p3, [r1], [r2]])
    initializations.append(p3)
    
    p4 = ico_verts.copy()
    p4[0] *= 1.2
    p4[1] *= 1.2
    p4 = np.vstack([p4, [[0.2, 0.2, 0.2]], [[-0.2, -0.2, -0.2]]])
    initializations.append(p4)
    
    p5 = np.random.randn(n, d)
    initializations.append(p5)

    best_points = initializations[0].copy()
    best_score = get_score(best_points)

    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
            
        try:
            x0 = init_pts.flatten()
            
            res = scipy.optimize.minimize(
                get_negative_score, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            
            if res.fun < 0: # res.fun is -score, so score > 0
                current_pts = res.x.reshape(n, d)
                current_score = get_score(current_pts)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts
                    
        except Exception as e:
            pass
            
        if time.time() > deadline:
            break

    return best_points
