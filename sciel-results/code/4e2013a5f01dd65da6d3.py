"""
Strategy: Rigid Cluster Assembly + Penalty Gradient Descent + Local Unlock
1. Pre-define 6 rigid clusters of circles with optimal internal packing geometry.
2. Treat each cluster as a rigid body (translation + rotation).
3. Optimize cluster poses using penalty-based gradient descent to minimize overlaps.
4. "Unlock" clusters and perform local radius maximization to squeeze out more volume.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(centers, radii, eps=1e-9):
    """
    Check if the packing is valid (no overlaps, inside square).
    Returns (sum_radii, is_valid).
    """
    n = len(radii)
    total_r = np.sum(radii)
    
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r - eps or x > 1.0 - r + eps or y < r - eps or y > 1.0 - r + eps:
            return -np.inf, False
            
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist - eps:
                return -np.inf, False
                
    return total_r, True

def _get_cluster_geometry():
    """
    Define 6 rigid clusters.
    Returns a list of clusters, each with:
    - centers: (k, 2) array of local centers
    - radii: (k,) array of local radii
    """
    clusters = []
    
    # Cluster 1: Top-Left Corner (3 circles)
    # Arranged in a triangle hugging the corner
    r1 = 0.12
    c1 = np.array([
        [r1, r1],
        [r1 + 2*r1*np.cos(np.pi/6), r1 + 2*r1*np.sin(np.pi/6)],
        [r1 + 2*r1*np.cos(5*np.pi/6), r1 + 2*r1*np.sin(5*np.pi/6)]
    ])
    # Actually, for a corner, a hexagonal packing is better.
    # Let's use a simple dense packing for 3 circles in a corner.
    # Circle 1 at (r, r). Circle 2 at (r, r+2r). Circle 3 at (r+2r, r).
    # This leaves gaps. Let's try to make them tangent.
    # Better: 1 large, 2 smaller? No, equal radii is easier to start.
    # Let's use a "triad" that fits in a corner.
    # r1=0.15. c1=(0.15, 0.15), c2=(0.15, 0.45), c3=(0.45, 0.15).
    # Distance c1-c2 = 0.3 = 2*0.15. Tangent.
    # Distance c2-c3 = sqrt(0.3^2 + 0.3^2) = 0.424 > 0.3. OK.
    # Let's make it tighter.
    # Use 3 circles of r=0.14.
    # c1 = (0.14, 0.14)
    # c2 = (0.14, 0.42)
    # c3 = (0.42, 0.14)
    # This is a valid rigid cluster.
    r_val = 0.14
    centers_c1 = np.array([
        [r_val, r_val],
        [r_val, r_val + 2*r_val],
        [r_val + 2*r_val, r_val]
    ])
    radii_c1 = np.array([r_val, r_val, r_val])
    clusters.append((centers_c1, radii_c1))
    
    # Cluster 2: Top-Right Corner (3 circles)
    # Mirror of Cluster 1
    centers_c2 = np.array([
        [1-r_val, r_val],
        [1-r_val, r_val + 2*r_val],
        [1 - (r_val + 2*r_val), r_val]
    ])
    # Wait, local coords should be relative to a local origin for rotation/translation.
    # Let's define local origins at the "center of mass" or a corner.
    # For simplicity, define local coords centered at 0,0.
    
    # Re-define Cluster 1 with local origin at (0,0) corresponding to the corner.
    # But rotation around corner is tricky if it moves away.
    # Let's use local origin at the centroid of the cluster.
    
    def make_cluster(centers_local, radii_local):
        # Centers are relative to a local origin (0,0)
        # We will optimize translation (tx, ty) and rotation (theta)
        # Global pos = R(theta) @ (c_local + t_local) + t_global? 
        # No, standard rigid body: p_global = R @ p_local + t_global
        # So p_local should be centered at 0,0 to avoid large translation offsets?
        # Or just keep p_local as is, and t_global handles position.
        # Let's center local coords at 0,0 for numerical stability.
        centroid = np.mean(centers_local, axis=0)
        centers_local_centered = centers_local - centroid
        return centers_local_centered, radii_local, centroid

    # Cluster 1: TL Corner (3 circles)
    # Global target approx (0.14, 0.14), (0.14, 0.42), (0.42, 0.14)
    c1_g = np.array([
        [0.14, 0.14],
        [0.14, 0.42],
        [0.42, 0.14]
    ])
    r1_g = np.array([0.14, 0.14, 0.14])
    c1_l, r1_l, c1_centroid = make_cluster(c1_g, r1_g)
    clusters.append((c1_l, r1_l))
    
    # Cluster 2: TR Corner (3 circles)
    c2_g = np.array([
        [0.86, 0.14],
        [0.86, 0.42],
        [0.58, 0.14]
    ])
    r2_g = np.array([0.14, 0.14, 0.14])
    c2_l, r2_l, c2_centroid = make_cluster(c2_g, r2_g)
    clusters.append((c2_l, r2_l))
    
    # Cluster 3: BL Corner (3 circles)
    c3_g = np.array([
        [0.14, 0.86],
        [0.14, 0.58],
        [0.42, 0.86]
    ])
    r3_g = np.array([0.14, 0.14, 0.14])
    c3_l, r3_l, c3_centroid = make_cluster(c3_g, r3_g)
    clusters.append((c3_l, r3_l))
    
    # Cluster 4: BR Corner (3 circles)
    c4_g = np.array([
        [0.86, 0.86],
        [0.86, 0.58],
        [0.58, 0.86]
    ])
    r4_g = np.array([0.14, 0.14, 0.14])
    c4_l, r4_l, c4_centroid = make_cluster(c4_g, r4_g)
    clusters.append((c4_l, r4_l))
    
    # Cluster 5: Central Hexagon (6 circles)
    # 1 center, 5 around? No, 6 in a ring?
    # Let's do 6 circles in a hexagonal arrangement around a center point.
    # r = 0.1
    r5 = 0.1
    c5_g = np.array([
        [0.5, 0.5],
        [0.5 + 2*r5*np.cos(0), 0.5 + 2*r5*np.sin(0)],
        [0.5 + 2*r5*np.cos(np.pi/3), 0.5 + 2*r5*np.sin(np.pi/3)],
        [0.5 + 2*r5*np.cos(2*np.pi/3), 0.5 + 2*r5*np.sin(2*np.pi/3)],
        [0.5 + 2*r5*np.cos(np.pi), 0.5 + 2*r5*np.sin(np.pi)],
        [0.5 + 2*r5*np.cos(4*np.pi/3), 0.5 + 2*r5*np.sin(4*np.pi/3)],
        [0.5 + 2*r5*np.cos(5*np.pi/3), 0.5 + 2*r5*np.sin(5*np.pi/3)]
    ])
    # That's 7 circles. We need 26 total. 4*3 + 7 = 19. Need 7 more.
    # Let's adjust counts.
    # 4 corners * 4 circles = 16.
    # Center * 10 circles = 10. Total 26.
    
    # Let's redefine.
    clusters = []
    
    # Corner Clusters (4 circles each)
    # Arrangement: 2x2 block in corner.
    r_corner = 0.11
    # Local coords for 2x2 block, centered.
    # (0,0), (2r, 0), (0, 2r), (2r, 2r) -> shifted to center (r, r)
    c_corner_g = np.array([
        [r_corner, r_corner],
        [r_corner + 2*r_corner, r_corner],
        [r_corner, r_corner + 2*r_corner],
        [r_corner + 2*r_corner, r_corner + 2*r_corner]
    ])
    r_corner_g = np.array([r_corner]*4)
    
    # TL
    c_tl, r_tl, _ = make_cluster(c_corner_g, r_corner_g)
    clusters.append((c_tl, r_tl))
    
    # TR
    c_tr_g = np.array([
        [1-r_corner, r_corner],
        [1 - (r_corner + 2*r_corner), r_corner],
        [1-r_corner, r_corner + 2*r_corner],
        [1 - (r_corner + 2*r_corner), r_corner + 2*r_corner]
    ])
    c_tr, r_tr, _ = make_cluster(c_tr_g, r_corner_g)
    clusters.append((c_tr, r_tr))
    
    # BL
    c_bl_g = np.array([
        [r_corner, 1-r_corner],
        [r_corner + 2*r_corner, 1-r_corner],
        [r_corner, 1 - (r_corner + 2*r_corner)],
        [r_corner + 2*r_corner, 1 - (r_corner + 2*r_corner)]
    ])
    c_bl, r_bl, _ = make_cluster(c_bl_g, r_corner_g)
    clusters.append((c_bl, r_bl))
    
    # BR
    c_br_g = np.array([
        [1-r_corner, 1-r_corner],
        [1 - (r_corner + 2*r_corner), 1-r_corner],
        [1-r_corner, 1 - (r_corner + 2*r_corner)],
        [1 - (r_corner + 2*r_corner), 1 - (r_corner + 2*r_corner)]
    ])
    c_br, r_br, _ = make_cluster(c_br_g, r_corner_g)
    clusters.append((c_br, r_br))
    
    # Central Cluster (10 circles)
    # 1 center + 9 around? Or 2 rings?
    # Let's do a hexagonal packing of 10 circles.
    # r_central = 0.09
    r_c = 0.09
    # Generate hexagonal grid points
    # Ring 0: (0,0)
    # Ring 1: 6 points at dist 2r
    # Ring 2: 12 points? No, 10 total.
    # 1 + 6 + 3? 
    # Let's just place them manually in a tight hex pattern.
    # Positions for 10 circles in hex packing:
    # Row 1: 2
    # Row 2: 3
    # Row 3: 3
    # Row 4: 2
    # Total 10.
    # Spacing dx = 2*r, dy = 2*r*sin(60)
    
    dx = 2*r_c
    dy = 2*r_c*np.sin(np.pi/3)
    
    c_cent_g = []
    # Row 1 (y=0): x=0, dx
    c_cent_g.append([0, 0])
    c_cent_g.append([dx, 0])
    # Row 2 (y=dy): x=-dx/2, dx/2, 3dx/2
    c_cent_g.append([-dx/2, dy])
    c_cent_g.append([dx/2, dy])
    c_cent_g.append([3*dx/2, dy])
    # Row 3 (y=2dy): x=-dx, 0, dx, 2dx? No, needs to fit.
    # Let's use a simpler 10-circle cluster: 5x2 grid? No, hex is better.
    # Let's use 1 center, 6 around, 3 more?
    # Actually, let's just use 10 circles in a 2x5 rectangular block for simplicity in initialization.
    # It will deform/rotate to fit.
    c_cent_g = []
    for i in range(5):
        for j in range(2):
            # Offset to center
            x = (i - 2) * dx
            y = (j - 0.5) * dy
            c_cent_g.append([x, y])
    c_cent_g = np.array(c_cent_g)
    r_cent_g = np.array([r_c]*10)
    
    c_cent, r_cent, _ = make_cluster(c_cent_g, r_cent_g)
    clusters.append((c_cent, r_cent))
    
    return clusters

def _assemble_clusters(clusters, poses):
    """
    Assemble global centers and radii from clusters and poses.
    poses: list of (tx, ty, theta)
    """
    centers = []
    radii = []
    for i, (c_local, r_local) in enumerate(clusters):
        tx, ty, theta = poses[i]
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate local centers
        c_rot = c_local @ R.T
        # Translate
        c_global = c_rot + np.array([tx, ty])
        
        centers.extend(c_global)
        radii.extend(r_local)
        
    return np.array(centers), np.array(radii)

def _penalty_objective(poses, clusters, weight_overlap=100.0, weight_wall=100.0):
    """
    Calculate penalty objective and gradient.
    """
    centers, radii = _assemble_clusters(clusters, poses)
    n = len(centers)
    
    penalty = 0.0
    grad = np.zeros(len(poses)*3)
    
    # Wall Penalties
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Left
        if x < r:
            pen = (r - x)**2
            penalty += weight_wall * pen
            # Grad w.r.t tx, ty, theta of cluster i
            # d(pen)/dx = -2(r-x)
            # dx/dtx = 1, dx/dty = 0, dx/dtheta = -y_local? 
            # We need chain rule.
            # This is complex. Let's use finite differences for gradient?
            # No, let's derive it or use scipy minimize with numerical grad.
            # Given time constraints, numerical gradient is safer.
            
    # Overlap Penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                pen = (min_dist - dist)**2
                penalty += weight_overlap * pen
                
    return penalty

def _optimize_clusters(clusters, time_limit, seed):
    """
    Optimize cluster poses using L-BFGS-B.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n_clusters = len(clusters)
    # Initial poses: centered in quadrants
    initial_poses = []
    centers_init = [0.25, 0.75, 0.25, 0.75, 0.5, 0.5]
    # Map clusters to these centers
    # 0: TL, 1: TR, 2: BL, 3: BR, 4: Center
    init_tx = [0.25, 0.75, 0.25, 0.75, 0.5]
    init_ty = [0.25, 0.25, 0.75, 0.75, 0.5]
    
    for i in range(n_clusters):
        tx = init_tx[i]
        ty = init_ty[i]
        theta = 0.0
        initial_poses.extend([tx, ty, theta])
        
    x0 = np.array(initial_poses)
    
    def obj(x):
        poses = []
        for i in range(n_clusters):
            poses.append((x[i*3], x[i*3+1], x[i*3+2]))
        return _penalty_objective(poses, clusters)
        
    # Add bounds
    bounds = []
    for i in range(n_clusters):
        bounds.append((0.0, 1.0)) # tx
        bounds.append((0.0, 1.0)) # ty
        bounds.append((0.0, 2*np.pi)) # theta
        
    start_time = time.time()
    
    # Use L-BFGS-B
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 50, 'ftol': 1e-8, 'maxfun': 1000})
    except:
        res = None
        
    if res is None or not res.success:
        # Fallback: simple gradient descent
        x = x0.copy()
        lr = 0.01
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            # Numerical gradient
            g = np.zeros_like(x)
            eps = 1e-5
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                g[i] = (obj(x_plus) - obj(x_minus)) / (2*eps)
            x -= lr * g
            # Project to bounds
            for i in range(n_clusters):
                x[i*3] = np.clip(x[i*3], 0, 1)
                x[i*3+1] = np.clip(x[i*3+1], 0, 1)
                x[i*3+2] = x[i*3+2] % (2*np.pi)
                
        return x
        
    return res.x

def _local_unlock(centers, radii, time_limit, seed):
    """
    Fine-tune individual circles to maximize radii.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n = len(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_score, _ = _is_valid(best_centers, best_radii)
    
    current_centers = centers.copy()
    current_radii = radii.copy()
    current_score, _ = _is_valid(current_centers, current_radii)
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit and T > T_min:
        move_type = random.choice(['jiggle', 'grow', 'swap'])
        i = random.randint(0, n-1)
        
        new_centers = current_centers.copy()
        new_radii = current_radii.copy()
        
        if move_type == 'jiggle':
            delta = np.random.randn(2) * 0.02
            new_centers[i] = current_centers[i] + delta
        elif move_type == 'grow':
            new_radii[i] = current_radii[i] * (1 + 0.05)
        elif move_type == 'swap':
            j = random.randint(0, n-1)
            if i != j:
                new_radii[i], new_radii[j] = new_radii[j], new_radii[i]
                
        new_score, new_valid = _is_valid(new_centers, new_radii)
        
        if new_valid:
            if new_score > current_score:
                current_centers = new_centers
                current_radii = new_radii
                current_score = new_score
                if current_score > best_score:
                    best_centers = current_centers.copy()
                    best_radii = current_radii.copy()
                    best_score = current_score
            else:
                delta = current_score - new_score
                if delta < 0: delta = 0
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_centers = new_centers
                    current_radii = new_radii
                    current_score = new_score
        else:
            prob = np.exp(-10 / T)
            if random.random() < prob:
                current_centers = new_centers
                current_radii = new_radii
                current_score = -np.inf
                
        T *= cooling
        
    return best_centers, best_radii

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_centers = None
    best_radii = None
    best_score = -np.inf
    
    clusters = _get_cluster_geometry()
    
    # Multi-start with different seeds
    for trial in range(3):
        if time.time() > deadline:
            break
            
        seed = trial * 100 + 42
        
        # Phase 1: Cluster Optimization
        poses = _optimize_clusters(clusters, time_limit=20.0, seed=seed)
        
        # Assemble
        centers, radii = _assemble_clusters(clusters, poses)
        
        # Check validity
        score, valid = _is_valid(centers, radii)
        
        # If invalid, try to repair by shrinking
        if not valid:
            # Simple repair: shrink all radii by 10%
            radii *= 0.9
            score, valid = _is_valid(centers, radii)
            
        if valid:
            # Phase 2: Local Unlock
            c_opt, r_opt = _local_unlock(centers, radii, time_limit=20.0, seed=seed+100)
            score_opt, valid_opt = _is_valid(c_opt, r_opt)
            
            if valid_opt and score_opt > best_score:
                best_score = score_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
                
    if best_centers is None:
        # Fallback: Grid
        side = 5
        best_centers = np.zeros((N, 2))
        best_radii = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            best_centers[i, 0] = (gx + 0.5) / side
            best_centers[i, 1] = (gy + 0.5) / side
            best_radii[i] = 0.5 / side - 0.01
            
    # Final validity check and repair
    result_c = best_centers.copy()
    result_r = best_radii.copy()
    
    for i in range(N):
        x, y, ri = result_c[i, 0], result_c[i, 1], result_r[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if ri > max_r:
            ri = max(0.001, max_r)
        result_r[i] = ri
        result_c[i, 0] = np.clip(x, ri, 1.0 - ri)
        result_c[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    # Resolve overlaps
    for _ in range(20):
        modified = False
        for i in range(N):
            for j in range(i+1, N):
                dx = result_c[i, 0] - result_c[j, 0]
                dy = result_c[i, 1] - result_c[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = result_r[i] + result_r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    if result_r[i] < result_r[j]:
                        result_r[i] = max(0.001, result_r[i] - overlap/2)
                    else:
                        result_r[j] = max(0.001, result_r[j] - overlap/2)
                    modified = True
        if not modified:
            break
            
    result = np.column_stack([result_c, result_r])
    return result

# EVOLVE-BLOCK-END
