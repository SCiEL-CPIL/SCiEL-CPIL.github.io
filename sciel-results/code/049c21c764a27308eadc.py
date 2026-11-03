import numpy as np
import scipy.optimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    def energy(x):
        # x is shape (2*N,)
        pts = x.reshape(N, 2)
        # Pairwise distances
        # Compute squared distances
        # Use broadcasting: (N, 1, 2) - (1, N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero on diagonal
        dist = np.sqrt(dist_sq)
        
        # Sum of inverse distances (or inverse squares)
        # Using 1/d^2 is more aggressive
        inv_dist_sq = 1.0 / dist_sq
        
        # Sum over unique pairs (i < j)
        # We can just sum all and divide by 2, or use triu_indices
        i, j = np.triu_indices(N, k=1)
        pair_energy = np.sum(inv_dist_sq[i, j])
        
        # Boundary energy
        # dist_to_boundary = 1 - norm(pts)
        norms = np.linalg.norm(pts, axis=1)
        # Avoid division by zero if pts are at origin? No, 1-0=1.
        # If pts are outside, norms > 1, 1-norms < 0.
        # We want to penalize being outside or close to boundary.
        # If norms > 1, it's invalid. We should penalize heavily.
        # Let's use a penalty for norms > 1 and a repulsion for norms close to 1.
        # Actually, the problem requires circles INSIDE.
        # So we must ensure norms + r <= 1.
        # The energy function should push them in.
        
        # Let's use: sum(1 / (1 - norm)^2) for norm < 1
        # If norm >= 1, add a large penalty.
        margin = 1.0 - norms
        # Clip margin to avoid division by zero or negative
        # If margin is small, energy is huge.
        # If margin is negative (outside), energy is huge.
        # Use max(margin, 0.01) to keep it positive and finite?
        # No, if outside, we want infinite penalty.
        # Let's use: if margin < 0.01, penalty = 10000 * (0.01 - margin)^2
        # Else penalty = 1 / margin^2
        
        # Simpler: Use a smooth potential that goes to infinity at boundary.
        # V(r) = 1 / (1 - r^2)^2 ?
        # Let's stick to: sum(1 / max(1 - norm, 0.05)^2)
        # This creates a hard wall at 0.95, which is too early.
        # We want the wall at 1.0.
        
        # Let's use:
        # boundary_term = np.sum(1.0 / np.maximum(1.0 - norms, 1e-6)**2)
        # This will be huge if close to boundary.
        
        boundary_energy = np.sum(1.0 / np.maximum(1.0 - norms, 1e-6)**2)
        
        return pair_energy + boundary_energy

    def get_radius(pts):
        if len(pts) != N:
            return 0.0
        # Check if all inside
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms >= 1.0):
            return 0.0
        # Min distance to boundary
        r_bound = np.min(1.0 - norms)
        
        # Min distance between circles
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        i, j = np.triu_indices(N, k=1)
        min_pair_dist = np.min(dist[i, j])
        r_pair = min_pair_dist / 2.0
        
        return min(r_bound, r_pair)

    best_radius = 0.0
    best_pts = None
    
    # Initialization strategies
    inits = []
    
    # 1. Random
    for _ in range(5):
        # Generate random points in unit circle
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        inits.append(pts.flatten())
        
    # 2. Hexagonal Lattice (from seed)
    spacing = 0.2
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points, radius = points[radius <= 1.0], radius[radius <= 1.0]
    if len(points) >= N:
        inits.append(points[:N].flatten())
        
    # 3. Spiral
    t = np.linspace(0, 1, N)
    r = np.sqrt(t)
    theta = t * 20 * np.pi
    pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    inits.append(pts.flatten())

    # Run optimization for each init
    for x0 in inits:
        if time.time() > deadline:
            break
            
        try:
            res = scipy.optimize.minimize(energy, x0, method='Powell', options={'maxiter': 200, 'disp': False})
            x_opt = res.x
            pts_opt = x_opt.reshape(N, 2)
            
            # Verify and calculate radius
            r_val = get_radius(pts_opt)
            
            if r_val > best_radius:
                best_radius = r_val
                best_pts = pts_opt
                
        except Exception:
            pass

    # If no valid point found, fallback to seed logic
    if best_pts is None:
        # Use the seed logic as fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    return best_pts
