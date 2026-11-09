import numpy as np
import time

N = 200

def compute_radius(points):
    if points is None:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return float(r)

def hex_init(n_rows):
    """Generate a hexagonal lattice initialization with n_rows rows."""
    c_base = N // n_rows
    rem = N % n_rows
    counts = [c_base] * n_rows
    for i in range(rem):
        counts[i] += 1
    
    pts = []
    y = 0.0
    for i, c in enumerate(counts):
        xs = np.arange(c) - (c - 1) / 2.0
        if i % 2 == 1:
            xs += 0.5
        for x in xs:
            pts.append([x, y])
        y += np.sqrt(3) / 2.0
    
    pts = np.array(pts, dtype=np.float64)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    max_extent = max(np.max(np.abs(pts[:, 0])), np.max(np.abs(pts[:, 1])))
    if max_extent > 0:
        scale = 0.48 / max_extent
        pts *= scale
    return pts

def anneal(pts, deadline, max_steps=3000):
    """Physics-based annealing: repulsive forces + wall constraints."""
    pts = pts.copy()
    n_steps = max_steps
    t0 = time.time()
    
    # Annealing schedule: temperature decays
    T_start = 0.01
    T_end = 0.0001
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Compute pairwise repulsive forces (inverse square)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Only consider pairs closer than threshold for efficiency
        threshold = 0.12
        mask = dist < threshold
        force_mag = np.zeros_like(dist)
        force_mag[mask] = 1.0 / (dist[mask] ** 2 + 1e-12)
        
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion
        x = pts[:, 0]
        y = pts[:, 1]
        wall_k = 5.0
        wall_margin = 0.04
        
        # Left wall
        d_left = x + 0.5
        wall_fx = np.where(d_left < wall_margin, wall_k * (wall_margin - d_left), 0.0)
        # Right wall
        d_right = 0.5 - x
        wall_fx -= np.where(d_right < wall_margin, wall_k * (wall_margin - d_right), 0.0)
        # Bottom wall
        d_bottom = y + 0.5
        wall_fy = np.where(d_bottom < wall_margin, wall_k * (wall_margin - d_bottom), 0.0)
        # Top wall
        d_top = 0.5 - y
        wall_fy -= np.where(d_top < wall_margin, wall_k * (wall_margin - d_top), 0.0)
        
        forces[:, 0] += wall_fx
        forces[:, 1] += wall_fy
        
        # Temperature for random perturbation (annealing)
        T = T_start * (T_end / T_start) ** (step / max(1, n_steps))
        
        # Update
        lr = 0.005
        pts += lr * forces + np.random.randn(N, 2) * T
        pts = np.clip(pts, -0.499, 0.499)
    
    return pts

def polish_softmin(pts, deadline, max_steps=2000):
    """Fine-tune using soft-min gradient descent."""
    pts = pts.copy()
    lambda_val = 200.0
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        w_pairs = np.exp(-lambda_val * dist_sq)
        w_pairs[~np.isfinite(w_pairs)] = 0
        
        sum_w = np.sum(w_pairs, axis=1)
        term = w_pairs[:, :, np.newaxis] * (-2 * lambda_val) * diff
        grad_pairs = np.sum(term, axis=1)
        sum_w_safe = np.where(sum_w > 0, sum_w, 1.0)
        grad_pairs /= sum_w_safe[:, np.newaxis]
        
        # Wall gradients
        x = pts[:, 0]
        y = pts[:, 1]
        d_xl = x + 0.5
        d_xr = 0.5 - x
        d_yb = y + 0.5
        d_yt = 0.5 - y
        
        wxl = np.exp(-lambda_val * d_xl**2)
        wxr = np.exp(-lambda_val * d_xr**2)
        wyb = np.exp(-lambda_val * d_yb**2)
        wyt = np.exp(-lambda_val * d_yt**2)
        
        sw = wxl + wxr + wyb + wyt
        sw_safe = np.where(sw > 0, sw, 1.0)
        
        gx = (wxl * (-2*lambda_val*d_xl) + wxr * (2*lambda_val*d_xr)) / sw_safe
        gy = (wyb * (-2*lambda_val*d_yb) + wyt * (2*lambda_val*d_yt)) / sw_safe
        
        total_grad = np.column_stack((gx, gy)) + grad_pairs
        
        lr = 0.002 / (1 + step * 0.0005)
        pts_new = pts - lr * total_grad
        pts_new = np.clip(pts_new, -0.5, 0.5)
        pts = pts_new
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    best_r = 0.0
    best_points = None
    
    def try_improve(pts):
        nonlocal best_r, best_points
        r = compute_radius(pts)
        if r > best_r:
            best_r = r
            best_points = pts.copy()
        return r
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal packings with different row counts
    for n_rows in [12, 13, 14, 15, 16, 17, 18, 19, 20]:
        inits.append(hex_init(n_rows))
    
    # A few random starts
    for _ in range(10):
        inits.append(np.random.rand(N, 2) - 0.5)
    
    # Shuffle to mix types
    np.random.shuffle(inits)
    
    # Process each initialization with anneal + polish
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        pts = init.copy()
        
        # Phase 1: Annealing (repulsive forces)
        pts = anneal(pts, deadline)
        try_improve(pts)
        
        # Phase 2: Soft-min polish
        if time.time() < deadline - 5:
            pts = polish_softmin(pts, deadline)
            try_improve(pts)
    
    # If we have time, do additional refinement on best
    if best_points is not None and time.time() < deadline - 10:
        pts = best_points.copy()
        pts = polish_softmin(pts, deadline)
        try_improve(pts)
    
    if best_points is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    return best_points
