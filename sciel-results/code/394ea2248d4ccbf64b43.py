"""Physics-based circle packing in perimeter-4 rectangle using Lennard-Jones + velocity-Verlet."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _check_valid(circles, tol=1e-7):
    """Check if circles are valid: disjoint, inside bounding rect with perimeter <= 4."""
    if circles.shape[0] != N:
        return False
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(rs < -1e-9):
        return False
    rs = np.maximum(rs, 0)
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    if 2 * (w + h) > 4 + tol:
        return False
    # Check pairwise separation
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if d < rs[i] + rs[j] - tol:
                return False
    return True


def _lj_potential_and_force(r, sigma, eps):
    """Lennard-Jones potential and force magnitude for distance r."""
    r6 = sigma**6 / r**6
    r12 = r6 * r6
    U = 4 * eps * (r12 - r6)
    # Force magnitude: -dU/dr = 4*eps*(12*r12/r - 6*r6/r) = 24*eps*(2*r12 - r6)/r
    F = 24 * eps * (2 * r12 - r6) / r
    return U, F


def _physics_simulate(W, H, r0, n_steps=5000, dt=0.002, cooling=0.9995, seed=42):
    """
    Run a physics simulation with Lennard-Jones repulsion and soft walls.
    
    Parameters:
    - W, H: box dimensions
    - r0: initial radius for all circles (will be adjusted)
    - n_steps: number of simulation steps
    - dt: time step
    - cooling: velocity damping factor per step
    - seed: random seed
    
    Returns: (circles, success) where circles is (N,3) array
    """
    rng = np.random.RandomState(seed)
    
    # Initialize positions in a grid-like pattern with small random jitter
    n_per_row = int(np.ceil(N / max(1, int(np.round(W / H))))) if H > 0 else 5
    n_rows = int(np.ceil(N / n_per_row))
    
    xs = np.zeros(N)
    ys = np.zeros(N)
    for i in range(N):
        row = i // n_per_row
        col = i % n_per_row
        xs[i] = (col + 0.5) * W / n_per_row + rng.randn() * 0.01
        ys[i] = (row + 0.5) * H / n_rows + rng.randn() * 0.01
    
    # Clip to valid range
    xs = np.clip(xs, r0, W - r0)
    ys = np.clip(ys, r0, H - r0)
    
    # Radii start at r0
    rs = np.full(N, r0)
    
    # Velocities
    vx = np.zeros(N)
    vy = np.zeros(N)
    vr = np.zeros(N)
    
    # LJ parameters: sigma = sum of radii (contact distance), eps = energy scale
    # We want strong repulsion when overlapping
    sigma_scale = 1.0  # sigma = r_i + r_j
    eps = 1.0
    
    # Wall stiffness: quadratic penalty for going outside
    wall_k = 100.0
    
    best_circles = None
    best_overlap = np.inf
    best_sum_r = 0.0
    
    for step in range(n_steps):
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Pairwise LJ forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-10:
                    # Add small random push
                    dx = rng.randn() * 0.001
                    dy = rng.randn() * 0.001
                    d = np.sqrt(dx*dx + dy*dy)
                
                sigma = rs[i] + rs[j]
                # Only apply force if within range (avoid huge forces at large distances)
                if d < 2.5 * sigma:
                    r6 = (sigma**6) / (d**6)
                    r12 = r6 * r6
                    F_mag = 24 * eps * (2 * r12 - r6) / d
                    fx[i] += F_mag * dx / d
                    fy[i] += F_mag * dy / d
                    fx[j] -= F_mag * dx / d
                    fy[j] -= F_mag * dy / d
                    # Radial force: push to increase radius if not overlapping, decrease if too far
                    # Actually, we want to maximize sum of radii, so add a small bias to increase r
                    # But constrained by non-overlap. Let's handle radius separately.
        
        # Wall forces (soft repulsion from boundaries)
        for i in range(N):
            # Left wall: x - r >= 0
            viol = max(0, rs[i] - xs[i])
            if viol > 0:
                fx[i] += wall_k * viol
            # Right wall: x + r <= W
            viol = max(0, xs[i] + rs[i] - W)
            if viol > 0:
                fx[i] -= wall_k * viol
            # Bottom wall
            viol = max(0, rs[i] - ys[i])
            if viol > 0:
                fy[i] += wall_k * viol
            # Top wall
            viol = max(0, ys[i] + rs[i] - H)
            if viol > 0:
                fy[i] -= wall_k * viol
        
        # Radius dynamics: try to increase radii, but penalize overlaps
        for i in range(N):
            # Gradient of overlap penalty w.r.t. r_i
            overlap_pen = 0.0
            for j in range(N):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-10:
                    d = 1e-10
                req = rs[i] + rs[j]
                if d < req:
                    overlap_pen += (req - d)**2
            # fr = -d(overlap_pen)/dr_i + small growth bias
            # d(overlap_pen)/dr_i = 2 * sum_j (req - d) * 1 for j where d < req
            grad_r = 0.0
            for j in range(N):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-10:
                    d = 1e-10
                req = rs[i] + rs[j]
                if d < req:
                    grad_r += 2 * (req - d)
            # Bias to grow: -grad_r + growth_rate
            growth_rate = 0.5
            fr[i] = -grad_r + growth_rate
        
        # Velocity Verlet integration
        # v += F*dt
        vx += fx * dt
        vy += fy * dt
        vr += fr * dt
        
        # Damping
        vx *= cooling
        vy *= cooling
        vr *= cooling
        
        # x += v*dt
        xs += vx * dt
        ys += vy * dt
        rs += vr * dt
        
        # Clamp radii to positive
        rs = np.maximum(rs, 0.001)
        
        # Clamp positions to stay in box (hard constraint for stability)
        for i in range(N):
            xs[i] = np.clip(xs[i], rs[i], W - rs[i])
            ys[i] = np.clip(ys[i], rs[i], H - rs[i])
            # If clamped, zero out velocity in that direction
            if xs[i] == rs[i]:
                vx[i] = max(0, vx[i])
            if xs[i] == W - rs[i]:
                vx[i] = min(0, vx[i])
            if ys[i] == rs[i]:
                vy[i] = max(0, vy[i])
            if ys[i] == H - rs[i]:
                vy[i] = min(0, vy[i])
        
        # Check validity and track best
        if step % 50 == 0:
            circles = np.column_stack([xs, ys, rs])
            if _check_valid(circles):
                s = np.sum(rs)
                if s > best_sum_r:
                    best_sum_r = s
                    best_circles = circles.copy()
    
    if best_circles is not None:
        return best_circles, True
    return None, False


def _polish_with_scipy(circles, W, H, deadline=None):
    """Polish using scipy minimize with penalty method."""
    params0 = circles.flatten()
    
    def obj(p):
        c = p.reshape(N, 3)
        xs, ys, rs = c[:, 0], c[:, 1], c[:, 2]
        pen = 0.0
        # Boundary
        for i in range(N):
            pen += max(0, rs[i] - xs[i])**2
            pen += max(0, xs[i] + rs[i] - W)**2
            pen += max(0, rs[i] - ys[i])**2
            pen += max(0, ys[i] + rs[i] - H)**2
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                pen += max(0, rs[i]+rs[j]-d)**2
        return -np.sum(rs) + 1e4 * pen
    
    try:
        res = minimize(obj, params0, method='L-BFGS-B',
                       options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12})
        new_c = res.x.reshape(N, 3)
        if _check_valid(new_c):
            return new_c
    except Exception:
        pass
    return circles


def _make_initial_grid(W, H, seed=0):
    """Create a reasonable initial grid configuration."""
    rng = np.random.RandomState(seed)
    
    # Try to fit N circles in a grid
    # Estimate grid dimensions
    aspect = W / H if H > 0 else 1
    if aspect >= 1:
        n_cols = int(np.round(np.sqrt(N * aspect)))
    else:
        n_cols = int(np.round(np.sqrt(N / aspect)))
    n_cols = max(1, min(N, n_cols))
    n_rows = int(np.ceil(N / n_cols))
    
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    cell_w = W / n_cols
    cell_h = H / n_rows
    
    for i in range(N):
        row = i // n_cols
        col = i % n_cols
        # Hexagonal offset for odd rows
        x_off = 0.0
        if row % 2 == 1:
            x_off = 0.5 * cell_w
        xs[i] = (col + 0.5) * cell_w + x_off
        ys[i] = (row + 0.5) * cell_h
        # Radius limited by cell size
        rs[i] = 0.5 * min(cell_w, cell_h) * 0.95
    
    # Add small jitter
    xs += rng.randn(N) * 0.005
    ys += rng.randn(N) * 0.005
    
    # Clamp
    for i in range(N):
        xs[i] = np.clip(xs[i], rs[i], W - rs[i])
        ys[i] = np.clip(ys[i], rs[i], H - rs[i])
    
    return np.column_stack([xs, ys, rs])


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    best_circles = None
    best_sum = 0.0
    
    # Try different aspect ratios
    # Perimeter = 2*(W+H) = 4, so W+H = 2
    aspect_configs = [
        (1.0, 1.0),    # square
        (1.2, 0.8),
        (0.8, 1.2),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.1, 0.9),
        (0.9, 1.1),
        (1.25, 0.75),
        (0.75, 1.25),
    ]
    
    for W, H in aspect_configs:
        if time.time() > deadline - 15:
            break
        
        # Try multiple initial configurations and seeds
        for seed in [42, 123, 777]:
            if time.time() > deadline - 10:
                break
            
            # Create initial grid
            circles0 = _make_initial_grid(W, H, seed=seed)
            
            # Estimate a reasonable initial radius
            # For N circles in W x H, average area per circle = W*H/N
            # Average radius ~ sqrt(W*H/(N*pi))
            r_avg = np.sqrt(W * H / (N * np.pi))
            r0 = min(0.5 * min(W, H) / np.ceil(np.sqrt(N)), r_avg * 1.2)
            r0 = max(0.02, min(r0, 0.5 * min(W, H) * 0.9))
            
            # Run physics simulation
            result, success = _physics_simulate(W, H, r0, n_steps=3000, dt=0.003, 
                                                 cooling=0.999, seed=seed)
            
            if success and result is not None:
                if _check_valid(result):
                    s = np.sum(result[:, 2])
                    if s > best_sum:
                        best_sum = s
                        best_circles = result.copy()
            
            # Polish with scipy if we have time
            if time.time() < deadline - 5 and best_circles is not None:
                polished = _polish_with_scipy(best_circles, W, H, deadline=deadline - 2)
                if _check_valid(polished):
                    s = np.sum(polished[:, 2])
                    if s > best_sum:
                        best_sum = s
                        best_circles = polished.copy()
        
        # Also try starting from the best found so far for this aspect ratio
        if best_circles is not None and time.time() < deadline - 8:
            # Refine with more steps
            result2, success2 = _physics_simulate(W, H, 
                                                   np.mean(best_circles[:, 2]), 
                                                   n_steps=2000, dt=0.002,
                                                   cooling=0.9995, seed=999)
            if success2 and result2 is not None:
                if _check_valid(result2):
                    s = np.sum(result2[:, 2])
                    if s > best_sum:
                        best_sum = s
                        best_circles = result2.copy()
    
    # Final polish
    if best_circles is not None and time.time() < deadline - 3:
        xs, ys, rs = best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        W = max_x - min_x
        H = max_y - min_y
        best_circles[:, 0] -= min_x
        best_circles[:, 1] -= min_y
        
        polished = _polish_with_scipy(best_circles, W, H, deadline=deadline - 1)
        if _check_valid(polished):
            s = np.sum(polished[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = polished.copy()
    
    if best_circles is not None:
        best_circles[:, 2] = np.maximum(best_circles[:, 2], 0.0)
        return best_circles
    
    # Fallback: simple grid
    W, H = 1.0, 1.0
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(N):
        row = i // side
        col = i % side
        cx = (col + 0.5) / side
        cy = (row + 0.5) / side
        r = 0.5 / side - 0.005
        cells.append([cx, cy, r])
    return np.array(cells)
