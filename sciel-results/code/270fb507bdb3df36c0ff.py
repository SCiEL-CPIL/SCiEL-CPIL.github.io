"""Force-directed circle packing with adaptive cooling and polish."""
import numpy as np
import time

N = 50
DEADLINE = 120.0  # wall-clock seconds (evaluator kills at 150s, keep margin)

def _score(pts):
    """Compute the packing radius r = min(min pairwise dist/2, min (1-|c_i|))."""
    n = len(pts)
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    boundary = 1.0 - np.sqrt(np.sum(pts * pts, axis=1))
    return min(min_pair, np.min(boundary))

def _repulsive_force(pts, target_r, k_rep, k_attr, k_boundary):
    """
    Compute forces for force-directed layout.
    - Repulsive: pushes apart pairs closer than 2*target_r
    - Attractive: pulls together pairs farther than 2*target_r (keeps packing dense)
    - Boundary: pushes inward any point too close to edge
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Pairwise repulsion/attraction
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.sqrt(np.sum(d * d))
            if dist < 1e-12:
                # random nudge
                d = np.random.randn(2) * 1e-8
                dist = np.sqrt(np.sum(d * d))
            # Ideal distance is 2*target_r
            ideal = 2.0 * target_r
            # Repulsive if too close, attractive if too far
            if dist < ideal:
                f = k_rep * (ideal - dist) / dist
            else:
                f = -k_attr * (dist - ideal) / dist
            forces[i] += f * d
            forces[j] -= f * d
    
    # Boundary force: push inward
    for i in range(n):
        r = np.sqrt(np.sum(pts[i] * pts[i]))
        if r > 1.0 - target_r:
            # Push toward center
            if r > 1e-12:
                forces[i] += k_boundary * (r - (1.0 - target_r)) * (pts[i] / r)
            else:
                forces[i] += k_boundary * (r - (1.0 - target_r)) * np.array([1.0, 0.0])
    
    return forces

def _hexagonal_init():
    """Generate initial hexagonal lattice points inside unit circle."""
    spacing = 0.25
    points = []
    for j in range(-6, 7):
        for i in range(-6, 7):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.sqrt(x*x + y*y) < 1.0 - spacing/2:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= N:
        # Sort by radius, take first N
        radii = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(radii)[:N]
        return points[idx]
    # Fallback: random
    return np.random.randn(N, 2) * 0.5

def _random_init():
    """Random initialization inside unit circle."""
    angles = np.random.uniform(0, 2*np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.9
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _force_directed(pts, target_r, max_iter, dt_init, cooling, deadline):
    """Run force-directed layout with adaptive cooling."""
    n = len(pts)
    pts = pts.copy()
    dt = dt_init
    best_r = _score(pts)
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        forces = _repulsive_force(pts, target_r, k_rep=1.0, k_attr=0.1, k_boundary=5.0)
        
        # Limit step size
        max_force = np.max(np.sqrt(np.sum(forces**2, axis=1)))
        if max_force > 1e-12:
            step = forces / max_force * dt
        else:
            step = np.zeros_like(pts)
        
        pts_new = pts + step
        
        # Check validity: all inside circle
        radii = np.sqrt(np.sum(pts_new**2, axis=1))
        if np.any(radii > 1.0):
            # Scale back
            scale = 0.99 / np.max(radii)
            pts_new = pts_new * scale
        
        r_new = _score(pts_new)
        
        if r_new > best_r:
            best_r = r_new
            pts = pts_new
        else:
            # Reject, cool down
            pass
        
        dt *= cooling
    
    return pts, best_r

def _polish(pts, deadline):
    """Final high-precision polish using gradient ascent on the min objective."""
    n = len(pts)
    pts = pts.copy()
    best_r = _score(pts)
    
    # Use log-sum-exp soft-min for smooth optimization
    beta = 50.0  # high beta for sharp min approximation
    
    for it in range(500):
        if time.time() > deadline - 3.0:
            break
        
        # Compute all constraints
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        pair_vals = d / 2.0  # candidate radii from pairs
        
        boundary_vals = 1.0 - np.sqrt(np.sum(pts * pts, axis=1))
        
        # Soft-min of all constraints
        all_vals = np.concatenate([pair_vals.ravel(), boundary_vals.ravel()])
        # Use log-sum-exp for soft-min: -log-sum-exp(-beta * x) / beta
        neg_beta_x = -beta * all_vals
        max_neg = np.max(neg_beta_x)
        log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_beta_x - max_neg)))
        soft_min = -log_sum_exp / beta
        
        # Gradient of soft-min w.r.t. positions
        # d(soft_min)/d(pts_i) = -sum_j w_j * d(pair_val_ij)/d(pts_i) - w_boundary * d(boundary_i)/d(pts_i)
        # where w_j are softmax weights
        
        # Pairwise gradients
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if d[i, j] < 1e-12:
                    continue
                w = np.exp(-beta * pair_vals[i, j] - log_sum_exp)  # softmax weight
                # d(pair_val_ij)/d(pts_i) = (pts_i - pts_j) / d_ij
                grad[i] += w * (pts[i] - pts[j]) / d[i, j]
        
        # Boundary gradients
        for i in range(n):
            r = np.sqrt(np.sum(pts[i] * pts[i]))
            if r < 1e-12:
                continue
            w_b = np.exp(-beta * boundary_vals[i] - log_sum_exp)
            # d(boundary_i)/d(pts_i) = -pts_i / r
            grad[i] += w_b * (-pts[i] / r)
        
        # Gradient ascent: pts += lr * grad
        lr = 0.01
        pts_new = pts + lr * grad
        
        # Project back inside circle
        radii = np.sqrt(np.sum(pts_new**2, axis=1))
        for i in range(n):
            if radii[i] > 1.0 - 1e-10:
                pts_new[i] = pts_new[i] * (1.0 - 1e-10) / radii[i]
        
        r_new = _score(pts_new)
        if r_new > best_r:
            best_r = r_new
            pts = pts_new
        else:
            lr *= 0.9  # reduce learning rate
    
    return pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 10.0  # safety margin
    
    best_pts = None
    best_r = 0.0
    
    # Multiple restarts with different initializations
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            pts = _hexagonal_init()
        else:
            pts = _random_init()
        
        # Initial score
        r_init = _score(pts)
        if r_init > best_r:
            best_r = r_init
            best_pts = pts.copy()
        
        # Force-directed with adaptive target radius
        # Start with a slightly smaller target, increase over iterations
        target_r = r_init * 0.95 if r_init > 0 else 0.1
        
        # Phase 1: coarse force-directed
        max_iter_coarse = 200
        dt_init = 0.1
        cooling = 0.995
        pts_fd, r_fd = _force_directed(pts, target_r, max_iter_coarse, dt_init, cooling, deadline)
        
        if r_fd > best_r:
            best_r = r_fd
            best_pts = pts_fd.copy()
        
        # Phase 2: increase target radius and refine
        if time.time() > deadline - 10.0:
            break
        
        target_r2 = r_fd * 1.02
        max_iter_fine = 100
        dt_init2 = 0.05
        cooling2 = 0.99
        pts_fd2, r_fd2 = _force_directed(pts_fd, target_r2, max_iter_fine, dt_init2, cooling2, deadline)
        
        if r_fd2 > best_r:
            best_r = r_fd2
            best_pts = pts_fd2.copy()
        
        # Phase 3: polish
        if time.time() > deadline - 5.0:
            break
        
        pts_pol, r_pol = _polish(pts_fd2, deadline)
        
        if r_pol > best_r:
            best_r = r_pol
            best_pts = pts_pol.copy()
    
    # Final polish on best result
    if time.time() < deadline - 3.0 and best_pts is not None:
        pts_final, r_final = _polish(best_pts, deadline)
        if r_final > best_r:
            best_r = r_final
            best_pts = pts_final.copy()
    
    # Ensure validity: project any points outside the circle
    if best_pts is not None:
        radii = np.sqrt(np.sum(best_pts**2, axis=1))
        for i in range(len(best_pts)):
            if radii[i] > 1.0:
                best_pts[i] = best_pts[i] * (1.0 - 1e-10) / radii[i]
    
    return best_pts if best_pts is not None else _hexagonal_init()
