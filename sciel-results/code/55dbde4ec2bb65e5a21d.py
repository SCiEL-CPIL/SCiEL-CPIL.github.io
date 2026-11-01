"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _score(pts):
    """Compute the objective: largest common radius."""
    n = len(pts)
    # pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # distance to boundary
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    min_bound = 1.0 - np.max(radii)
    return min(min_pair, min_bound)


def _hex_lattice_init(n):
    """Generate a hexagonal lattice packing inside the unit circle."""
    # Try various spacings
    best_pts = None
    best_score = 0.0
    for trial in range(20):
        spacing = 0.22 - trial * 0.003
        if spacing < 0.1:
            break
        pts = []
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.sqrt(x * x + y * y)
                if r <= 1.0 - spacing / 2.0:
                    pts.append((x, y))
        pts = np.array(pts)
        if len(pts) >= n:
            # Select n points closest to center for better packing
            radii = np.sqrt(np.sum(pts ** 2, axis=1))
            idx = np.argsort(radii)[:n]
            pts = pts[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    if best_pts is None:
        # Fallback: uniform random
        best_pts = np.random.randn(n, 2) * 0.8
    return best_pts


def _repulsive_energy(pts, softness=0.01):
    """Compute a smooth surrogate objective using log-sum-exp for soft-min."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e10)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    bound = 1.0 - radii
    
    # Soft min of pairwise distances (we want to maximize the min)
    # Use log-sum-exp trick: soft_min = -log-sum-exp(-dist/softness) * softness
    # But we want to maximize min, so minimize -soft_min
    # Actually, let's use a different approach: 
    # We want to maximize min(pairwise_dists/2, boundary_dists)
    # Equivalently, minimize -min(...)
    # Use soft approximation: -soft_min = log-sum-exp(-values/softness) * softness
    
    all_vals = np.concatenate([dist / 2.0, bound])
    # Remove diagonal entries (they're 1e10, so they won't affect min)
    all_vals = all_vals[all_vals < 1e9]
    
    # Soft min via log-sum-exp
    m = np.min(all_vals)
    soft_min = m - softness * np.log(np.sum(np.exp((all_vals - m) / softness)))
    # Wait, that's soft max. For soft min:
    # soft_min = -soft_max(-vals) = -(-m - softness * log(sum(exp((-(-vals) - (-m))/softness))))
    # = m - softness * log(sum(exp((all_vals - m)/softness)))
    # Hmm, let me just use: soft_min ≈ m - softness * log(1 + sum(exp((v-m)/softness)))
    # Actually the standard soft min is:
    # soft_min(x) = -log(sum(exp(-x_i / t))) * t
    neg_vals = -all_vals
    m2 = np.min(neg_vals)
    soft_min_val = -t * np.log(np.sum(np.exp((neg_vals - m2) / t)))
    # This gives a value close to min(all_vals)
    
    return -soft_min_val  # We want to minimize this (maximize soft_min)


def _grad_repulsive(pts, softness=0.01):
    """Gradient of the repulsive energy."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e10)
    
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    bound = 1.0 - radii
    
    all_vals = np.concatenate([dist / 2.0, bound])
    all_vals = all_vals[all_vals < 1e9]
    
    # Soft min gradient
    t = softness
    neg_vals = -all_vals
    m2 = np.min(neg_vals)
    exps = np.exp((neg_vals - m2) / t)
    weights = exps / np.sum(exps)
    
    # Gradient of soft_min w.r.t. each value is -weights
    # But we need gradient w.r.t. pts
    
    grad = np.zeros_like(pts)
    
    # Pairwise contributions
    # For each pair i<j, contribution to soft_min from dist_ij/2
    # d(soft_min)/d(dist_ij) = -w_ij
    # d(dist_ij)/d(pts_i) = (pts_i - pts_j) / dist_ij
    # So d(soft_min)/d(pts_i) += -w_ij * (pts_i - pts_j) / dist_ij
    # And d(soft_min)/d(pts_j) += -w_ij * (pts_j - pts_i) / dist_ij
    
    # This is getting complex. Let's use finite differences or a simpler approach.
    # For now, use a simple repulsive force model.
    
    # Simple repulsive forces
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist_ij = np.sqrt(np.sum(d ** 2))
            if dist_ij > 1e-10:
                # Repulsive force
                force = d / (dist_ij ** 2)
                grad[i] += force
                grad[j] -= force
        
        # Boundary constraint: push inward if close to boundary
        r_i = np.sqrt(np.sum(pts[i] ** 2))
        if r_i > 0.9:
            grad[i] += pts[i] / r_i * 10.0
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial solution: hexagonal lattice
    pts = _hex_lattice_init(N)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Gradient-based optimization with soft constraints
    t = 0.05  # softness parameter
    lr = 0.01
    n_iter = 5000
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Compute gradient using simple repulsive model
        grad = np.zeros_like(pts)
        n = len(pts)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist_ij = np.sqrt(np.sum(d ** 2))
                if dist_ij > 1e-10:
                    # Stronger repulsion when close
                    force = d / (dist_ij ** 2) * 0.5
                    grad[i] += force
                    grad[j] -= force
            
            # Boundary: push inward
            r_i = np.sqrt(np.sum(pts[i] ** 2))
            if r_i > 0.85:
                grad[i] += pts[i] / max(r_i, 1e-10) * 5.0
        
        # Gradient descent (minimize -score, so we add gradient)
        pts = pts + lr * grad
        
        # Project back inside unit circle
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        mask = radii > 0.99
        if np.any(mask):
            pts[mask] = pts[mask] * (0.99 / radii[mask])
        
        # Evaluate
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95
            t *= 0.9
    
    # Final polish: try to improve with smaller steps
    if time.time() < deadline - 5.0:
        lr = 0.001
        for it in range(2000):
            if time.time() > deadline:
                break
            
            grad = np.zeros_like(best_pts)
            n = len(best_pts)
            for i in range(n):
                for j in range(i + 1, n):
                    d = best_pts[i] - best_pts[j]
                    dist_ij = np.sqrt(np.sum(d ** 2))
                    if dist_ij > 1e-10:
                        force = d / (dist_ij ** 2) * 0.5
                        grad[i] += force
                        grad[j] -= force
                
                r_i = np.sqrt(np.sum(best_pts[i] ** 2))
                if r_i > 0.85:
                    grad[i] += best_pts[i] / max(r_i, 1e-10) * 5.0
            
            new_pts = best_pts + lr * grad
            radii = np.sqrt(np.sum(new_pts ** 2, axis=1))
            mask = radii > 0.99
            if np.any(mask):
                new_pts[mask] = new_pts[mask] * (0.99 / radii[mask])
            
            s = _score(new_pts)
            if s > best_score:
                best_score = s
                best_pts = new_pts.copy()
            else:
                lr *= 0.9
    
    return best_pts
# EVOLVE-BLOCK-END
