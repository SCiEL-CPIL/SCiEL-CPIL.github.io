import numpy as np
from math import fsum, log

N = 150

def compute_energy(pts):
    """Compute log(1/d) energy using fsum for precision."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(log(1.0/d))
    return fsum(terms)

def compute_energy_fast(pts):
    """Vectorized energy computation."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Mask diagonal
    np.fill_diagonal(dists, 1.0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0/d))

def compute_energy_fsum(pts):
    """Energy with fsum for precision."""
    n = len(pts)
    iu = np.triu_indices(n, k=1)
    diff = pts[iu[0]] - pts[iu[1]]
    d = np.sqrt(np.sum(diff**2, axis=1))
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0/d)
    return fsum(terms.tolist())

def random_rotation_3d(angle):
    """Generate a random 3D rotation matrix for a given angle."""
    # Random axis
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    x, y, z = axis
    R = np.array([
        [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s,   c + y*y*(1-c),    y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s,   z*y*(1-c) + x*s,  c + z*z*(1-c)]
    ])
    return R

def construct():
    np.random.seed(42)
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Simulated annealing
    T0 = 0.05  # Initial temperature
    Tf = 0.0001  # Final temperature
    n_iter = 200000
    alpha = (Tf/T0)**(1.0/n_iter)
    T = T0
    
    current_energy = compute_energy_fsum(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    for it in range(n_iter):
        T = max(T * alpha, Tf)
        
        # Pick a random point to move
        k = np.random.randint(N)
        
        # Small random rotation
        angle = np.random.uniform(0.01, 0.2)
        R = random_rotation_3d(angle)
        new_pt = R @ pts[k]
        new_pt /= np.linalg.norm(new_pt)  # Ensure on sphere
        
        # Compute energy change
        old_pt = pts[k]
        # Old contribution of point k
        if k > 0:
            d_old = np.sqrt(np.sum((pts[:k] - old_pt)**2, axis=1))
        else:
            d_old = np.empty(0)
        if k < N-1:
            d_old2 = np.sqrt(np.sum((pts[k+1:] - old_pt)**2, axis=1))
        else:
            d_old2 = np.empty(0)
        d_old_all = np.concatenate([d_old, d_old2])
        d_old_all = np.maximum(d_old_all, 1e-15)
        old_contrib = np.sum(np.log(1.0/d_old_all))
        
        # New contribution
        if k > 0:
            d_new = np.sqrt(np.sum((pts[:k] - new_pt)**2, axis=1))
        else:
            d_new = np.empty(0)
        if k < N-1:
            d_new2 = np.sqrt(np.sum((pts[k+1:] - new_pt)**2, axis=1))
        else:
            d_new2 = np.empty(0)
        d_new_all = np.concatenate([d_new, d_new2])
        d_new_all = np.maximum(d_new_all, 1e-15)
        new_contrib = np.sum(np.log(1.0/d_new_all))
        
        delta_E = new_contrib - old_contrib
        
        # Metropolis acceptance
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E/T):
            pts[k] = new_pt
            current_energy += delta_E
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        # Periodic recompute to avoid drift
        if it % 10000 == 0:
            current_energy = compute_energy_fsum(pts)
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
    
    # Polish with projected gradient descent
    pts = best_pts.copy()
    lr = 0.01
    for p_iter in range(500):
        # Compute gradient
        grads = np.zeros_like(pts)
        for k in range(N):
            others = np.delete(pts, k, axis=0)
            diff = pts[k][np.newaxis, :] - others
            dists_sq = np.sum(diff**2, axis=1)
            dists_sq = np.maximum(dists_sq, 1e-30)
            # grad_k = sum_{j≠k} -(x_k - x_j)/||x_k - x_j||^2
            grads[k] = -np.sum(diff / dists_sq[:, np.newaxis], axis=0)
        
        # Update
        new_pts = pts - lr * grads
        # Project to sphere
        norms = np.linalg.norm(new_pts, axis=1)
        norms = np.maximum(norms, 1e-15)
        new_pts = new_pts / norms[:, np.newaxis]
        
        # Check improvement
        new_energy = compute_energy_fsum(new_pts)
        if new_energy < compute_energy_fsum(pts):
            pts = new_pts
        else:
            lr *= 0.9
    
    # Final check
    final_energy = compute_energy_fsum(pts)
    if final_energy < best_energy:
        best_pts = pts
    
    return best_pts
