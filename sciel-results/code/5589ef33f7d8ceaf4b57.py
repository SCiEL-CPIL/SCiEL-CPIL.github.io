import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_hex_points(spacing):
        k_max = int(2.0 / spacing) + 2
        points = []
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                points.append((x, y))
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        mask = dists <= 1.0 - spacing / 2.0
        points = points[mask]
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        return points[idx]

    # Initialization
    pts = get_hex_points(0.11)
    if len(pts) < N:
        pts = get_hex_points(0.10)
    if len(pts) < N:
        # Fallback
        x = np.linspace(-0.9, 0.9, 20)
        y = np.linspace(-0.9, 0.9, 20)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        dists = np.linalg.norm(pts, axis=1)
        pts = pts[dists < 1]
        if len(pts) < N:
            return pts[:N]

    c = pts[:N].copy()
    
    # Initial r
    def get_min_dist(c):
        d = np.sqrt(((c[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        return np.min(d)
    
    def get_min_bound(c):
        return np.min(1.0 - np.linalg.norm(c, axis=1))

    r = min(get_min_dist(c) / 2.0, get_min_bound(c))
    r = max(r - 0.001, 0.01) # Start slightly below to be safe
    
    best_r = r
    best_c = c.copy()
    
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(c)
    
    while time.time() < deadline:
        # Compute Gradient
        d = np.sqrt(((c[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        
        v_pair = np.maximum(0, 2*r - d)
        safe_d = np.where(d > 1e-6, d, 1e-6)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        grad_pair = 2.0 * v_pair[:, :, np.newaxis] * (diff / safe_d[:, :, np.newaxis])
        grad = np.sum(grad_pair, axis=1)
        
        dist_c = np.linalg.norm(c, axis=1)
        v_bound = np.maximum(0, r + dist_c - 1.0)
        safe_dist_c = np.where(dist_c > 1e-6, dist_c, 1e-6)
        grad_bound = 2.0 * v_bound[:, np.newaxis] * (c / safe_dist_c[:, np.newaxis])
        grad += grad_bound
        
        E = np.sum(v_pair ** 2) + np.sum(v_bound ** 2)
        
        v = momentum * v - lr * grad
        c_new = c + v
        
        if E < 1e-5:
            min_d = np.min(d)
            min_b = np.min(1.0 - dist_c)
            max_r = min(min_d / 2.0, min_b)
            
            if max_r > r:
                delta_r = min(0.1 * (max_r - r), 1e-4)
                r += delta_r
                if r > best_r:
                    best_r = r
                    best_c = c_new.copy()
            else:
                c_new += np.random.normal(0, 0.002, c.shape)
        else:
            lr *= 0.995
            if lr < 1e-6:
                lr = 1e-6
                c_new += np.random.normal(0, 0.005, c.shape)
        
        c = c_new
        
        # Boundary constraint enforcement
        dists_new = np.linalg.norm(c, axis=1)
        out_mask = dists_new > 1.0
        if np.any(out_mask):
            c[out_mask] *= (1.0 / dists_new[out_mask])[:, np.newaxis] * 0.99

    return best_c
