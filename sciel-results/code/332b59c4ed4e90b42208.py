import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 560
    n = 14

    def normalize_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        mask = norms < 1e-12
        if np.any(mask):
            norms[mask] = 1.0
            pts[mask, 0] = 1.0
            pts[mask, 1:] = 0.0
        return pts / norms

    def ratio(pts):
        d = pdist(pts)
        d_min = float(d.min())
        d_max = float(d.max())
        if d_max <= 1e-15:
            return 0.0
        return d_min / d_max

    def optimize_repulsion(pts, steps=2000, alpha_init=0.1, alpha_decay=0.999):
        pts = pts.copy()
        alpha = alpha_init
        for i in range(steps):
            if time.time() > deadline - 50:
                break
            # Calculate distances
            d = pdist(pts)
            d_mat = squareform(d)
            # Mask diagonal
            np.fill_diagonal(d_mat, 1.0)
            
            # Force: repulsion proportional to 1/d^2 (or 1/d^p)
            # Using 1/d^2 is standard Coulomb
            inv_d2 = 1.0 / (d_mat ** 2)
            np.fill_diagonal(inv_d2, 0.0)
            
            # Force vector for each point i: sum_j (p_i - p_j) / d_ij^2
            # p_i - p_j is (n, 3, 3) broadcast
            # Actually, easier to loop or use matrix ops
            # F_i = sum_j (p_i - p_j) * inv_d2[i,j]
            # (p_i - p_j) has shape (n, n, 3)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
            
            # Project forces onto tangent plane
            # Tangent vector v is orthogonal to p_i
            # proj(v) = v - (v . p_i) p_i
            dot = np.sum(forces * pts, axis=1, keepdims=True)
            forces = forces - dot * pts
            
            # Update
            pts = pts + alpha * forces
            pts = normalize_sphere(pts)
            
            alpha *= alpha_decay
            
        return pts

    def squareform(c):
        # Simple squareform for pdist output
        n = int((2 * len(c)) ** 0.5)
        if n * (n - 1) // 2 != len(c):
            raise ValueError("Invalid distance matrix size")
        d = np.zeros((n, n))
        iu = np.triu_indices(n, 1)
        d[iu] = c
        d = d + d.T
        return d

    # Initial candidates
    k = np.arange(n)
    phi = np.arccos(1 - 2 * (k + 0.5) / n)
    golden = np.pi * (1 + np.sqrt(5))
    fib = np.column_stack([
        np.sin(phi) * np.cos(golden * k),
        np.sin(phi) * np.sin(golden * k),
        np.cos(phi)
    ])
    
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, 1, t], [1, 1, t], [-1, -1, t], [1, -1, t],
        [1, t, -1], [1, t, 1], [-1, t, -1], [-1, t, 1],
        [t, -1, -1], [t, 1, -1], [t, -1, 1], [t, 1, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    ico14 = np.vstack([ico, [[0, 0, 1.0], [0, 0, -1.0]]])

    candidates = [fib, ico14]
    np.random.seed(42)
    for _ in range(10):
        v = np.random.randn(n, 3)
        v = normalize_sphere(v)
        candidates.append(v)

    best_pts = None
    best_r = -1.0

    for cand in candidates:
        if time.time() > deadline - 100:
            break
        try:
            opt_pts = optimize_repulsion(cand, steps=5000, alpha_init=0.05, alpha_decay=0.9995)
            r = ratio(opt_pts)
            if r > best_r:
                best_r = r
                best_pts = opt_pts
        except Exception:
            pass

    if best_pts is None:
        best_pts = fib

    return best_pts
