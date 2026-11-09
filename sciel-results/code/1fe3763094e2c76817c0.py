import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 560
    n = 14
    
    def get_grad(x):
        # x is (n, 3)
        # Returns gradient of -log(dmin) + log(dmax)
        # d = pdist(x)
        # We need indices of min and max
        # pdist doesn't return indices easily. 
        # We can compute the distance matrix to get indices.
        # N=14, so 14x14 matrix is small.
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(x[i] - x[j])
                dists[i, j] = d
                dists[j, i] = d
        
        d_min = np.min(dists[dists > 0])
        d_max = np.max(dists)
        
        if d_min < 1e-12 or d_max < 1e-12:
            return np.zeros_like(x), 0.0
            
        grad = np.zeros_like(x)
        
        # Gradient of log(d_max)
        # Sum over pairs (i,j) where dists[i,j] == d_max
        mask_max = np.isclose(dists, d_max)
        # Exclude diagonal
        np.fill_diagonal(mask_max, False)
        
        # For each i, sum contributions from j where mask_max[i,j] is True
        # Contribution to x_i from pair (i,j) is (x_i - x_j) / d_max
        # Contribution to x_j from pair (i,j) is (x_j - x_i) / d_max
        
        # We can vectorize this a bit or just loop since N=14
        for i in range(n):
            for j in range(n):
                if mask_max[i, j]:
                    # x_i gets (x_i - x_j)/d_max
                    # x_j gets (x_j - x_i)/d_max
                    # We are iterating i, so we update grad[i]
                    # But we also need to update grad[j]. 
                    # To avoid double counting or missing, let's just accumulate.
                    # Actually, the loop over i,j will cover each pair twice (i,j) and (j,i).
                    # So we should divide by 2? Or just sum all.
                    # Let's sum all contributions.
                    grad[i] += (x[i] - x[j]) / d_max
                    # The contribution to x[j] will be added when i=j? No, when outer loop is j.
                    # So this loop adds (x_i-x_j)/d_max to grad[i] for every j such that (i,j) is max.
                    # And when outer loop is j, it adds (x_j-x_i)/d_max to grad[j].
                    # This is correct.
        
        # Gradient of -log(d_min)
        mask_min = np.isclose(dists, d_min)
        np.fill_diagonal(mask_min, False)
        
        for i in range(n):
            for j in range(n):
                if mask_min[i, j]:
                    # We want to minimize -log(d_min) => maximize log(d_min)
                    # Grad of log(d_min) wrt x_i is (x_i - x_j) / d_min
                    # So grad of -log(d_min) is -(x_i - x_j) / d_min
                    grad[i] -= (x[i] - x[j]) / d_min
                    
        return grad, -(np.log(d_min) - np.log(d_max)) # This is the value to minimize: -log(dmin) + log(dmax) = log(dmax/dmin)

    def objective(x):
        pts = x.reshape(n, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1)
        mask = norms < 1e-12
        if np.any(mask):
            norms[mask] = 1.0
            pts[mask] = 0.0
            pts[mask, 0] = 1.0
        pts = pts / norms[:, np.newaxis]
        
        grad, val = get_grad(pts)
        return val

    def objective_with_norm(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1)
        mask = norms < 1e-12
        if np.any(mask):
            norms[mask] = 1.0
            pts[mask] = 0.0
            pts[mask, 0] = 1.0
        pts = pts / norms[:, np.newaxis]
        
        d = pdist(pts)
        d_min = float(d.min())
        d_max = float(d.max())
        if d_max < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return -np.log(ratio) # Minimize -log(ratio) = log(1/ratio) = log(dmax/dmin)

    # Initial Candidates
    candidates = []
    
    # 1. Icosahedron + Poles
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, 1, t], [1, 1, t], [-1, -1, t], [1, -1, t],
        [1, t, -1], [1, t, 1], [-1, t, -1], [-1, t, 1],
        [t, -1, -1], [t, 1, -1], [t, -1, 1], [t, 1, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    base_ico_poles = np.vstack([ico, poles])
    candidates.append(base_ico_poles.flatten())
    
    # Perturbed Icosahedron + Poles
    np.random.seed(42)
    for _ in range(10):
        pts = base_ico_poles + np.random.randn(14, 3) * 0.1
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 2. Fibonacci Sphere
    k = np.arange(n)
    phi = np.arccos(1 - 2 * (k + 0.5) / n)
    golden = np.pi * (1 + np.sqrt(5))
    fib = np.column_stack([
        np.sin(phi) * np.cos(golden * k),
        np.sin(phi) * np.sin(golden * k),
        np.cos(phi)
    ])
    candidates.append(fib.flatten())
    
    # 3. Random
    for _ in range(20):
        pts = np.random.randn(n, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())

    best_val = np.inf
    best_pts = None

    # Optimize each candidate using custom gradient descent
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 60:
            break
            
        x = cand.copy()
        # Gradient Descent on Sphere
        alpha = 0.01
        for step in range(1000):
            pts = x.reshape(n, 3)
            norms = np.linalg.norm(pts, axis=1)
            mask = norms < 1e-12
            if np.any(mask):
                norms[mask] = 1.0
                pts[mask] = 0.0
                pts[mask, 0] = 1.0
            pts = pts / norms[:, np.newaxis]
            
            grad, val = get_grad(pts)
            
            if val < best_val:
                best_val = val
                best_pts = pts.copy()
                
            # Check convergence
            if step % 50 == 0 and step > 0:
                # Optional: check gradient norm
                pass
                
            # Project gradient to tangent space
            # g_tan = g - (g . x) x
            g_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            
            # Update
            pts = pts - alpha * g_tan
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            x = pts.flatten()
            
            # Decay alpha
            if step % 100 == 99:
                alpha *= 0.5
                if alpha < 1e-6:
                    break

    # Polish best result with scipy
    if best_pts is not None and time.time() < deadline - 30:
        x0 = best_pts.flatten()
        try:
            res = minimize(objective_with_norm, x0, method='Powell',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_pts = res.x.reshape(n, 3)
        except Exception:
            pass

    if best_pts is None:
        best_pts = candidates[0].reshape(n, 3)

    # Final normalization
    pts = best_pts.reshape(n, 3)
    norms = np.linalg.norm(pts, axis=1)
    mask = norms < 1e-12
    if np.any(mask):
        norms[mask] = 1.0
        pts[mask] = 0.0
        pts[mask, 0] = 1.0
    pts = pts / norms[:, np.newaxis]
    
    return pts
