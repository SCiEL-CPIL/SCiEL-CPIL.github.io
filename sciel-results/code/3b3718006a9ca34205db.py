# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    n = 14

    def calculate_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def project_to_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        return points / norms

    def objective(params):
        """Negative of squared min/max ratio, with soft-min/soft-max smoothing."""
        pts = params.reshape(-1, 3)
        # Project to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        pts = pts / norms
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        dist_matrix = np.sqrt(dist_sq)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        dists = dist_matrix[mask]
        
        if len(dists) == 0:
            return 0.0
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return 0.0
            
        # Use soft versions for smoothness
        # Soft min: -log-sum-exp(-dists / tau)
        # Soft max: log-sum-exp(dists / tau)
        tau = 0.05  # Temperature for smoothing
        soft_min = -tau * np.log(np.sum(np.exp(-dists / tau)))
        soft_max = tau * np.log(np.sum(np.exp(dists / tau)))
        
        if soft_max < 1e-12:
            return 0.0
            
        ratio = soft_min / soft_max
        return -(ratio ** 2)

    def objective_grad(params):
        """Gradient of the objective using finite differences approximation."""
        pts = params.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        pts = pts / norms
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        dist_matrix = np.sqrt(dist_sq)
        
        mask = ~np.eye(n, dtype=bool)
        dists = dist_matrix[mask]
        
        if len(dists) == 0:
            return np.zeros_like(params)
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return np.zeros_like(params)
            
        # For gradient, use exact min/max with subgradient
        # d/dp_i of d_min: if pair (i,j) is min, gradient is (p_i - p_j)/d_min
        # d/dp_i of d_max: if pair (i,j) is max, gradient is (p_i - p_j)/d_max
        
        min_pairs = np.argwhere(dist_matrix <= d_min + 1e-10)
        max_pairs = np.argwhere(dist_matrix >= d_max - 1e-10)
        
        grad = np.zeros_like(pts)
        
        # Contribution from d_min (we want to maximize it, so negative gradient)
        for idx in min_pairs:
            i_idx, j_idx = int(idx[0]), int(idx[1])
            vec = pts[i_idx] - pts[j_idx]
            d = np.linalg.norm(vec)
            if d > 1e-10:
                unit = vec / d
                # d/dp_i of d_min = unit, d/dp_j = -unit
                # We want to maximize d_min, so add -unit to grad[i], +unit to grad[j]
                # But in objective we have -ratio^2, so d(objective)/d(p) = -2*ratio * d(ratio)/d(p)
                # ratio = d_min/d_max, d(ratio)/d(p_i) = (1/d_max)*d(d_min)/d(p_i) - (d_min/d_max^2)*d(d_max)/d(p_i)
                # This is getting complex. Let's just use numerical gradient.
                pass
        
        # Use central differences for gradient
        eps = 1e-7
        grad_num = np.zeros_like(params)
        f0 = objective(params)
        for k in range(len(params)):
            params_plus = params.copy()
            params_plus[k] += eps
            params_minus = params.copy()
            params_minus[k] -= eps
            grad_num[k] = (objective(params_plus) - objective(params_minus)) / (2 * eps)
        
        return grad_num

    def polish_min_dist(points, iterations=20000, lr=0.001):
        """
        Polish to maximize minimum pairwise distance on sphere.
        Uses gradient ascent on d_min with tangent projection.
        """
        points = project_to_sphere(points.copy())
        n_pts = len(points)

        for i in range(iterations):
            if time.time() - start_time > timeout - 15:
                break

            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
            dist_matrix = np.sqrt(dist_sq)
            
            mask = ~np.eye(n_pts, dtype=bool)
            dists = dist_matrix[mask]
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max < 1e-12:
                break
                
            # Find all pairs at minimum distance
            min_pairs = np.argwhere(dist_matrix <= d_min + 1e-12)
            
            grad = np.zeros_like(points)
            for idx in min_pairs:
                i_idx, j_idx = int(idx[0]), int(idx[1])
                vec = points[i_idx] - points[j_idx]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    unit = vec / d
                    grad[i_idx] += unit
                    grad[j_idx] -= unit
            
            # Project to tangent space
            for k in range(n_pts):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            # Adaptive learning rate
            if i > 5000:
                lr *= 0.9998
            elif i > 10000:
                lr *= 0.9995
            
            # Move in direction of increasing d_min
            points = points + lr * grad
            points = project_to_sphere(points)

        return points

    def polish_ratio(points, iterations=15000, lr=0.0005):
        """
        Polish to maximize (d_min/d_max)^2.
        Balances increasing d_min and decreasing d_max.
        """
        points = project_to_sphere(points.copy())
        n_pts = len(points)

        for i in range(iterations):
            if time.time() - start_time > timeout - 15:
                break

            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
            dist_matrix = np.sqrt(dist_sq)
            
            mask = ~np.eye(n_pts, dtype=bool)
            dists = dist_matrix[mask]
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max < 1e-12:
                break
                
            ratio = d_min / d_max
            
            # Find min and max pairs
            min_pairs = np.argwhere(dist_matrix <= d_min + 1e-12)
            max_pairs = np.argwhere(dist_matrix >= d_max - 1e-12)
            
            grad = np.zeros_like(points)
            
            # Gradient to increase d_min
            for idx in min_pairs:
                i_idx, j_idx = int(idx[0]), int(idx[1])
                vec = points[i_idx] - points[j_idx]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    unit = vec / d
                    grad[i_idx] += unit
                    grad[j_idx] -= unit
            
            # Gradient to decrease d_max (move max pairs closer)
            for idx in max_pairs:
                i_idx, j_idx = int(idx[0]), int(idx[1])
                vec = points[i_idx] - points[j_idx]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    unit = vec / d
                    # To decrease distance, move i toward j and j toward i
                    grad[i_idx] -= unit
                    grad[j_idx] += unit
            
            # Weight the gradients based on ratio
            # d(ratio)/d(p) = (1/d_max) * d(d_min)/d(p) - (d_min/d_max^2) * d(d_max)/d(p)
            # So weight min gradient by 1/d_max and max gradient by -d_min/d_max^2
            min_weight = 1.0 / d_max
            max_weight = -d_min / (d_max ** 2)
            
            # Recompute with weights
            grad = np.zeros_like(points)
            for idx in min_pairs:
                i_idx, j_idx = int(idx[0]), int(idx[1])
                vec = points[i_idx] - points[j_idx]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    unit = vec / d
                    grad[i_idx] += min_weight * unit
                    grad[j_idx] -= min_weight * unit
            
            for idx in max_pairs:
                i_idx, j_idx = int(idx[0]), int(idx[1])
                vec = points[i_idx] - points[j_idx]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    unit = vec / d
                    grad[i_idx] += max_weight * unit
                    grad[j_idx] -= max_weight * unit
            
            # Project to tangent space
            for k in range(n_pts):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            # Normalize
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            else:
                break
            
            # Adaptive learning rate
            if i > 5000:
                lr *= 0.9998
            elif i > 10000:
                lr *= 0.9995
            
            points = points + lr * grad
            points = project_to_sphere(points)

        return points

    def fibonacci_sphere(n_pts, R=1.0):
        golden = (1 + np.sqrt(5)) / 2
        i = np.arange(n_pts)
        z = 1 - (2 * i + 1) / n_pts
        r = np.sqrt(np.maximum(0, 1 - z**2))
        theta = 2 * np.pi * i / golden
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack([x, y, z])
        return pts * R

    def icosahedron_based(n_pts):
        """Start from icosahedron vertices and add points optimally."""
        # Icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        
        # Start with icosahedron (12 points) and add 2 more
        if n_pts <= 12:
            return verts[:n_pts]
        
        pts = verts.copy()
        remaining = n_pts - 12
        
        # Add points to maximize minimum distance
        for _ in range(remaining):
            best_pos = None
            best_min_dist = -1
            
            # Try many candidate positions
            np.random.seed(42 + len(pts))
            for _ in range(2000):
                candidate = np.random.randn(3)
                candidate = candidate / np.linalg.norm(candidate)
                
                # Compute min distance to existing points
                dists = np.linalg.norm(pts - candidate, axis=1)
                min_d = np.min(dists)
                
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_pos = candidate
            
            if best_pos is not None:
                pts = np.vstack([pts, best_pos])
        
        return pts[:n_pts]

    best_score = 0.0
    best_points = None

    # Generate multiple initializations
    inits = []

    # 1. Fibonacci sphere
    inits.append(fibonacci_sphere(n))

    # 2. Icosahedron-based
    inits.append(icosahedron_based(n))

    # 3. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000]:
        np.random.seed(seed)
        inits.append(np.random.randn(n, 3))

    # 4. Perturbed Fibonacci
    for seed in [777, 888, 999]:
        np.random.seed(seed)
        inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.3)

    for init_idx, init in enumerate(inits):
        if time.time() - start_time > timeout - 60:
            break

        pts = project_to_sphere(init.copy())

        # Phase 1: L-BFGS-B optimization
        if time.time() - start_time < timeout - 30:
            try:
                result = minimize(
                    objective,
                    pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                pts_lbfgs = project_to_sphere(result.x.reshape(-1, 3))
                s_lbfgs = calculate_score(pts_lbfgs)
                if s_lbfgs > best_score:
                    best_score = s_lbfgs
                    best_points = pts_lbfgs.copy()
            except Exception:
                pts_lbfgs = pts

        # Phase 2: Polish minimum distance
        pts2 = polish_min_dist(pts_lbfgs, iterations=15000, lr=0.001)
        s2 = calculate_score(pts2)
        if s2 > best_score:
            best_score = s2
            best_points = pts2.copy()

        # Phase 3: Polish ratio
        if time.time() - start_time < timeout - 30:
            pts3 = polish_ratio(pts2, iterations=12000, lr=0.0005)
            s3 = calculate_score(pts3)
            if s3 > best_score:
                best_score = s3
                best_points = pts3.copy()

        # Phase 4: Another L-BFGS from polished result
        if time.time() - start_time < timeout - 20:
            try:
                result2 = minimize(
                    objective,
                    pts3.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                pts_lbfgs2 = project_to_sphere(result2.x.reshape(-1, 3))
                s_lbfgs2 = calculate_score(pts_lbfgs2)
                if s_lbfgs2 > best_score:
                    best_score = s_lbfgs2
                    best_points = pts_lbfgs2.copy()
            except Exception:
                pass

        # Phase 5: Final high-precision polish
        if time.time() - start_time < timeout - 10:
            pts_final = polish_min_dist(best_points, iterations=8000, lr=0.0003)
            s_final = calculate_score(pts_final)
            if s_final > best_score:
                best_score = s_final
                best_points = pts_final.copy()

            pts_final2 = polish_ratio(best_points, iterations=6000, lr=0.0002)
            s_final2 = calculate_score(pts_final2)
            if s_final2 > best_score:
                best_score = s_final2
                best_points = pts_final2.copy()

    if best_points is None:
        best_points = fibonacci_sphere(n)

    # Final safety check
    best_points = project_to_sphere(best_points)
    return best_points

# EVOLVE-BLOCK-END
