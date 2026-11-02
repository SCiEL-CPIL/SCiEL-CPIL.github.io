"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    best_value = -np.inf
    best_result = None
    
    def evaluate(circles, w, h):
        """Return sum of radii if valid, else -inf."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -1e-5 or x + r > w + 1e-5 or y - r < -1e-5 or y + r > h + 1e-5:
                return -np.inf
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if d < ri + rj - 1e-5:
                    return -np.inf
        return float(np.sum(circles[:, 2]))
    
    def check_and_update(x, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        circles = np.column_stack([xs, ys, rs])
        val = evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = x.copy()
    
    def softplus(x, beta=100.0):
        """Numerically stable softplus."""
        x_beta = beta * x
        result = np.where(x_beta > 0,
                         x + np.log1p(np.exp(-x_beta)) / beta,
                         np.log1p(np.exp(x_beta)) / beta)
        return result
    
    def softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)."""
        return 1.0 / (1.0 + np.exp(-beta * x))
    
    def objective_and_grad(x, w, h, beta=100.0):
        """
        Force-directed penalty objective with analytical gradients (vectorized).
        x: [x0, y0, r0, x1, y1, r1, ...]
        """
        c = x.reshape(-1, 3)
        n = len(c)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]
        
        obj = -np.sum(rs)
        grad = np.zeros_like(x)
        
        # Boundary penalties (vectorized)
        v1 = rs - xs
        obj += np.sum(softplus(v1, beta))
        g1 = softplus_grad(v1, beta)
        grad[0::3] -= g1
        grad[2::3] += g1
        
        v2 = xs + rs - w
        obj += np.sum(softplus(v2, beta))
        g2 = softplus_grad(v2, beta)
        grad[0::3] += g2
        grad[2::3] += g2
        
        v3 = rs - ys
        obj += np.sum(softplus(v3, beta))
        g3 = softplus_grad(v3, beta)
        grad[1::3] -= g3
        grad[2::3] += g3
        
        v4 = ys + rs - h
        obj += np.sum(softplus(v4, beta))
        g4 = softplus_grad(v4, beta)
        grad[1::3] += g4
        grad[2::3] += g4
        
        v5 = -rs
        obj += np.sum(softplus(v5, beta))
        g5 = softplus_grad(v5, beta)
        grad[2::3] -= g5
        
        # Pairwise overlap penalties (vectorized)
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        
        rsum = rs[:, None] + rs[None, :]
        iu = np.triu_indices(n, 1)
        
        v = rsum[iu] - d[iu]
        obj += np.sum(softplus(v, beta))
        g = softplus_grad(v, beta)
        
        d_safe = np.maximum(d[iu], 1e-12)
        dx_iu = dx[iu]
        dy_iu = dy[iu]
        
        idx_i = iu[0]
        idx_j = iu[1]
        gx = g * dx_iu / d_safe
        gy = g * dy_iu / d_safe
        
        np.add.at(grad[0::3], idx_i, -gx)
        np.add.at(grad[0::3], idx_j, gx)
        np.add.at(grad[1::3], idx_i, -gy)
        np.add.at(grad[1::3], idx_j, gy)
        np.add.at(grad[2::3], idx_i, g)
        np.add.at(grad[2::3], idx_j, g)
        
        grad[2::3] += -1.0
        
        return obj, grad
    
    def optimize_from(circles0, w, h, max_iter=400, beta=100.0):
        """L-BFGS-B optimization with softplus penalties."""
        x0 = circles0.flatten()
        
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))
        
        try:
            res = minimize(
                lambda p: objective_and_grad(p, w, h, beta),
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res.x
        except Exception:
            return x0
    
    def polish(circles0, w, h, max_iter=500, beta=200.0):
        """High-precision polish pass."""
        return optimize_from(circles0, w, h, max_iter=max_iter, beta=beta)
    
    def hex_lattice_init(w, h):
        """Generate hexagonal lattice initialization for N=21 circles."""
        best_circles = None
        best_val = -np.inf
        
        row_configs = [
            [3, 4, 4, 4, 3, 3],
            [4, 4, 4, 4, 3, 2],
            [2, 4, 4, 4, 4, 3],
            [3, 4, 4, 4, 4, 2],
            [4, 3, 4, 4, 4, 2],
            [2, 3, 4, 4, 4, 4],
            [3, 3, 4, 4, 4, 3],
            [4, 4, 3, 4, 4, 2],
            [2, 4, 3, 4, 4, 4],
            [3, 4, 3, 4, 4, 3],
            [5, 4, 4, 4, 4],
            [4, 5, 4, 4, 4],
            [3, 5, 4, 4, 5],
            [4, 4, 5, 4, 4],
            [2, 5, 4, 5, 5],
            [5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5],
            [3, 4, 5, 4, 5],
            [5, 4, 5, 4, 3],
            [4, 4, 4, 4, 4, 1],
            [1, 4, 4, 4, 4, 4],
            [4, 4, 4, 5, 4],
            [5, 4, 4, 4, 4],
            [4, 4, 5, 4, 4],
            [3, 4, 4, 5, 5],
            [5, 5, 4, 4, 3],
            [4, 3, 4, 5, 5],
            [5, 5, 3, 4, 4],
            [2, 4, 4, 4, 3, 4],
            [4, 3, 4, 4, 4, 2],
            [3, 3, 3, 4, 4, 4],
            [4, 4, 4, 3, 3, 3],
            [6, 4, 4, 4, 3],
            [3, 4, 4, 4, 6],
            [7, 4, 4, 4, 2],
            [2, 4, 4, 4, 7],
            [5, 4, 4, 4, 4],
            [4, 4, 4, 4, 5],
        ]
        
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            
            n_rows = len(row_counts)
            max_cols = max(row_counts)
            
            est_r_h = w / (max_cols + 0.5)
            est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
            r_est = min(est_r_h, est_r_v) * 0.95
            
            if r_est < 0.01:
                continue
            
            circles = []
            for row_idx, n_cols in enumerate(row_counts):
                y = (row_idx + 0.5) * r_est * np.sqrt(3)
                x_offset = r_est if row_idx % 2 == 1 else 0.0
                row_width = n_cols * 2 * r_est
                center_x = w / 2
                for col_idx in range(n_cols):
                    x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                    circles.append([x, y, r_est])
            
            circles = np.array(circles)
            val = evaluate(circles, w, h)
            
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
        
        if best_circles is None:
            a = int(np.ceil(np.sqrt(N * w / h)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            circles = []
            for i in range(N):
                col = i % a
                row = i // a
                r = min(w / a, h / b) * 0.45
                x = (col + 0.5) * w / a
                y = (row + 0.5) * h / b
                circles.append([x, y, r])
            best_circles = np.array(circles)
        
        return best_circles
    
    def cluster_objective_and_grad(p, beta=100.0):
        """
        Rigid-cluster objective: optimize rotation angle theta, translation (tx, ty),
        and per-circle radii. The relative positions of circles are fixed from a
        hexagonal template.
        
        p = [theta, tx, ty, r0, r1, ..., r20]
        """
        theta, tx, ty = p[0], p[1], p[2]
        rs = p[3:]
        
        # Fixed relative positions (hexagonal template centered at origin)
        rel_pos = _get_hex_template()  # (N, 2)
        
        # Rotate and translate
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        xs = rel_pos[:, 0] * c - rel_pos[:, 1] * s + tx
        ys = rel_pos[:, 0] * s + rel_pos[:, 1] * c + ty
        
        obj = -np.sum(rs)
        grad = np.zeros_like(p)
        
        # Compute rectangle dimensions from aspect ratio (fixed for this call)
        w, h = _current_w, _current_h
        
        # Boundary penalties
        v1 = rs - xs
        obj += np.sum(softplus(v1, beta))
        g1 = softplus_grad(v1, beta)
        # d/dtheta: -(-rel_y)*s - rel_x*c ... let's compute properly
        # d(xs)/dtheta = -rel_x*s - rel_y*c
        # d(ys)/dtheta = rel_x*c - rel_y*s
        d_x_dtheta = -rel_pos[:, 0] * s - rel_pos[:, 1] * c
        d_y_dtheta = rel_pos[:, 0] * c - rel_pos[:, 1] * s
        
        grad[0] += np.sum(g1 * d_x_dtheta)  # d(obj)/dtheta from v1
        grad[1] += -np.sum(g1)  # d(obj)/dtx from v1
        # d(v1)/drs = 1
        grad[3:] += g1
        
        v2 = xs + rs - w
        obj += np.sum(softplus(v2, beta))
        g2 = softplus_grad(v2, beta)
        grad[0] += np.sum(g2 * d_x_dtheta)
        grad[1] += np.sum(g2)
        grad[3:] += g2
        
        v3 = rs - ys
        obj += np.sum(softplus(v3, beta))
        g3 = softplus_grad(v3, beta)
        grad[0] += np.sum(g3 * d_y_dtheta)
        grad[2] += -np.sum(g3)
        grad[3:] += g3
        
        v4 = ys + rs - h
        obj += np.sum(softplus(v4, beta))
        g4 = softplus_grad(v4, beta)
        grad[0] += np.sum(g4 * d_y_dtheta)
        grad[2] += np.sum(g4)
        grad[3:] += g4
        
        v5 = -rs
        obj += np.sum(softplus(v5, beta))
        g5 = softplus_grad(v5, beta)
        grad[3:] += -g5
        
        # Pairwise overlap penalties
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        rsum = rs[:, None] + rs[None, :]
        iu = np.triu_indices(N, 1)
        
        v = rsum[iu] - d[iu]
        obj += np.sum(softplus(v, beta))
        g = softplus_grad(v, beta)
        
        d_safe = np.maximum(d[iu], 1e-12)
        gx = g * dx[iu] / d_safe
        gy = g * dy[iu] / d_safe
        
        idx_i = iu[0]
        idx_j = iu[1]
        
        # d(xs_i)/dtheta = d_x_dtheta[i], d(xs_j)/dtheta = d_x_dtheta[j]
        # d(v)/dtheta = (d_x_dtheta[i] - d_x_dtheta[j]) * (-gx[iu]) + (d_y_dtheta[i] - d_y_dtheta[j]) * (-gy[iu])
        # Wait: v = rsum - d, so d(v)/dtheta = -d(d)/dtheta
        # d/dtheta d = (dx_i - dx_j)/d * (d_x_dtheta[i] - d_x_dtheta[j]) + (dy_i - dy_j)/d * (d_y_dtheta[i] - d_y_dtheta[j])
        # = -gx[iu] * (d_x_dtheta[i] - d_x_dtheta[j]) - gy[iu] * (d_y_dtheta[i] - d_y_dtheta[j])
        # So d(v)/dtheta = gx[iu] * (d_x_dtheta[i] - d_x_dtheta[j]) + gy[iu] * (d_y_dtheta[i] - d_y_dtheta[j])
        
        dtheta_contrib = gx * (d_x_dtheta[idx_i] - d_x_dtheta[idx_j]) + gy * (d_y_dtheta[idx_i] - d_y_dtheta[idx_j])
        grad[0] += np.sum(g * dtheta_contrib)
        
        # d(v)/dtx = -gx (since d(xs_i)/dtx = 1, d(xs_j)/dtx = 1, so d(d)/dtx = (dx_i-dx_j)/d * 0 = 0... wait)
        # Actually d/dtx d = (dx_i - dx_j)/d * (1 - 1) + ... = 0. Translation doesn't change distances!
        # So grad[1] and grad[2] get no contribution from pairwise terms.
        
        # d(v)/drs_i = 1, d(v)/drs_j = 1
        np.add.at(grad[3:], idx_i, g)
        np.add.at(grad[3:], idx_j, g)
        
        # Gradient of -sum(r)
        grad[3:] += -1.0
        
        return obj, grad
    
    _current_w = 1.0
    _current_h = 1.0
    
    def _get_hex_template():
        """Return fixed relative positions for 21 circles in hexagonal arrangement, centered at origin."""
        # Use a 4-4-4-4-3-2 or similar configuration
        # Let's use 4 rows of 5 and 1 row of 1, or better: 3-4-4-4-3-3
        row_counts = [3, 4, 4, 4, 3, 3]
        n_rows = len(row_counts)
        sqrt3 = np.sqrt(3)
        
        positions = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx - (n_rows - 1) / 2.0) * sqrt3
            x_offset = 1.0 if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2.0
            for col_idx in range(n_cols):
                x = x_offset - row_width / 2.0 + (col_idx + 0.5) * 2.0
                positions.append([x, y])
        
        positions = np.array(positions)
        # Center at origin
        positions -= positions.mean(axis=0)
        return positions
    
    def optimize_cluster(w, h, beta=100.0, max_iter=500):
        """Optimize the rigid cluster: rotation, translation, and radii."""
        global _current_w, _current_h
        _current_w = w
        _current_h = h
        
        # Initial guess: theta=0, centered in rectangle, equal radii
        r_est = min(w, h) / 8.0  # rough estimate
        p0 = np.zeros(3 + N)
        p0[0] = 0.0  # theta
        p0[1] = w / 2.0  # tx
        p0[2] = h / 2.0  # ty
        p0[3:] = r_est
        
        bounds = [
            (-np.pi, np.pi),  # theta
            (0, w),           # tx
            (0, h),           # ty
        ]
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((1e-6, r_max))
        
        try:
            res = minimize(
                lambda p: cluster_objective_and_grad(p, beta),
                p0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res.x
        except Exception:
            return p0
    
    def cluster_to_circles(p, w, h):
        """Convert cluster parameters to circle positions."""
        theta, tx, ty = p[0], p[1], p[2]
        rs = p[3:]
        rel_pos = _get_hex_template()
        c = np.cos(theta)
        s = np.sin(theta)
        xs = rel_pos[:, 0] * c - rel_pos[:, 1] * s + tx
        ys = rel_pos[:, 0] * s + rel_pos[:, 1] * c + ty
        circles = np.column_stack([xs, ys, rs])
        return circles
    
    # Try multiple aspect ratios
    aspects = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7]
    
    for aspect in aspects:
        if time.time() > deadline - 5:
            break
        
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Standard approach: hex lattice init + gradient optimization
        circles0 = hex_lattice_init(w, h)
        check_and_update(circles0.flatten(), w, h)
        
        if time.time() < deadline - 10:
            x1 = optimize_from(circles0, w, h, max_iter=300, beta=50.0)
            check_and_update(x1, w, h)
        
        if time.time() < deadline - 8:
            x2 = optimize_from(x1, w, h, max_iter=300, beta=100.0)
            check_and_update(x2, w, h)
        
        if time.time() < deadline - 5:
            x3 = polish(x2, w, h, max_iter=400, beta=200.0)
            check_and_update(x3, w, h)
        
        # Cluster-based optimization
        if time.time() < deadline - 3:
            pc = optimize_cluster(w, h, beta=100.0, max_iter=400)
            circles_c = cluster_to_circles(pc, w, h)
            # Clamp to bounds
            for i in range(N):
                r = max(circles_c[i, 2], 1e-6)
                circles_c[i, 0] = np.clip(circles_c[i, 0], r, w - r)
                circles_c[i, 1] = np.clip(circles_c[i, 1], r, h - r)
            check_and_update(circles_c.flatten(), w, h)
            
            # Polish the cluster result with standard gradient method
            if time.time() < deadline - 2:
                x4 = polish(circles_c, w, h, max_iter=300, beta=200.0)
                check_and_update(x4, w, h)
    
    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.099
        best_result = np.zeros(3 * N)
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[3 * idx] = (i + 0.5) * w / a
                best_result[3 * idx + 1] = (j + 0.5) * h / b
                best_result[3 * idx + 2] = r
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3 * i:3 * i + 3]
    
    return result
# EVOLVE-BLOCK-END
