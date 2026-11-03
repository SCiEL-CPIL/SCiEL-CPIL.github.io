import numpy as np
import time
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin for 150s limit
    
    def elapsed():
        return time.time() - start_time
    
    # Precompute pairwise indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)
    
    def check_valid(x, y, r, W, H, tol=1e-8):
        if np.any(x - r < -tol) or np.any(x + r > W + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > H + tol):
            return False
        if np.any(r <= tol):
            return False
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        req = r[i_idx] + r[j_idx]
        if np.any(dist < req - tol):
            return False
        return True
    
    def hex_init(W, H, n_circles):
        """Initialize with hexagonal packing pattern."""
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)
        
        # Estimate radius for hex packing
        # Area per circle in hex packing ~ (2*sqrt(3)/3) * r^2
        # Total area = W * H
        # n * (2*sqrt(3)/3) * r^2 ≈ W * H
        r_est = np.sqrt(W * H / (n_circles * 2 * np.sqrt(3) / 3))
        r_est = min(r_est * 0.85, W / 4, H / 4)  # Add margin
        
        # Generate hex lattice points
        points = []
        dx_hex = 2 * r_est
        dy_hex = np.sqrt(3) * r_est
        
        row = 0
        while len(points) < n_circles:
            offset = 0 if row % 2 == 0 else r_est
            col = 0
            while len(points) < n_circles:
                px = r_est + offset + col * dx_hex
                py = r_est + row * dy_hex
                if px + r_est <= W and py + r_est <= H:
                    points.append([px, py])
                    if len(points) >= n_circles:
                        break
                col += 1
            row += 1
            # Stop if we've gone too far
            if r_est + row * dy_hex > H:
                break
        
        # If we don't have enough, fill remaining with grid
        if len(points) < n_circles:
            used = set()
            for i in range(n_circles):
                if i >= len(points):
                    # Place in remaining grid positions
                    gx = (i % 5 + 0.5) * (W / 5)
                    gy = (i // 5 + 0.5) * (H / 5)
                    points.append([gx, gy])
        
        for i in range(n_circles):
            x[i] = points[i][0]
            y[i] = points[i][1]
            r[i] = r_est
        
        return x, y, r
    
    def repair(x, y, r, W, H, max_iters=20):
        """Iteratively repair overlaps by moving centers and shrinking radii."""
        for _ in range(max_iters):
            # Fix boundary
            r = np.minimum(r, np.minimum(np.minimum(x, W - x), np.minimum(y, H - y)))
            r = np.maximum(r, 1e-8)
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
            # Check overlaps
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            req = r[i_idx] + r[j_idx]
            overlap = np.maximum(0, req - dist)
            
            if np.all(overlap < 1e-8):
                break
            
            # Move apart overlapping pairs
            for idx in range(n_pairs):
                if overlap[idx] > 1e-8:
                    i, j = i_idx[idx], j_idx[idx]
                    d = dist[idx]
                    if d < 1e-10:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = overlap[idx] / 2 + 1e-6
                        x[i] -= push * np.cos(angle)
                        y[i] -= push * np.sin(angle)
                        x[j] += push * np.cos(angle)
                        y[j] += push * np.sin(angle)
                    else:
                        ux, uy = dx[idx] / d, dy[idx] / d
                        push = overlap[idx] / 2 + 1e-6
                        x[i] -= push * ux
                        y[i] -= push * uy
                        x[j] += push * ux
                        y[j] += push * uy
            
            # Re-clamp
            r = np.minimum(r, np.minimum(np.minimum(x, W - x), np.minimum(y, H - y)))
            r = np.maximum(r, 1e-8)
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
        
        return x, y, r
    
    def sa_optimize(x0, y0, r0, W, H, max_iter=30000, T0=0.05):
        """Simulated annealing to maximize sum of radii."""
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        current_sum = np.sum(r)
        best_sum = current_sum
        best_x = x.copy()
        best_y = y.copy()
        best_r = r.copy()
        
        T = T0
        T_min = 1e-7
        cooling = 0.9995
        
        # Precompute for speed
        for it in range(max_iter):
            if elapsed() > deadline:
                break
            
            # Pick random circle
            i = np.random.randint(N)
            
            # Perturb
            scale = T * 2.0
            dx = np.random.normal(0, scale * 0.3)
            dy = np.random.normal(0, scale * 0.3)
            dr = np.random.normal(0, scale * 0.05)
            
            x_new = x[i] + dx
            y_new = y[i] + dy
            r_new = max(r[i] + dr, 1e-8)
            
            # Tentatively update
            old_x, old_y, old_r = x[i], y[i], r[i]
            x[i] = x_new
            y[i] = y_new
            r[i] = r_new
            
            # Repair
            x, y, r = repair(x, y, r, W, H, max_iters=8)
            
            new_sum = np.sum(r)
            
            # Accept/reject
            delta = new_sum - current_sum
            if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-10)):
                current_sum = new_sum
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()
            else:
                x[i] = old_x
                y[i] = old_y
                r[i] = old_r
                # Re-repair to restore state
                x, y, r = repair(x, y, r, W, H, max_iters=4)
                current_sum = np.sum(r)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_sum, best_x, best_y, best_r
    
    def gradient_polish(x, y, r, W, H, max_iter=5000):
        """Gradient-based refinement using force-directed relaxation."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        best_sum = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        lr = 0.01
        for it in range(max_iter):
            if elapsed() > deadline:
                break
            
            # Compute forces from overlaps and boundary pressure
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Overlap forces
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            req = r[i_idx] + r[j_idx]
            overlap = np.maximum(0, req - dist)
            
            valid = dist > 1e-10
            for idx in range(n_pairs):
                if not valid[idx]:
                    continue
                if overlap[idx] > 0:
                    ux = dx[idx] / dist[idx]
                    uy = dy[idx] / dist[idx]
                    f = overlap[idx] * 10.0  # Spring constant
                    fx[i_idx[idx]] -= f * ux
                    fy[i_idx[idx]] -= f * uy
                    fx[j_idx[idx]] += f * ux
                    fy[j_idx[idx]] += f * uy
            
            # Boundary forces (push inward)
            margin_x = np.minimum(x, W - x)
            margin_y = np.minimum(y, H - y)
            for i in range(N):
                if x[i] - r[i] < 0.01:
                    fx[i] += (0.01 - (x[i] - r[i])) * 50.0
                if x[i] + r[i] > W - 0.01:
                    fx[i] -= (x[i] + r[i] - (W - 0.01)) * 50.0
                if y[i] - r[i] < 0.01:
                    fy[i] += (0.01 - (y[i] - r[i])) * 50.0
                if y[i] + r[i] > H - 0.01:
                    fy[i] -= (y[i] + r[i] - (H - 0.01)) * 50.0
            
            # Update positions
            x += lr * fx
            y += lr * fy
            
            # Clamp and shrink if needed
            r = np.minimum(r, np.minimum(np.minimum(x, W - x), np.minimum(y, H - y)))
            r = np.maximum(r, 1e-8)
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
            current_sum = np.sum(r)
            if current_sum > best_sum:
                best_sum = current_sum
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            
            lr *= 0.999
        
        return best_sum, best_x, best_y, best_r
    
    def del_delaunay_polish(x, y, r, W, H, max_iter=2000):
        """Use Delaunay triangulation to identify critical pairs and polish."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        best_sum = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for it in range(max_iter):
            if elapsed() > deadline:
                break
            
            # Get Delaunay neighbors for constraint reduction
            try:
                tri = Delaunay(np.column_stack([x, y]))
                neighbors = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i+1, 3):
                            neighbors.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
            except:
                neighbors = set(zip(i_idx, j_idx))
            
            # Force-directed on critical pairs only
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            for (i, j) in neighbors:
                dx_ij = x[j] - x[i]
                dy_ij = y[j] - y[i]
                dist_ij = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                req_ij = r[i] + r[j]
                
                if dist_ij < req_ij and dist_ij > 1e-10:
                    ux = dx_ij / dist_ij
                    uy = dy_ij / dist_ij
                    f = (req_ij - dist_ij) * 5.0
                    fx[i] -= f * ux
                    fy[i] -= f * uy
                    fx[j] += f * ux
                    fy[j] += f * uy
            
            # Try to increase radii slightly
            for i in range(N):
                # Check if we can grow
                can_grow = True
                max_grow = min(x[i], W - x[i], y[i], H - y[i]) - r[i]
                for (a, b) in neighbors:
                    if i in (a, b):
                        other = b if i == a else a
                        dx_ij = x[other] - x[i]
                        dy_ij = y[other] - y[i]
                        dist_ij = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                        if dist_ij < r[i] + r[other] + 1e-6:
                            can_grow = False
                            break
                        else:
                            # Max grow limited by this neighbor
                            max_grow = min(max_grow, dist_ij - r[i] - r[other])
                
                if can_grow and max_grow > 1e-6:
                    r[i] += min(max_grow, 0.001)
            
            # Position update
            lr = 0.005
            x += lr * fx
            y += lr * fy
            
            # Clamp
            r = np.minimum(r, np.minimum(np.minimum(x, W - x), np.minimum(y, H - y)))
            r = np.maximum(r, 1e-8)
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
            current_sum = np.sum(r)
            if current_sum > best_sum:
                best_sum = current_sum
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        return best_sum, best_x, best_y, best_r
    
    # Main: try multiple aspect ratios and initializations
    best_sum = 0.0
    best_solution = None
    
    np.random.seed(42)
    
    # Aspect ratios to try
    W_values = np.linspace(0.8, 2.0, 13)
    
    for W in W_values:
        if elapsed() > deadline:
            break
        H = 2.0 - W
        if H < 0.2:
            continue
        
        # Multiple initializations
        inits = []
        
        # Hex init
        x_h, y_h, r_h = hex_init(W, H, N)
        inits.append((x_h, y_h, r_h))
        
        # Grid init
        x_g = np.zeros(N)
        y_g = np.zeros(N)
        r_g = np.zeros(N)
        for i in range(N):
            gx = (i % 5 + 0.5) * (W / 5)
            gy = (i // 5 + 0.5) * (H / 5)
            x_g[i] = gx
            y_g[i] = gy
            r_g[i] = min(W / 10, H / 10) * 0.9
        inits.append((x_g, y_g, r_g))
        
        # Random init
        r_rand = np.full(N, 0.04)
        x_r = np.random.uniform(r_rand, W - r_rand, N)
        y_r = np.random.uniform(r_rand, H - r_rand, N)
        inits.append((x_r, y_r, r_rand))
        
        best_local_sum = 0.0
        best_local_x, best_local_y, best_local_r = None, None, None
        
        for x0, y0, r0 in inits:
            if elapsed() > deadline:
                break
            
            # SA optimization
            s1, x1, y1, r1 = sa_optimize(x0, y0, r0, W, H, max_iter=15000, T0=0.04)
            
            # Gradient polish
            s2, x2, y2, r2 = gradient_polish(x1, y1, r1, W, H, max_iter=2000)
            
            # Delaunay polish
            s3, x3, y3, r3 = del_delaunay_polish(x2, y2, r2, W, H, max_iter=1000)
            
            if s3 > best_local_sum:
                best_local_sum = s3
                best_local_x, best_local_y, best_local_r = x3, y3, r3
        
        # Final verification
        if best_local_x is not None and best_local_sum > best_sum:
            if check_valid(best_local_x, best_local_y, best_local_r, W, H):
                best_sum = best_local_sum
                best_solution = np.column_stack([best_local_x, best_local_y, best_local_r])
    
    # Fallback
    if best_solution is None:
        side = 5
        cells = []
        for i in range(N):
            x_c = (i % side + 0.5) * (1.0 / side)
            y_c = (i // side + 0.5) * (1.0 / side)
            r_c = 0.5 / side - 1e-4
            cells.append([x_c, y_c, r_c])
        best_solution = np.array(cells)
    
    return best_solution
