import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # 1. Generate initial hexagonal layout
    # We want to pack 26 circles. 
    # A 5x5 square grid has 25 circles. We need 26.
    # Let's use a hexagonal grid.
    
    def get_initial_layout():
        # Generate a hex grid of points
        # r is the "radius" in the grid sense, distance between centers is 2r
        # Let's estimate r. Area of square is 1. 
        # Approx area per circle in hex pack is ~ 2.17 r^2.
        # 26 * 2.17 r^2 <= 1 => r^2 <= 1/56.4 => r <= 0.133
        # Let's start with r = 0.12
        r0 = 0.12
        centers = []
        # Generate enough points
        for row in range(10):
            for col in range(10):
                x = col * 2 * r0
                y = row * np.sqrt(3) * r0
                if row % 2 == 1:
                    x += r0
                centers.append((x, y))
        
        # Sort by distance from center (0.5, 0.5) to get a compact cluster
        centers.sort(key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
        centers = centers[:N]
        
        # Calculate bounding box to scale and center
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        
        # Scale to fit in [0.1, 0.9] to allow room for optimization
        scale = min(0.8 / width, 0.8 / height)
        
        scaled_centers = []
        for x, y in centers:
            sx = (x - min_x) * scale + 0.1
            sy = (y - min_y) * scale + 0.1
            scaled_centers.append((sx, sy))
            
        # Initial radii
        # Set all radii to the same value, slightly less than half the min distance
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((scaled_centers[i][0]-scaled_centers[j][0])**2 + (scaled_centers[i][1]-scaled_centers[j][1])**2)
                if d < min_dist:
                    min_dist = d
        # Also check boundary
        min_boundary = np.inf
        for x, y in scaled_centers:
            min_boundary = min(min_boundary, x, 1-x, y, 1-y)
            
        r_init = min(min_dist / 2, min_boundary) * 0.95
        
        v = []
        for x, y in scaled_centers:
            v.extend([x, y, r_init])
        return np.array(v)

    best_score = -1
    best_v = None
    
    # Softplus function for smooth penalties
    def softplus(z, eps):
        return np.logaddexp(0, z / eps) * eps

    # Objective function using smooth penalties
    def objective(v, penalty=1000.0, eps=1e-4):
        x = v[0::3]
        y = v[1::3]
        r = np.maximum(v[2::3], 1e-6)
        
        score = -np.sum(r)
        
        # Boundary violations
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0
        wall_pen = np.sum(softplus(lx, eps)**2) + np.sum(softplus(ux, eps)**2) + \
                   np.sum(softplus(ly, eps)**2) + np.sum(softplus(uy, eps)**2)
        
        # Overlap violations
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (r[iu[0]] + r[iu[1]]) - dist
        pair_pen = np.sum(softplus(overlap, eps)**2)
        
        return score + penalty * (wall_pen + pair_pen)

    # Analytic gradient of the smooth penalty objective
    def gradient(v, penalty=1000.0, eps=1e-4):
        x = v[0::3]
        y = v[1::3]
        r = np.maximum(v[2::3], 1e-6)
        
        grad = np.zeros_like(v)
        
        # Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        # Helper for derivative of softplus(z, eps)^2
        # d/dz [softplus(z, eps)^2] = 2 * softplus(z, eps) * sigmoid(z/eps)
        def d_softplus_sq(z, eps):
            sp = softplus(z, eps)
            sig = 1.0 / (1.0 + np.exp(-z / eps))
            return 2.0 * sp * sig

        # Boundary gradients
        for i in range(N):
            # Left wall: z = r - x
            z = r[i] - x[i]
            dz = d_softplus_sq(z, eps)
            grad[i*3] += -penalty * dz
            grad[i*3+2] += penalty * dz
            
            # Right wall: z = x + r - 1
            z = x[i] + r[i] - 1.0
            dz = d_softplus_sq(z, eps)
            grad[i*3] += penalty * dz
            grad[i*3+2] += penalty * dz
            
            # Bottom wall: z = r - y
            z = r[i] - y[i]
            dz = d_softplus_sq(z, eps)
            grad[i*3+1] += -penalty * dz
            grad[i*3+2] += penalty * dz
            
            # Top wall: z = y + r - 1
            z = y[i] + r[i] - 1.0
            dz = d_softplus_sq(z, eps)
            grad[i*3+1] += penalty * dz
            grad[i*3+2] += penalty * dz

        # Pairwise gradients
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (r[iu[0]] + r[iu[1]]) - dist
        
        dz = d_softplus_sq(overlap, eps)
        ux = dx / dist
        uy = dy / dist
        
        # Accumulate into grad
        # d/dx_i: -penalty * dz * ux
        # d/dx_j: penalty * dz * ux
        # d/dr_i: penalty * dz
        # d/dr_j: penalty * dz
        
        np.add.at(grad[0::3], iu[0], -penalty * dz * ux)
        np.add.at(grad[0::3], iu[1], penalty * dz * ux)
        np.add.at(grad[1::3], iu[0], -penalty * dz * uy)
        np.add.at(grad[1::3], iu[1], penalty * dz * uy)
        np.add.at(grad[2::3], iu[0], penalty * dz)
        np.add.at(grad[2::3], iu[1], penalty * dz)
        
        return grad

    def repair(v):
        """Enforce validity by shrinking radii if constraints are violated."""
        p = v.reshape(N, 3).copy()
        p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
        p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
        p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
        
        # Enforce wall constraints
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
        
        # Enforce pairwise constraints iteratively
        for _ in range(50):
            iu = np.triu_indices(N, k=1)
            dx = p[iu[0], 0] - p[iu[1], 0]
            dy = p[iu[0], 1] - p[iu[1], 1]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
            active = overlap > 0
            if not np.any(active):
                break
            for k in range(len(active)):
                if not active[k]:
                    continue
                i = iu[0][k]
                j = iu[1][k]
                ov = overlap[k]
                # Shrink the larger radius or split if equal
                if p[i, 2] >= p[j, 2]:
                    shrink = min(ov, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink
                    rem = ov - shrink
                    if rem > 0:
                        shrink2 = min(rem, p[j, 2] - 1e-6)
                        p[j, 2] -= shrink2
                else:
                    shrink = min(ov, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink
                    rem = ov - shrink
                    if rem > 0:
                        shrink2 = min(rem, p[i, 2] - 1e-6)
                        p[i, 2] -= shrink2
            # Re-enforce wall constraints after shrinking
            wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                              np.minimum(p[:, 1], 1.0 - p[:, 1]))
            p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
            
        return p.flatten()

    def is_valid(v, tol=1e-7):
        p = v.reshape(N, 3)
        x, y, r = p[:, 0], p[:, 1], p[:, 2]
        if np.any(r < 0) or np.any(x < -tol) or np.any(x > 1+tol) or np.any(y < -tol) or np.any(y > 1+tol):
            return False
        if np.any(r > x + tol) or np.any(r > 1-x + tol) or np.any(r > y + tol) or np.any(r > 1-y + tol):
            return False
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(r[iu[0]] + r[iu[1]] > dist + tol):
            return False
        return True

    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N
    
    v_init = get_initial_layout()
    
    # Multi-start loop
    while time.time() < deadline:
        # Optimize
        try:
            res = minimize(objective, v_init, jac=gradient, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            v_res = res.x
            
            # Repair to ensure strict validity
            v_repaired = repair(v_res)
            
            if is_valid(v_repaired):
                score = np.sum(v_repaired[2::3])
                if score > best_score:
                    best_score = score
                    best_v = v_repaired.copy()
            
            # Perturb for next restart
            if best_v is not None and np.random.rand() < 0.6:
                # Perturb around best solution
                v_init = best_v + np.random.normal(0, 0.03, size=3*N)
            else:
                # New random-ish hex layout
                v_init = get_initial_layout()
                v_init = v_init + np.random.normal(0, 0.05, size=3*N)
            
            # Clip to bounds
            v_init[0::3] = np.clip(v_init[0::3], 0, 1)
            v_init[1::3] = np.clip(v_init[1::3], 0, 1)
            v_init[2::3] = np.clip(v_init[2::3], 1e-6, 0.5)
            
        except Exception as e:
            # If optimization fails, try a smaller step or restart
            v_init = get_initial_layout()
            v_init = v_init + np.random.normal(0, 0.02, size=3*N)
            v_init[0::3] = np.clip(v_init[0::3], 0, 1)
            v_init[1::3] = np.clip(v_init[1::3], 0, 1)
            v_init[2::3] = np.clip(v_init[2::3], 1e-6, 0.5)

    if best_v is None:
        # Fallback
        v_fallback = get_initial_layout()
        v_fallback = repair(v_fallback)
        return v_fallback.reshape(N, 3)
    
    return best_v.reshape(N, 3)
