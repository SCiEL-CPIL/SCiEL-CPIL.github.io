import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    def compute_objective(pts, radii):
        """Compute sum of radii for a valid packing, or -1 if invalid."""
        for i in range(N):
            x, y = pts[i]
            r = radii[i]
            if r > x + 1e-9 or r > 1 - x + 1e-9 or r > y + 1e-9 or r > 1 - y + 1e-9:
                return -1.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
                if radii[i] + radii[j] > d + 1e-9:
                    return -1.0
        return np.sum(radii)

    def penalty_objective(vec, penalty_weight):
        """
        vec: flattened array of shape (3*N,) = [x0,y0,r0, x1,y1,r1, ...]
        Returns: negative sum of radii + penalty * (constraint violations)
        """
        pts = vec[0::3].reshape(N, 2)
        radii = vec[1::3].reshape(N)
        
        # Objective: maximize sum of radii -> minimize -sum(radii)
        obj = -np.sum(radii)
        
        # Penalty for boundary violations
        penalty = 0.0
        for i in range(N):
            x, y, r = pts[i][0], pts[i][1], radii[i]
            # Boundary: r <= x, r <= 1-x, r <= y, r <= 1-y
            penalty += max(0, r - x)**2
            penalty += max(0, r - (1 - x))**2
            penalty += max(0, r - y)**2
            penalty += max(0, r - (1 - y))**2
            # Radius non-negativity
            penalty += max(0, -r)**2
        
        # Penalty for overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d = np.sqrt(dx*dx + dy*dy)
                violation = radii[i] + radii[j] - d
                if violation > 0:
                    penalty += violation**2
        
        return obj + penalty_weight * penalty

    def penalty_gradient(vec, penalty_weight):
        """
        Gradient of penalty_objective.
        """
        pts = vec[0::3].reshape(N, 2)
        radii = vec[1::3].reshape(N)
        
        grad = np.zeros(3 * N)
        
        # Gradient of -sum(radii)
        for i in range(N):
            grad[3*i + 2] = -1.0
        
        # Gradient of boundary penalties
        for i in range(N):
            x, y, r = pts[i][0], pts[i][1], radii[i]
            # r - x > 0
            if r - x > 0:
                grad[3*i + 2] += 2 * penalty_weight * (r - x)  # dr
                grad[3*i + 0] += -2 * penalty_weight * (r - x)  # dx
            # r - (1-x) > 0
            if r - (1 - x) > 0:
                grad[3*i + 2] += 2 * penalty_weight * (r - (1 - x))
                grad[3*i + 0] += -2 * penalty_weight * (r - (1 - x))
            # r - y > 0
            if r - y > 0:
                grad[3*i + 2] += 2 * penalty_weight * (r - y)
                grad[3*i + 1] += -2 * penalty_weight * (r - y)
            # r - (1-y) > 0
            if r - (1 - y) > 0:
                grad[3*i + 2] += 2 * penalty_weight * (r - (1 - y))
                grad[3*i + 1] += -2 * penalty_weight * (r - (1 - y))
            # -r > 0
            if -r > 0:
                grad[3*i + 2] += -2 * penalty_weight * r
        
        # Gradient of overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d = np.sqrt(dx*dx + dy*dy)
                violation = radii[i] + radii[j] - d
                if violation > 0 and d > 1e-12:
                    factor = 2 * penalty_weight * violation
                    # d/dxi of (ri + rj - d)^2 = 2*(ri+rj-d) * (-dx/d)
                    grad[3*i + 0] += factor * (-dx / d)
                    grad[3*i + 1] += factor * (-dy / d)
                    grad[3*j + 0] += factor * (dx / d)
                    grad[3*j + 1] += factor * (dy / d)
                    # d/dri
                    grad[3*i + 2] += factor
                    grad[3*j + 2] += factor
        
        return grad

    def hex_init():
        """Hexagonal lattice initialization."""
        r0 = 0.08
        dx = 2 * r0
        dy = r0 * np.sqrt(3)
        positions = []
        for j in range(5):
            for i in range(6):
                x = (i + 0.5 * (j % 2)) * dx
                y = j * dy
                positions.append([x, y])
        pos = np.array(positions[:N])
        pos[:, 0] -= pos[:, 0].mean()
        pos[:, 1] -= pos[:, 1].mean()
        x_max, x_min = pos[:, 0].max(), pos[:, 0].min()
        y_max, y_min = pos[:, 1].max(), pos[:, 1].min()
        scale_x = 0.8 / (x_max - x_min) if x_max > x_min else 1.0
        scale_y = 0.8 / (y_max - y_min) if y_max > y_min else 1.0
        scale = min(scale_x, scale_y)
        pos *= scale
        pos[:, 0] += 0.5
        pos[:, 1] += 0.5
        pts = pos.tolist()
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                min_dist = min(min_dist, d)
        r_init = min_dist * 0.4
        radii = np.full(N, r_init)
        return pts, radii

    def ring_init():
        """Ring-based initialization."""
        pts = []
        radii = []
        center = [0.5, 0.5]
        pts.append(center)
        radii.append(0.1)
        r1 = 0.12
        d1 = 0.25
        for k in range(6):
            angle = k * np.pi / 3
            x = 0.5 + d1 * np.cos(angle)
            y = 0.5 + d1 * np.sin(angle)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pts.append([x, y])
            radii.append(r1)
        r2 = 0.08
        d2 = 0.42
        for k in range(12):
            angle = k * 2 * np.pi / 12
            x = 0.5 + d2 * np.cos(angle)
            y = 0.5 + d2 * np.sin(angle)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pts.append([x, y])
            radii.append(r2)
        extra_positions = [
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
            [0.5, 0.08], [0.5, 0.92], [0.08, 0.5]
        ]
        for pos in extra_positions[:7]:
            pts.append(pos)
            radii.append(0.06)
        return pts, np.array(radii)

    def random_init():
        """Random initialization with minimum distance guarantee."""
        pts = []
        radii = []
        min_dist = 0.15
        attempts = 0
        while len(pts) < N and attempts < 10000:
            attempts += 1
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            ok = True
            for px, py in pts:
                if np.sqrt((x - px)**2 + (y - py)**2) < min_dist:
                    ok = False
                    break
            if ok:
                pts.append([x, y])
                radii.append(0.05)
        if len(pts) < N:
            min_dist = 0.1
            while len(pts) < N and attempts < 20000:
                attempts += 1
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                ok = True
                for px, py in pts:
                    if np.sqrt((x - px)**2 + (y - py)**2) < min_dist:
                        ok = False
                        break
                if ok:
                    pts.append([x, y])
                    radii.append(0.04)
        while len(pts) < N:
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            pts.append([x, y])
            radii.append(0.03)
        return pts, np.array(radii[:N])

    def grid_init():
        """Grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        pts = []
        radii = []
        for i in range(side):
            for j in range(side):
                if len(pts) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                pts.append([x, y])
                radii.append(0.5 / side * 0.8)
        return pts, np.array(radii[:N])

    def optimize_from_init(pts, radii, time_limit):
        """
        Run L-BFGS-B optimization from a given initialization.
        Returns best (pts, radii, value) found.
        """
        # Flatten to vector
        vec = np.zeros(3 * N)
        for i in range(N):
            vec[3*i + 0] = pts[i][0]
            vec[3*i + 1] = pts[i][1]
            vec[3*i + 2] = radii[i]
        
        best_vec = vec.copy()
        best_val = compute_objective(
            vec[0::3].reshape(N, 2), 
            vec[1::3].reshape(N)
        )
        if best_val < 0:
            best_val = 0.0
        
        # Adaptive penalty weight
        penalty_weight = 10.0
        
        # Multiple L-BFGS-B runs with increasing penalty
        max_iterations = 500
        
        while time.time() < time_limit:
            # Run L-BFGS-B
            result = minimize(
                penalty_objective,
                vec,
                args=(penalty_weight,),
                jac=penalty_gradient,
                method='L-BFGS-B',
                options={'maxiter': max_iterations, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_vec = result.x
            new_pts = new_vec[0::3].reshape(N, 2)
            new_radii = new_vec[1::3].reshape(N)
            
            new_val = compute_objective(new_pts, new_radii)
            if new_val > best_val:
                best_val = new_val
                best_vec = new_vec.copy()
            
            # Update vec for next iteration
            vec = new_vec
            
            # Increase penalty weight if still infeasible
            if new_val < 0:
                penalty_weight *= 2.0
                if penalty_weight > 10000:
                    penalty_weight = 10000
            else:
                # Feasible, try to improve
                penalty_weight = min(penalty_weight * 1.5, 10000)
            
            # Check time
            if time.time() > time_limit:
                break
        
        return best_vec[0::3].reshape(N, 2).tolist(), best_vec[1::3].reshape(N), best_val

    best_overall_val = 0.0
    best_overall_pts = None
    best_overall_radii = None
    
    # Time allocation for different initializations
    time_per_init = 30  # 4 inits * 30s = 120s, leaving 25s buffer
    
    inits = [
        ("hex", hex_init),
        ("ring", ring_init),
        ("random", random_init),
        ("grid", grid_init),
    ]
    
    for init_name, init_func in inits:
        if time.time() > deadline - 15:
            break
        
        init_start = time.time()
        pts, radii = init_func()
        
        # Make initial state feasible by shrinking radii
        for _ in range(50):
            if compute_objective(pts, radii) >= 0:
                break
            radii = radii * 0.8
        if compute_objective(pts, radii) < 0:
            radii = np.full(N, 0.01)
        
        # Run optimization
        init_time_left = deadline - (time.time() - init_start) - 5
        if init_time_left < 10:
            init_time_left = 10
        
        result_pts, result_radii, result_val = optimize_from_init(
            pts, radii, start_time + init_start - time.time() + init_time_left
        )
        
        if result_val > best_overall_val:
            best_overall_val = result_val
            best_overall_pts = result_pts
            best_overall_radii = result_radii
    
    # Final fallback
    if best_overall_pts is None or best_overall_val <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.9
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    # Convert to numpy and final validation
    pts_arr = np.array(best_overall_pts)
    radii_arr = np.array(best_overall_radii)
    
    # Final safety: ensure all constraints are met
    for _ in range(20):
        valid = True
        for i in range(N):
            x, y = pts_arr[i]
            r = radii_arr[i]
            if r > x + 1e-7 or r > 1 - x + 1e-7 or r > y + 1e-7 or r > 1 - y + 1e-7:
                valid = False
                break
            for j in range(i + 1, N):
                d = np.sqrt((pts_arr[i][0] - pts_arr[j][0])**2 + (pts_arr[i][1] - pts_arr[j][1])**2)
                if radii_arr[i] + radii_arr[j] > d + 1e-7:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        radii_arr = radii_arr * 0.95
    
    # Ensure non-negative
    radii_arr = np.maximum(radii_arr, 0)
    
    return np.column_stack((pts_arr[:, 0], pts_arr[:, 1], radii_arr))
