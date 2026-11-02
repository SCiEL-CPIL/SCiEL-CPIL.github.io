import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # 1. Define the smooth objective function
    # We want to maximize sum(r). 
    # Equivalent to minimizing -sum(r) + Penalty(overlaps) + Penalty(boundary)
    # Using softplus for smoothness of max(0, x)
    def softplus(x, beta=100.0):
        # Numerically stable softplus
        return np.where(x > 0, x + np.log1p(np.exp(-beta * x)) / beta, 
                        np.log1p(np.exp(beta * x)) / beta)
        
    def objective_and_grad(x):
        """
        x: [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
        Returns: (value, gradient)
        """
        # Reshape
        centers = x[0:2*N:3].copy() # x coords
        ys = x[1:2*N:3].copy()      # y coords
        radii = x[2::3].copy()      # r coords
        
        # Dimensions: Perimeter 4 => w+h=2. 
        # We fix aspect ratio externally or optimize it? 
        # Let's fix w and h for this specific run.
        # w, h are passed from closure? No, let's make them part of the global state or args.
        # To keep it simple, we will iterate over aspects outside.
        # For now, assume global w, h set by caller.
        
        val = -np.sum(radii)
        grad_val = np.zeros_like(x)
        
        # Gradient of objective: d(-sum r)/dr = -1
        grad_val[2::3] = -1.0
        
        # 1. Pairwise Overlap Penalty
        # Constraint: dist(i,j) >= r_i + r_j
        # Penalty: softplus(r_i + r_j - dist(i,j))
        
        # Compute all pairwise distances
        # Efficiently: only check pairs that might overlap? 
        # With N=21, 210 pairs. Full check is cheap.
        diff = centers[:, np.newaxis] - centers[np.newaxis, :]
        diff_y = ys[:, np.newaxis] - ys[np.newaxis, :]
        dists = np.sqrt(diff**2 + diff_y**2 + 1e-12)
        
        # Avoid self-interaction
        np.fill_diagonal(dists, np.inf)
        
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount: r_i + r_j - dist
        overlap_amt = r_sums - dists
        
        # Only penalize if overlap > 0
        mask = overlap_amt > 0
        if np.any(mask):
            # Softplus penalty
            pen = softplus(overlap_amt[mask])
            val += np.sum(pen)
            
            # Gradient
            # d(softplus(u))/du * du/dx
            # u = r_i + r_j - dist
            # du/dr_i = 1, du/dr_j = 1
            # du/dx_i = -(x_i - x_j)/dist, du/dy_i = -(y_i - y_j)/dist
            # du/dx_j = (x_i - x_j)/dist, du/dy_j = (y_i - y_j)/dist
            
            # We need to accumulate gradients for all pairs in mask
            # This is O(N^2) loop, acceptable for N=21
            indices = np.where(mask)
            for i, j in zip(*indices):
                if i == j: continue
                u = overlap_amt[i, j]
                # d(softplus)/du
                # stable derivative: sigmoid(beta * u) approx 1 if u>0, 0 if u<0
                # For large beta, it's a step function. 
                # Let's use the derivative of the softplus we defined.
                # d/dx log1p(exp(beta*x))/beta = exp(beta*x)/(1+exp(beta*x)) = sigmoid(beta*x)
                # If x > 0, sigmoid ~ 1. If x < 0, sigmoid ~ 0.
                # Since mask is u > 0, derivative is ~ 1.
                # Let's use 1.0 for simplicity and robustness, or compute exactly.
                # Exact: 
                exp_term = np.exp(-beta * u) # u > 0, so exp(-beta*u) is small
                # Wait, softplus(x) = log(1+exp(x)). d/dx = 1/(1+exp(-x)) = sigmoid(x).
                # My softplus implementation: 
                # if x>0: x + log(1+exp(-beta*x))/beta
                # derivative: 1 - exp(-beta*x)/(1+exp(-beta*x)) = exp(beta*x)/(1+exp(beta*x)) = sigmoid(beta*x)
                # Since u > 0 and beta=100, sigmoid(100*u) is essentially 1.
                d_soft = 1.0
                
                # Contribution to value gradient
                # d(val)/dr_i += d_soft * 1
                # d(val)/dr_j += d_soft * 1
                # d(val)/dx_i += d_soft * (-(x_i - x_j)/dist)
                # d(val)/dy_i += d_soft * (-(y_i - y_j)/dist)
                # d(val)/dx_j += d_soft * ((x_i - x_j)/dist)
                # d(val)/dy_j += d_soft * ((y_i - y_j)/dist)
                
                idx_ri = 2*i + 2
                idx_rj = 2*j + 2
                idx_xi = 2*i
                idx_yi = 2*i + 1
                idx_xj = 2*j
                idx_yj = 2*j + 1
                
                grad_val[idx_ri] += d_soft
                grad_val[idx_rj] += d_soft
                
                dist_ij = dists[i, j]
                dx = (centers[i] - centers[j]) / dist_ij
                dy = (ys[i] - ys[j]) / dist_ij
                
                grad_val[idx_xi] += -d_soft * dx
                grad_val[idx_yi] += -d_soft * dy
                grad_val[idx_xj] += d_soft * dx
                grad_val[idx_yj] += d_soft * dy
        
        # 2. Boundary Penalty
        # Constraint: r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i
        # Penalty: softplus(r_i - x_i) + softplus(r_i - (w-x_i)) + ...
        
        for i in range(N):
            r = radii[i]
            xi = centers[i]
            yi = ys[i]
            
            # Left
            u1 = r - xi
            if u1 > 0:
                val += softplus(u1)
                grad_val[2*i + 2] += 1.0 # dr
                grad_val[2*i] += -1.0    # dx
            # Right
            u2 = r - (w - xi)
            if u2 > 0:
                val += softplus(u2)
                grad_val[2*i + 2] += 1.0 # dr
                grad_val[2*i] += 1.0     # dx (d(w-x)/dx = -1, so d(r - (w-x))/dx = -(-1) = 1? 
                                         # Wait. u = r - w + x. du/dx = 1. Correct.)
            # Bottom
            u3 = r - yi
            if u3 > 0:
                val += softplus(u3)
                grad_val[2*i + 2] += 1.0 # dr
                grad_val[2*i + 1] += -1.0 # dy
            # Top
            u4 = r - (h - yi)
            if u4 > 0:
                val += softplus(u4)
                grad_val[2*i + 2] += 1.0 # dr
                grad_val[2*i + 1] += 1.0 # dy
        
        return val, grad_val

    def check_validity(circles, w, h, tol=1e-5):
        xy = circles[:, :2]
        r = circles[:, 2]
        if np.any(xy[:, 0] < r - tol) or np.any(xy[:, 0] > w - r + tol):
            return False
        if np.any(xy[:, 1] < r - tol) or np.any(xy[:, 1] > h - r + tol):
            return False
        dists = np.sqrt(np.sum((xy[:, np.newaxis] - xy[np.newaxis, :])**2, axis=-1))
        r_sums = r[:, np.newaxis] + r[np.newaxis, :]
        mask = ~np.eye(N, dtype=bool)
        if np.any(dists[mask] < r_sums[mask] - tol):
            return False
        return True

    # Helper to generate hexagonal initial guess
    def generate_hex(w, h, r_target):
        """
        Generate N circles in a hexagonal pattern scaled to fit w,h.
        """
        points = []
        # Hex lattice vectors
        a = 2.0 * r_target
        # a is diameter
        # Distance between centers in same row: a
        # Distance between rows: a * sqrt(3)/2
        
        # Estimate number of rows
        # Area approx N * (sqrt(3)/2) * a^2
        # w * h ~ N * 0.866 * a^2
        # a ~ sqrt(w*h / (N * 0.866))
        
        # We'll just place them and then optimize.
        # Start with a grid and shear it?
        # Simpler: Place in hex pattern, then scale to fit.
        
        # Determine grid size
        # Approximate rows
        cols = int(np.ceil(np.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        
        # Create hex points
        pts = []
        for i in range(rows):
            for j in range(cols):
                if len(pts) >= N: break
                x = (j + 0.5 * (i % 2)) * a
                y = i * a * np.sqrt(3) / 2
                pts.append([x, y])
            if len(pts) >= N: break
        
        pts = np.array(pts[:N])
        
        # Center the cloud
        pts -= np.mean(pts, axis=0)
        
        # Scale to fit w, h
        # Find bounding box
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        width_pts = max_x - min_x
        height_pts = max_y - min_y
        
        # Scale factor
        scale_x = (w - 2*r_target) / width_pts if width_pts > 0 else 1
        scale_y = (h - 2*r_target) / height_pts if height_pts > 0 else 1
        scale = min(scale_x, scale_y, 1.0) # Don't upscale too much initially
        
        pts *= scale
        
        # Center in rectangle
        pts += np.array([w/2, h/2])
        
        # Initial radii
        radii = np.full(N, r_target * 0.9)
        
        # Construct vector
        x_vec = np.zeros(3*N)
        x_vec[0:2*N:3] = pts[:, 0]
        x_vec[1:2*N:3] = pts[:, 1]
        x_vec[2::3] = radii
        
        return x_vec

    # Main Loop
    random.seed(42)
    np.random.seed(42)
    
    aspects = [0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    
    for aspect in aspects:
        if time.time() > deadline:
            break
            
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Estimate optimal radius
        # Packing density hex ~ 0.9069
        # Area_rect = w*h
        # Area_circles = N * pi * r^2
        # r ~ sqrt(w*h * 0.9 / (N * pi))
        r_est = np.sqrt(w * h * 0.9 / (N * np.pi))
        
        # Try a few restarts for this aspect
        for restart in range(3):
            if time.time() > deadline:
                break
                
            # Initial guess
            if restart == 0:
                x0 = generate_hex(w, h, r_est)
            else:
                # Perturb hex
                x0 = generate_hex(w, h, r_est)
                noise = np.random.normal(0, 0.01 * r_est, size=3*N)
                # Don't move radii too much
                noise[2::3] *= 0.1
                x0 += noise
                # Ensure positive radii
                x0[2::3] = np.maximum(x0[2::3], 0.01)
                # Ensure inside bounds (rough)
                x0[0:2*N:3] = np.clip(x0[0:2*N:3], x0[2::3], w - x0[2::3])
                x0[1:2*N:3] = np.clip(x0[1:2*N:3], x0[2::3], h - x0[2::3])

            # Optimize
            try:
                res = minimize(objective_and_grad, x0, method='SLSQP', 
                               jac=True, # We return gradient
                               bounds=[(0, None)] * N + [(0, w)] * N + [(0, h)] * N, 
                               # Note: bounds structure: [r0, x0, y0, r1, x1, y1...]
                               # My x vector is [x0, y0, r0, x1, y1, r1...]
                               # So bounds: 
                               # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
                               options={'maxiter': 500, 'ftol': 1e-8, 'disp': False})
                
                if res.success or res.fun < 0: # res.fun is negative sum r + penalties
                    # Extract result
                    x_opt = res.x
                    centers = np.column_stack([x_opt[0:2*N:3], x_opt[1:2*N:3]])
                    radii = x_opt[2::3]
                    circles = np.column_stack([centers, radii])
                    
                    # Validate
                    if check_validity(circles, w, h):
                        score = np.sum(radii)
                        if score > best_score:
                            best_score = score
                            best_result = circles.copy()
                            
            except Exception as e:
                pass

    if best_result is None:
        # Fallback: Grid
        w, h = 1.0, 1.0
        res = np.zeros((N, 3))
        r = 0.08
        for i in range(N):
            row = i // 5
            col = i % 5
            res[i, 0] = (col + 0.5) * 0.2
            res[i, 1] = (row + 0.5) * 0.2
            res[i, 2] = r
        return res

    return best_result
