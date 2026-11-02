"""Circle packing in perimeter-4 rectangle: maximize sum of 21 circle radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    t0 = time.time()
    
    # Rectangle with perimeter 4: w + h = 2. Use square w=h=1 for best packing.
    W, H = 1.0, 1.0
    
    # Initial guess: hexagonal packing pattern
    # Arrange circles in rows with hexagonal offset
    def hex_layout(n):
        """Create hexagonal packing positions in unit square."""
        # Try different numbers of rows
        best_positions = None
        best_score = -1
        for ncol in range(3, 8):
            # Distribute n circles into ncol columns
            rows = []
            remaining = n
            for c in range(ncol):
                r = (n - c) // ncol if c < n % ncol else (n // ncol)
                # Actually let's do it differently: fill rows
                pass
            
            # Simpler: use a grid with hexagonal offset
            nrow = int(np.ceil(n / ncol))
            positions = []
            for i in range(nrow):
                for j in range(ncol):
                    if len(positions) >= n:
                        break
                    x = (j + 0.5) / ncol
                    y = (i + 0.5) / nrow
                    # Hexagonal offset for alternating rows
                    if i % 2 == 1:
                        x += 0.5 / ncol
                    positions.append([x, y])
            
            if len(positions) == n:
                positions = np.array(positions)
                # Check bounds
                if np.all(positions[:, 0] >= 0) and np.all(positions[:, 0] <= W) and \
                   np.all(positions[:, 1] >= 0) and np.all(positions[:, 1] <= H):
                    # Score: sum of min distance to neighbors / 2 (approximate max radius)
                    score = 0
                    for i in range(n):
                        min_dist = min(W - positions[i, 0], positions[i, 0],
                                      H - positions[i, 1], positions[i, 1])
                        for j in range(n):
                            if i != j:
                                d = np.linalg.norm(positions[i] - positions[j])
                                min_dist = min(min_dist, d / 2)
                        score += min_dist
                    if score > best_score:
                        best_score = score
                        best_positions = positions
        
        if best_positions is None:
            # Fallback: simple grid
            side = int(np.ceil(np.sqrt(n)))
            positions = np.array([[(i + 0.5) / side, (j + 0.5) / side] 
                                  for i in range(side) for j in range(side)][:n])
        return best_positions
    
    # Try multiple initial configurations
    best_result = None
    best_value = -1
    
    configs = []
    
    # Config 1: Hexagonal layout
    pos1 = hex_layout(N)
    configs.append(pos1)
    
    # Config 2: Grid layout
    side = int(np.ceil(np.sqrt(N)))
    pos2 = np.array([[(i + 0.5) / side, (j + 0.5) / side] 
                     for i in range(side) for j in range(side)][:N])
    configs.append(pos2)
    
    # Config 3: Random with some structure
    np.random.seed(42)
    pos3 = np.random.uniform(0.1, 0.9, (N, 2))
    configs.append(pos3)
    
    # Config 4: Concentric ring pattern
    pos4 = np.zeros((N, 2))
    pos4[0] = [0.5, 0.5]
    # Ring 1: 6 circles
    for i in range(6):
        angle = i * 2 * np.pi / 6
        pos4[i+1] = [0.5 + 0.25*np.cos(angle), 0.5 + 0.25*np.sin(angle)]
    # Ring 2: 14 circles
    for i in range(14):
        angle = i * 2 * np.pi / 14
        pos4[i+7] = [0.5 + 0.45*np.cos(angle), 0.5 + 0.45*np.sin(angle)]
    configs.append(pos4)
    
    def objective_and_grad(x):
        """
        x = [x1, y1, x2, y2, ..., xN, yN, r1, r2, ..., rN]
        Minimize: -sum(r_i) + penalty for overlaps and boundary violations
        """
        elapsed = time.time() - t0
        if elapsed > DEADLINE:
            return 1e10, np.zeros_like(x)
        
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        
        # Boundary constraints: circles must fit in [0,W] x [0,H]
        # r_i <= min(x_i, W-x_i, y_i, H-y_i)
        boundary_violation = np.maximum(0, radii - np.minimum(
            np.minimum(centers[:, 0], W - centers[:, 0]),
            np.minimum(centers[:, 1], H - centers[:, 1])
        ))
        
        # Pairwise overlap: r_i + r_j <= ||c_i - c_j||
        overlap_violation = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                viol = radii[i] + radii[j] - d
                if viol > 0:
                    overlap_violation += viol**2
        
        # Objective: negative sum of radii + penalties
        obj = -np.sum(radii) + 100.0 * np.sum(boundary_violation**2) + 100.0 * overlap_violation
        
        # Gradient
        grad = np.zeros_like(x)
        
        # Gradient of -sum(radii)
        grad[2*N:] = -1.0
        
        # Gradient of boundary penalty
        for i in range(N):
            bv = boundary_violation[i]
            if bv > 0:
                # d(boundary_violation)/d(r_i) = 1
                grad[2*N + i] += 200.0 * bv
                # d(boundary_violation)/d(x_i), d(y_i)
                # boundary_violation = max(0, r_i - min(min(x_i, W-x_i), min(y_i, H-y_i)))
                # Let m = min(min(x_i, W-x_i), min(y_i, H-y_i))
                # If x_i < W-x_i and x_i < min(y_i, H-y_i): dm/dx_i = 1, dm/dy_i = 0
                # etc.
                m_x = min(centers[i, 0], W - centers[i, 0])
                m_y = min(centers[i, 1], H - centers[i, 1])
                m = min(m_x, m_y)
                if bv > 0:
                    # Only the binding constraint matters
                    if m_x <= m_y:
                        if centers[i, 0] <= W - centers[i, 0]:
                            grad[2*i] -= 200.0 * bv  # d(-boundary_viol)/dx = -d(boundary_viol)/dx
                        else:
                            grad[2*i] += 200.0 * bv
                    else:
                        if centers[i, 1] <= H - centers[i, 1]:
                            grad[2*i+1] -= 200.0 * bv
                        else:
                            grad[2*i+1] += 200.0 * bv
        
        # Gradient of overlap penalty
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                viol = radii[i] + radii[j] - d
                if viol > 0:
                    diff = centers[i] - centers[j]
                    if d > 1e-10:
                        # d(viol^2)/d(centers[i]) = 2*viol * (centers[i]-centers[j])/d
                        grad[2*i] += 200.0 * viol * diff[0] / d
                        grad[2*i+1] += 200.0 * viol * diff[1] / d
                        grad[2*j] -= 200.0 * viol * diff[0] / d
                        grad[2*j+1] -= 200.0 * viol * diff[1] / d
                    # d(viol^2)/d(r_i) = 2*viol
                    grad[2*N + i] += 200.0 * viol
                    grad[2*N + j] += 200.0 * viol
        
        return obj, grad
    
    for config in configs:
        if time.time() - t0 > DEADLINE - 10:
            break
        
        # Initial radii: small
        r0 = np.full(N, 0.05)
        
        # Build initial x vector
        x0 = np.concatenate([config.ravel(), r0])
        
        # Bounds: centers in [0, W] x [0, H], radii in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, W))
            bounds.append((0.0, H))
        for i in range(N):
            bounds.append((0.0, 0.5))
        
        try:
            result = minimize(objective_and_grad, x0, method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            
            if time.time() - t0 > DEADLINE:
                break
            
            if result.success or result.fun < 1e6:
                centers = result.x[:2*N].reshape(N, 2)
                radii = result.x[2*N:]
                
                # Validate and clamp
                # Ensure circles are within bounds
                for i in range(N):
                    radii[i] = min(radii[i], centers[i, 0], W - centers[i, 0],
                                   centers[i, 1], H - centers[i, 1])
                    radii[i] = max(0, radii[i])
                
                # Check for overlaps and fix
                for it in range(10):
                    fixed = False
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.linalg.norm(centers[i] - centers[j])
                            if radii[i] + radii[j] > d + 1e-9:
                                # Reduce the larger radius
                                if radii[i] >= radii[j]:
                                    radii[i] = max(0, (d - radii[j]) - 1e-6)
                                else:
                                    radii[j] = max(0, (d - radii[i]) - 1e-6)
                                fixed = True
                    if not fixed:
                        break
                
                value = np.sum(radii)
                if value > best_value:
                    best_value = value
                    best_result = np.column_stack([centers, radii])
                    
        except Exception:
            pass
    
    # If no valid result, fallback to seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final validation
    centers = best_result[:, :2]
    radii = best_result[:, 2]
    
    # Ensure all circles are within bounds
    for i in range(N):
        radii[i] = min(radii[i], centers[i, 0], W - centers[i, 0],
                       centers[i, 1], H - centers[i, 1])
        radii[i] = max(0, radii[i])
    
    # Final overlap check
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                # Scale down both proportionally
                excess = radii[i] + radii[j] - d
                if radii[i] + radii[j] > 0:
                    scale = d / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
    
    best_result[:, 2] = radii
    return best_result
