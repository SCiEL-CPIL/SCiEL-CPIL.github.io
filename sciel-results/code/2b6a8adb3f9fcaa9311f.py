import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 110.0  # safety margin before 120s kill

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    
    np.random.seed(42)
    
    # Try several aspect ratios (w + h = 2)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43]
    
    for w in aspect_ratios:
        if time.time() - start_time > DEADLINE - 25:
            break
        h = 2.0 - w
        if h <= 0:
            continue
        
        for restart in range(2):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            # Create initial grid packing
            cols, rows = 5, 5
            positions = []
            for i in range(cols):
                for j in range(rows):
                    if len(positions) < N:
                        x = (i + 0.5) * w / cols
                        y = (j + 0.5) * h / rows
                        positions.append([x, y])
            
            if restart > 0:
                for pos in positions:
                    pos[0] += np.random.uniform(-0.05, 0.05)
                    pos[1] += np.random.uniform(-0.05, 0.05)
            
            positions = np.array(positions[:N])
            
            # Estimate initial radius from min spacing
            min_spacing = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(positions[i] - positions[j])
                    min_spacing = min(min_spacing, d)
            r_init = min(min_spacing / 2, w / 2, h / 2) * 0.95
            
            # Force-based pre-relaxation to get a good starting point
            for _ in range(200):
                forces = np.zeros_like(positions)
                for i in range(N):
                    for j in range(i + 1, N):
                        d = positions[i] - positions[j]
                        dist = np.linalg.norm(d)
                        if dist < 1e-10:
                            d = np.array([0.01, 0.01])
                            dist = np.linalg.norm(d)
                        min_dist = 2 * r_init
                        if dist < min_dist:
                            force = (min_dist - dist) / dist * d
                            forces[i] += force
                            forces[j] -= force
                # Boundary forces
                for i in range(N):
                    if positions[i, 0] < r_init:
                        forces[i, 0] += (r_init - positions[i, 0])
                    if positions[i, 0] > w - r_init:
                        forces[i, 0] -= (positions[i, 0] - (w - r_init))
                    if positions[i, 1] < r_init:
                        forces[i, 1] += (r_init - positions[i, 1])
                    if positions[i, 1] > h - r_init:
                        forces[i, 1] -= (positions[i, 1] - (h - r_init))
                positions += 0.1 * forces
                for i in range(N):
                    positions[i, 0] = np.clip(positions[i, 0], r_init, w - r_init)
                    positions[i, 1] = np.clip(positions[i, 1], r_init, h - r_init)
            
            # Build optimization variables
            x0 = np.zeros(N * 3)
            for i in range(N):
                x0[3*i] = positions[i, 0]
                x0[3*i+1] = positions[i, 1]
                x0[3*i+2] = r_init
            
            def objective(x):
                return -np.sum(x[2::3])
            
            # Constraints
            constraints = []
            for i in range(N):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[3*i] - x[3*i+2]})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[3*i+1] - x[3*i+2]})
            for i in range(N):
                for j in range(i + 1, N):
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, i=i, j=j:
                            (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2
                    })
            
            bounds = [(0, w), (0, h), (1e-8, min(w, h) / 2)] * N
            
            try:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                                  constraints=constraints,
                                  options={'maxiter': 300, 'ftol': 1e-8})
            except Exception:
                continue
            
            x_opt = result.x
            circles = x_opt.reshape(N, 3)
            
            # Validate
            valid = True
            for i in range(N):
                cx, cy, r = circles[i]
                if r <= 0 or cx < -1e-9 or cy < -1e-9 or cx + r > w + 1e-9 or cy + r > h + 1e-9:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = np.sqrt((circles[i,0]-circles[j,0])**2 + (circles[i,1]-circles[j,1])**2)
                        if dist < circles[i,2] + circles[j,2] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = circles.copy()
    
    # Fallback: seed grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
