"""Rewritten program: Circle packing in unit square for N=26.

Strategy:
1. Generate diverse initial configurations (grid-based, jittered, random).
2. Use SLSQP optimization to maximize sum of radii subject to disjointness and boundary constraints.
3. Apply a final high-precision polish pass with tighter tolerances.
4. Deadline guard to ensure completion within 120s.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0  # Safety margin before 120s limit


def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Objective: maximize sum of radii (minimize negative sum)
    def objective(x):
        # x = [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
        radii = x[2::3]
        return -np.sum(radii)
    
    def constraints(x):
        cons = []
        for i in range(N):
            xi = x[3*i]
            yi = x[3*i+1]
            ri = x[3*i+2]
            
            # Boundary constraints: circle must be inside [0,1]^2
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, xi=xi, yi=yi, ri=ri: xi - ri})  # xi - ri >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, xi=xi, yi=yi, ri=ri: 1.0 - xi - ri})  # 1 - xi - ri >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, xi=xi, yi=yi, ri=ri: yi - ri})  # yi - ri >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, xi=xi, yi=yi, ri=ri: 1.0 - yi - ri})  # 1 - yi - ri >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, ri=ri: ri})  # ri >= 0
        
        # Disjointness constraints: distance between centers >= sum of radii
        for i in range(N):
            for j in range(i+1, N):
                xi = x[3*i]
                yi = x[3*i+1]
                ri = x[3*i+2]
                xj = x[3*j]
                yj = x[3*j+1]
                rj = x[3*j+2]
                # sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
                # Use squared form to avoid sqrt: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                # But this is not equivalent for ineq. Use the direct form.
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt((x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2) - x[3*i+2] - x[3*j+2]
                })
        
        return cons
    
    def bounds():
        b = []
        for i in range(N):
            b.append((0.0, 1.0))  # x
            b.append((0.0, 1.0))  # y
            b.append((0.0, 0.5))  # r (max radius is 0.5)
        return b
    
    # Generate initial configurations
    def grid_init():
        """Grid-based initial configuration."""
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    def jittered_grid_init(rng):
        """Grid with random jitter."""
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side + rng.uniform(-0.05, 0.05)
                y = (j + 0.5) / side + rng.uniform(-0.05, 0.05)
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    def random_init(rng):
        """Random initial configuration with small radii."""
        cells = []
        for i in range(N):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            r = rng.uniform(0.02, 0.08)
            cells.append([x, y, r])
        return np.array(cells)
    
    def cluster_init(rng):
        """Cluster-based initial configuration."""
        # Create 5 clusters
        n_clusters = 5
        cluster_centers = rng.uniform(0.15, 0.85, size=(n_clusters, 2))
        cells = []
        for i in range(N):
            cluster_idx = i % n_clusters
            cx, cy = cluster_centers[cluster_idx]
            angle = rng.uniform(0, 2*np.pi)
            dist = rng.uniform(0, 0.12)
            x = cx + dist * np.cos(angle)
            y = cy + dist * np.sin(angle)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            r = rng.uniform(0.03, 0.07)
            cells.append([x, y, r])
        return np.array(cells)
    
    # Generate diverse initializations
    rng = np.random.default_rng(42)
    initializations = []
    
    # Grid
    initializations.append(grid_init())
    
    # Jittered grids
    for _ in range(3):
        initializations.append(jittered_grid_init(rng))
    
    # Random
    for _ in range(3):
        initializations.append(random_init(rng))
    
    # Cluster
    for _ in range(3):
        initializations.append(cluster_init(rng))
    
    # Hexagonal-like pattern
    def hex_init():
        cells = []
        # Hexagonal packing pattern
        dx = 1.0 / 5
        dy = dx * np.sqrt(3) / 2
        for row in range(6):
            for col in range(5):
                if len(cells) >= N:
                    break
                x = (col + 0.5) * dx + (0.5 if row % 2 else 0) * dx
                y = (row + 0.5) * dy
                if x < 1.0 and y < 1.0:
                    r = dx * 0.45
                    cells.append([x, y, r])
        # Fill remaining
        while len(cells) < N:
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            r = 0.05
            cells.append([x, y, r])
        return np.array(cells[:N])
    
    initializations.append(hex_init())
    initializations.append(hex_init())
    
    best_result = None
    best_value = -np.inf
    
    # Try each initialization
    for init_idx, init in enumerate(initializations):
        if time_left() < 10:
            break
        
        # Flatten to 1D
        x0 = init.flatten()
        
        # SLSQP optimization
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds(),
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.success:
                value = -result.fun
                if value > best_value:
                    best_value = value
                    best_result = result.x.reshape(N, 3)
        except Exception:
            pass
    
    # If no valid result, use grid
    if best_result is None:
        best_result = grid_init()
        best_value = np.sum(best_result[:, 2])
    
    # Final polish pass with tighter tolerances
    if time_left() > 15:
        try:
            result = minimize(
                objective,
                best_result.flatten(),
                method='SLSQP',
                bounds=bounds(),
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.success:
                value = -result.fun
                if value > best_value:
                    best_value = value
                    best_result = result.x.reshape(N, 3)
        except Exception:
            pass
    
    # Second perturbed polish attempt
    if time_left() > 15:
        try:
            perturbed = best_result.copy()
            perturbed[:, :2] += rng.uniform(-0.01, 0.01, size=(N, 2))
            perturbed[:, :2] = np.clip(perturbed[:, :2], 0.0, 1.0)
            
            result = minimize(
                objective,
                perturbed.flatten(),
                method='SLSQP',
                bounds=bounds(),
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
            )
            
            if result.success:
                value = -result.fun
                if value > best_value:
                    best_value = value
                    best_result = result.x.reshape(N, 3)
        except Exception:
            pass
    
    return best_result
