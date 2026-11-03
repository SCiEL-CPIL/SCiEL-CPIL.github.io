"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Helper to calculate score
    def calc_score(centers):
        min_dist = 1.0
        # Pairwise
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
        # Walls
        for i in range(N):
            for k in range(2):
                d = 0.5 - abs(centers[i][k])
                if d < min_dist:
                    min_dist = d
        return min_dist / 2.0

    # Generate initial guesses
    guesses = []
    
    # 1. 7x7 Grid (49 pts) + 1 duplicate with jitter
    ticks = np.linspace(-0.5 + 0.5/7, 0.5 - 0.5/7, 7)
    grid = np.array([(x, y) for y in ticks for x in ticks])
    g1 = grid.copy()
    # Duplicate center (index 24)
    g1[24] = g1[24] + np.array([0.02, 0.02])
    guesses.append(g1)
    
    # 2. 7x7 Grid + 1 duplicate corner
    g2 = grid.copy()
    g2[0] = g2[0] + np.array([0.02, 0.02])
    guesses.append(g2)
    
    # 3. Jittered 7x7 Grid + 1 random
    g3 = grid.copy() + np.random.normal(0, 0.01, (49, 2))
    extra = np.random.uniform(-0.4, 0.4, (1, 2))
    guesses.append(np.vstack([g3, extra]))
    
    # 4. Hexagonal Lattice
    hex_pts = []
    s = 0.14
    for i in range(-6, 7):
        for j in range(-6, 7):
            x = i * s + (j % 2) * s / 2
            y = j * s * np.sqrt(3) / 2
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    if len(hex_pts) >= 50:
        idx = np.random.choice(len(hex_pts), 50, replace=False)
        guesses.append(hex_pts[idx])
    else:
        # Pad
        pad = np.random.uniform(-0.4, 0.4, (50 - len(hex_pts), 2))
        guesses.append(np.vstack([hex_pts, pad]))

    best_centers = None
    best_score = 0.0

    # Optimization function
    def optimize(init_centers):
        # Flatten centers and add r
        x0 = np.concatenate([init_centers.flatten(), [0.01]])
        
        # Constraints
        def constraint_fun(x):
            c = x[:100].reshape(50, 2)
            r = x[100]
            cons = []
            # Pairwise
            for i in range(50):
                for j in range(i + 1, 50):
                    d = np.linalg.norm(c[i] - c[j])
                    cons.append(d - 2 * r)
            # Walls
            for i in range(50):
                cons.append(0.5 - abs(c[i][0]) - r)
                cons.append(0.5 - abs(c[i][1]) - r)
            return np.array(cons)

        def constraint_jac(x):
            c = x[:100].reshape(50, 2)
            r = x[100]
            J = np.zeros((1325, 101))
            idx = 0
            # Pairwise
            for i in range(50):
                for j in range(i + 1, 50):
                    diff = c[i] - c[j]
                    d = np.linalg.norm(diff)
                    if d > 1e-10:
                        J[idx, 2*i] = diff[0] / d
                        J[idx, 2*i+1] = diff[1] / d
                        J[idx, 2*j] = -diff[0] / d
                        J[idx, 2*j+1] = -diff[1] / d
                    else:
                        J[idx, 2*i] = 1.0
                        J[idx, 2*j] = -1.0
                    J[idx, 100] = -2.0
                    idx += 1
            # Walls
            for i in range(50):
                J[idx, 2*i] = -np.sign(c[i][0])
                J[idx, 100] = -1.0
                idx += 1
                J[idx, 2*i+1] = -np.sign(c[i][1])
                J[idx, 100] = -1.0
                idx += 1
            return J

        constraints = [{'type': 'ineq', 'fun': constraint_fun, 'jac': constraint_jac}]
        
        def objective(x):
            return -x[100]
            
        # Bounds
        bounds = [(-0.5, 0.5)] * 100 + [(0.0, 0.5)]
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-9})
            if res.success or res.fun < 0: # res.fun is -r, so smaller is better
                return res.x[:100].reshape(50, 2)
        except Exception:
            pass
        return None

    for g in guesses:
        if time.time() > deadline - 10:
            break
        
        centers = optimize(g)
        if centers is not None:
            score = calc_score(centers)
            if score > best_score:
                best_score = score
                best_centers = centers

    if best_centers is None:
        # Fallback
        ticks = np.linspace(-0.5 + 0.5/7, 0.5 - 0.5/7, 7)
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
