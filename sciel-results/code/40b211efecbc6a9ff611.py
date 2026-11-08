import numpy as np
import time
from scipy.optimize import differential_evolution

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Bounds for x, y, r
    # x, y in [0, 1], r in [1e-6, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N

    def objective(v):
        """
        Objective function for Differential Evolution.
        Returns the negative sum of radii plus penalties for violations.
        DE minimizes this function.
        """
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]

        # 1. Base cost: Minimize negative sum of radii (Maximize sum of radii)
        cost = -np.sum(r)

        # 2. Boundary penalties
        # We use a large penalty for any violation to guide the optimizer
        # away from invalid states, but DE is stochastic so it might still land there.
        # The repair step will handle the final fix.
        
        # Left: x >= r  => x - r >= 0
        # Right: x <= 1 - r => x + r <= 1
        # Bottom: y >= r
        # Top: y <= 1 - r
        
        v_left = np.maximum(0, r - x)
        v_right = np.maximum(0, x + r - 1)
        v_bottom = np.maximum(0, r - y)
        v_top = np.maximum(0, y + r - 1)
        
        boundary_penalty = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)

        # 3. Overlap penalties
        # dist(i,j) >= r_i + r_j
        # Vectorized calculation
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        req = r[:, None] + r[None, :]
        
        # Only consider i < j
        i_idx, j_idx = np.triu_indices(N, k=1)
        viol = np.maximum(0, req[i_idx, j_idx] - dist[i_idx, j_idx])
        
        overlap_penalty = np.sum(viol**2)

        # Total penalty weight
        # DE needs a strong signal to avoid overlaps, but we also want to maximize radius.
        # A weight of 100-1000 is typical.
        penalty_weight = 500.0
        cost += penalty_weight * (boundary_penalty + overlap_penalty)

        return cost

    def repair_and_project(sol):
        """
        Takes a candidate solution (N, 3) and attempts to make it valid.
        If it fails, returns None.
        """
        x = sol[:, 0].copy()
        y = sol[:, 1].copy()
        r = sol[:, 2].copy()

        # 1. Clip radii and positions to rough bounds
        r = np.clip(r, 1e-6, 0.5)
        
        # 2. Iterative repair loop
        # We try to resolve overlaps by moving circles apart or shrinking them.
        # Since we want to maximize sum of radii, we prefer moving over shrinking if possible,
        # but for a quick valid solution, shrinking is safer.
        # Let's use a simple relaxation: if overlapping, push apart slightly.
        # If still overlapping, shrink.
        
        max_iter = 50
        tol = 1e-9
        
        for _ in range(max_iter):
            # Check boundaries
            # If x < r, set x = r. If x > 1-r, set x = 1-r.
            # This might cause overlaps, which we resolve in the next step.
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
            # Check overlaps
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            req = r[:, None] + r[None, :]
            
            # Find violations
            # We need to fix pairs where dist < req
            # Simple approach: for each violating pair, push them apart along the vector.
            # This is O(N^2) per iteration, which is fine for N=26.
            
            max_violation = 0.0
            any_violation = False
            
            for i in range(N):
                for j in range(i + 1, N):
                    d = dist[i, j]
                    r_sum = req[i, j]
                    if d < r_sum - tol:
                        any_violation = True
                        violation = r_sum - d
                        max_violation = max(max_violation, violation)
                        
                        # Direction from i to j
                        if d > 1e-12:
                            dir_x = (x[j] - x[i]) / d
                            dir_y = (y[j] - y[i]) / d
                        else:
                            # Random direction if coincident
                            angle = np.random.uniform(0, 2 * np.pi)
                            dir_x = np.cos(angle)
                            dir_y = np.sin(angle)
                        
                        # Push apart by half the violation each
                        push = violation * 0.5 + 1e-6
                        
                        x[i] -= dir_x * push
                        y[i] -= dir_y * push
                        x[j] += dir_x * push
                        y[j] += dir_y * push
            
            if not any_violation:
                break
            
            # If we are stuck (oscillating or not converging), shrink radii slightly
            # to guarantee convergence to a valid state.
            # This reduces the score but ensures validity.
            if _ > 20:
                r *= 0.99
                x = np.clip(x, r, 1 - r)
                y = np.clip(y, r, 1 - r)

        # Final validation check
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        req = r[:, None] + r[None, :]
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        if np.any(dist[i_idx, j_idx] < req[i_idx, j_idx] - 1e-8):
            return None
        if np.any(x - r < -1e-8) or np.any(x + r > 1 + 1e-8):
            return None
        if np.any(y - r < -1e-8) or np.any(y + r > 1 + 1e-8):
            return None
        if np.any(r <= 0):
            return None
            
        return np.column_stack((x, y, r))

    def get_initial_guess():
        # Random initialization within bounds
        # DE handles its own population, but we can provide a good start if needed.
        # However, DE is global, so random start is fine.
        # We just need to return a valid vector for DE if we were to use it, 
        # but DE generates its own population.
        # We'll just run DE.
        pass

    best_score = -np.inf
    best_solution = None

    # Run Differential Evolution
    # maxiter: number of generations. 100 is reasonable for 120s.
    # popsize: population size. 10 means 10 * 26 * 3 = 780 evaluations per gen? No, popsize * N.
    # DE with popsize=15 means 15 * 26 = 390 individuals.
    # Each evaluation is O(N^2) = 676 ops. Very fast.
    # We can afford many iterations.
    
    try:
        # DE is stochastic. We run it once with a decent budget.
        # If it times out, we catch it.
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            seed=42,
            tol=1e-9,
            polish=False  # Disable L-BFGS-B polish to save time, we have repair
        )
        
        if res.success or res.fun < np.inf:
            v_sol = res.x
            sol = np.column_stack((v_sol[0::3], v_sol[1::3], v_sol[2::3]))
            
            # Repair the solution
            repaired_sol = repair_and_project(sol)
            
            if repaired_sol is not None:
                score = np.sum(repaired_sol[:, 2])
                if score > best_score:
                    best_score = score
                    best_solution = repaired_sol

    except Exception:
        pass

    # If DE didn't find a good solution or timed out, try a few random restarts with repair
    # This acts as a fallback and potentially a better search if DE gets stuck.
    for i in range(10):
        if time.time() > deadline:
            break
            
        # Random valid-ish start
        # Grid based start is often better than pure random for packing
        side = 5
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        
        x_coords = []
        y_coords = []
        r_coords = []
        
        for idx in range(N):
            i = idx % side
            j = idx // side
            x = xs[i] + np.random.uniform(-0.02, 0.02)
            y = ys[j] + np.random.uniform(-0.02, 0.02)
            r = 0.5 / side * 0.8
            x_coords.append(x)
            y_coords.append(y)
            r_coords.append(r)
            
        sol = np.column_stack((np.array(x_coords), np.array(y_coords), np.array(r_coords)))
        
        # Try to improve this specific configuration using local optimization?
        # Or just repair it?
        # Repairing a grid start usually yields the seed score (1.0).
        # To improve, we need to allow radii to grow.
        
        # Let's use a simple gradient ascent on radii with position relaxation?
        # No, let's stick to the DE result primarily.
        # But we can try a simple "grow and fix" loop on this random grid.
        
        current_sol = sol.copy()
        for _ in range(20):
            # Try to increase radii slightly
            current_sol[:, 2] *= 1.05
            # Repair
            repaired = repair_and_project(current_sol)
            if repaired is not None:
                current_sol = repaired
            else:
                # Revert growth
                current_sol[:, 2] /= 1.05
                break
        
        score = np.sum(current_sol[:, 2])
        if score > best_score:
            best_score = score
            best_solution = current_sol

    # Fallback: Trivial valid solution
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N])
        
    return best_solution
