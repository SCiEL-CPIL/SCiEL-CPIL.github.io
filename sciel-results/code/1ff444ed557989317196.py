"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 21

# EVOLVE-BLOCK-START

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4 (i.e., 2*(w+h)=4 => w+h=2).
    Maximize sum of radii.
    Returns array of shape (21, 3) with columns [x, y, r].
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s kill

    # Strategy: Use a grid-like initial placement and then optimize
    # using gradient ascent / projected gradient on the radii and positions.
    #
    # Variables: x_i, y_i, r_i for i=1..21
    # Constraints:
    #   0 <= x_i - r_i, x_i + r_i <= w
    #   0 <= y_i - r_i, y_i + r_i <= h
    #   (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i != j
    #   w + h = 2, w > 0, h > 0
    #
    # Objective: maximize sum(r_i)
    #
    # We can eliminate w by setting w = 2 - h. But h is also a variable.
    # Alternatively, fix an aspect ratio and optimize, or let w and h vary.
    #
    # A simpler approach: Since the rectangle has fixed perimeter, we can
    # parameterize by w (then h = 2 - w). We'll optimize over w in (0, 2).
    #
    # For a given w, h = 2 - w. We need to place 21 circles.
    #
    # Let's use a simulated annealing / random restart local optimization approach.
    #
    # Better idea: Use scipy's minimize with constraints, or implement a custom
    # gradient-based optimizer with penalty methods.
    
    best_value = 0.0
    best_solution = None
    
    # Initial guess: grid layout
    # For 21 circles, a 5x5 grid has 25 cells, we use 21.
    # Or 4x6 = 24, use 21.
    # Let's try different grid configurations.
    
    def make_grid_solution(cols, rows, w, h):
        """Create a grid-based initial solution."""
        cells = []
        for i in range(cols):
            for j in range(rows):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * w / cols
                y = (j + 0.5) * h / rows
                r = 0.5 * min(w / cols, h / rows)
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    def evaluate_solution(sol, w, h):
        """Check if solution is valid and return sum of radii."""
        n = len(sol)
        if n != N:
            return 0.0
        
        # Check bounds
        for i in range(n):
            x, y, r = sol[i]
            if r <= 0:
                return 0.0
            if x - r < -1e-9 or x + r > w + 1e-9 or y - r < -1e-9 or y + r > h + 1e-9:
                return 0.0
        
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = sol[i][0] - sol[j][0]
                dy = sol[i][1] - sol[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = sol[i][2] + sol[j][2]
                if dist < min_dist - 1e-9:
                    return 0.0
        
        return np.sum(sol[:, 2])
    
    def project_to_feasible(sol, w, h, tol=1e-6, max_iter=1000):
        """
        Project a solution to be feasible using iterative pairwise separation.
        Returns the projected solution and whether it converged.
        """
        sol = sol.copy()
        n = len(sol)
        
        for iteration in range(max_iter):
            if time.time() - start_time > deadline:
                break
            
            # Enforce bounds
            for i in range(n):
                x, y, r = sol[i]
                # Adjust radius to fit in bounds
                max_r = min(x, w - x, y, h - y)
                if r > max_r:
                    sol[i][2] = max(0, max_r)
                    r = sol[i][2]
            
            # Pairwise separation
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = sol[i][0] - sol[j][0]
                    dy = sol[i][1] - sol[j][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    min_dist = sol[i][2] + sol[j][2]
                    
                    if dist < min_dist:
                        # Push apart
                        if dist < 1e-12:
                            # Random perturbation
                            angle = np.random.uniform(0, 2 * np.pi)
                            dx = math.cos(angle)
                            dy = math.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        overlap = min_dist - dist
                        # Distribute the push based on radii
                        r_i = sol[i][2]
                        r_j = sol[j][2]
                        total_r = r_i + r_j
                        if total_r < 1e-12:
                            continue
                        
                        # Move each circle away by a proportion of its radius
                        push_i = overlap * r_i / total_r
                        push_j = overlap * r_j / total_r
                        
                        sol[i][0] += dx * push_i
                        sol[i][1] += dy * push_i
                        sol[j][0] -= dx * push_j
                        sol[j][1] -= dy * push_j
                        
                        changed = True
            
            # Re-enforce bounds after pushing
            for i in range(n):
                x, y, r = sol[i]
                max_r = min(x, w - x, y, h - y)
                if r > max_r:
                    sol[i][2] = max(0, max_r)
            
            if not changed:
                break
        
        return sol
    
    def optimize_for_aspect(w, h, init_sol, max_iter=5000, lr=0.01):
        """
        Optimize the solution for a fixed aspect ratio using gradient ascent
        with penalty for constraint violations.
        """
        sol = init_sol.copy()
        n = len(sol)
        
        for iteration in range(max_iter):
            if time.time() - start_time > deadline:
                break
            
            # Compute objective: sum of radii - penalty for violations
            penalty = 0.0
            grad = np.zeros_like(sol)
            
            # Gradient of sum of radii: just 1 for each radius
            for i in range(n):
                grad[i, 2] = 1.0
            
            # Penalty for overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    dx = sol[i][0] - sol[j][0]
                    dy = sol[i][1] - sol[j][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    min_dist = sol[i][2] + sol[j][2]
                    
                    if dist < min_dist:
                        violation = min_dist - dist
                        penalty += violation ** 2
                        if dist > 1e-12:
                            # Gradient of violation w.r.t. positions
                            grad_x = dx / dist
                            grad_y = dy / dist
                            # Push apart: increase dist
                            grad[i, 0] += 2 * violation * grad_x
                            grad[i, 1] += 2 * violation * grad_y
                            grad[j, 0] -= 2 * violation * grad_x
                            grad[j, 1] -= 2 * violation * grad_y
                        # Gradient w.r.t. radii
                        grad[i, 2] -= 2 * violation
                        grad[j, 2] -= 2 * violation
            
            # Penalty for bounds
            for i in range(n):
                x, y, r = sol[i]
                # Left bound
                if x - r < 0:
                    violation = - (x - r)
                    penalty += violation ** 2
                    grad[i, 0] += 2 * violation
                    grad[i, 2] -= 2 * violation
                # Right bound
                if x + r > w:
                    violation = x + r - w
                    penalty += violation ** 2
                    grad[i, 0] += 2 * violation
                    grad[i, 2] += 2 * violation
                # Bottom bound
                if y - r < 0:
                    violation = - (y - r)
                    penalty += violation ** 2
                    grad[i, 1] += 2 * violation
                    grad[i, 2] -= 2 * violation
                # Top bound
                if y + r > h:
                    violation = y + r - h
                    penalty += violation ** 2
                    grad[i, 1] += 2 * violation
                    grad[i, 2] += 2 * violation
            
            # Update
            sol += lr * grad
            
            # Project to feasible region (simple clipping)
            for i in range(n):
                # Ensure radius is positive
                sol[i, 2] = max(0.001, sol[i, 2])
                # Clip positions
                r = sol[i, 2]
                sol[i, 0] = np.clip(sol[i, 0], r, w - r)
                sol[i, 1] = np.clip(sol[i, 1], r, h - r)
            
            # Check convergence
            if iteration % 100 == 0:
                val = evaluate_solution(sol, w, h)
                if val > best_value:
                    best_value = val
                    best_solution = sol.copy()
        
        # Final check
        val = evaluate_solution(sol, w, h)
        if val > best_value:
            best_value = val
            best_solution = sol.copy()
        
        return best_value, best_solution
    
    # Try different aspect ratios and grid configurations
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    grid_configs = [(5, 5), (4, 6), (3, 7), (7, 3), (6, 4), (10, 3), (3, 10), (14, 2), (2, 14), (21, 1), (1, 21)]
    
    # Also try a more systematic approach: start with a good grid and optimize
    for w_ratio in aspect_ratios:
        if time.time() - start_time > deadline:
            break
        
        # w_ratio = w/h, so w = 2 * w_ratio / (1 + w_ratio), h = 2 / (1 + w_ratio)
        w = 2 * w_ratio / (1 + w_ratio)
        h = 2 / (1 + w_ratio)
        
        for cols, rows in grid_configs:
            if time.time() - start_time > deadline:
                break
            
            if cols * rows < N:
                continue
            
            init_sol = make_grid_solution(cols, rows, w, h)
            
            # Optimize
            val, sol = optimize_for_aspect(w, h, init_sol, max_iter=3000, lr=0.05)
            
            if val > best_value:
                best_value = val
                best_solution = sol.copy()
    
    # If we have a best solution, return it
    if best_solution is not None:
        return best_solution
    
    # Fallback: simple grid in a square
    side = math.sqrt(2)
    w = side
    h = side
    sol = make_grid_solution(5, 5, w, h)
    return sol

# EVOLVE-BLOCK-END
