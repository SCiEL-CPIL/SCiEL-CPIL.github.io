"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    def validate_and_score(x, y, r, w, h, tol=1e-9):
        """Check if circles are valid and return sum of radii."""
        if np.any(r <= 0):
            return 0.0
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return 0.0
        return float(np.sum(r))

    def compute_radii_from_positions(x, y, w, h, shrink_factor=0.999):
        """Compute maximum possible radii given positions."""
        n = len(x)
        radii = np.zeros(n)
        for i in range(n):
            # Distance to walls
            min_dist = min(x[i], w - x[i], y[i], h - y[i])
            # Distance to other circles
            for j in range(n):
                if i != j:
                    d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    min_dist = min(min_dist, d / 2.0)
            radii[i] = min_dist * shrink_factor
        return radii

    def score_configuration(x, y, w, h, shrink_factor=0.999):
        """Score a configuration by sum of radii, checking validity."""
        r = compute_radii_from_positions(x, y, w, h, shrink_factor)
        if np.any(r <= 0):
            return 0.0, x, y, r
        # Check validity
        tol = 1e-9
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return 0.0, x, y, r
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return 0.0, x, y, r
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return 0.0, x, y, r
        return float(np.sum(r)), x, y, r

    def simulated_annealing_grid(w, h, n_grid_x, n_grid_y, max_iter=5000, 
                                  t_start=1.0, t_end=0.01, seed=42):
        """Simulated annealing on a discrete grid."""
        rng = np.random.RandomState(seed)
        
        # Create grid
        x_grid = np.linspace(0.02, w - 0.02, n_grid_x)
        y_grid = np.linspace(0.02, h - 0.02, n_grid_y)
        
        # Initialize positions randomly on grid
        x_idx = rng.randint(0, n_grid_x, N)
        y_idx = rng.randint(0, n_grid_y, N)
        
        # Convert to continuous coordinates
        x = x_grid[x_idx].copy()
        y = y_grid[y_idx].copy()
        
        # Compute initial score
        score, x, y, r = score_configuration(x, y, w, h)
        best_score = score
        best_x = x.copy()
        best_y = y.copy()
        best_r = r.copy()
        
        # Simulated annealing
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Cooling schedule
            t = t_start * (t_end / t_start) ** (it / max_iter)
            
            # Choose a random circle to move
            i = rng.randint(N)
            
            # Choose a move: shift by 1 grid cell in random direction
            dx = rng.randint(-1, 2)
            dy = rng.randint(-1, 2)
            
            new_xi = x_idx[i] + dx
            new_yi = y_idx[i] + dy
            
            # Check bounds
            if new_xi < 0 or new_xi >= n_grid_x or new_yi < 0 or new_yi >= n_grid_y:
                continue
            
            # Try the move
            old_xi = x[i]
            old_yi = y[i]
            x[i] = x_grid[new_xi]
            y[i] = y_grid[new_yi]
            
            new_score, _, _, _ = score_configuration(x, y, w, h)
            
            # Accept if better, or with probability exp((new-old)/t)
            delta = new_score - score
            if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-10)):
                score = new_score
                x_idx[i] = new_xi
                y_idx[i] = new_yi
                if score > best_score:
                    best_score = score
                    best_x = x.copy()
                    best_y = y.copy()
                    _, _, _, best_r = score_configuration(x, y, w, h)
            else:
                # Revert
                x[i] = old_xi
                y[i] = old_yi
        
        return best_score, best_x, best_y, best_r

    def refine_with_local_search(x, y, w, h, max_iter=1000, step_size=0.01):
        """Local search to refine positions."""
        rng = np.random.RandomState(123)
        score, x, y, r = score_configuration(x, y, w, h)
        best_score = score
        best_x = x.copy()
        best_y = y.copy()
        best_r = r.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            # Choose random circle and direction
            i = rng.randint(N)
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            
            new_xi = x[i] + dx
            new_yi = y[i] + dy
            
            # Check bounds
            if new_xi < 0.01 or new_xi > w - 0.01 or new_yi < 0.01 or new_yi > h - 0.01:
                continue
            
            old_xi = x[i]
            old_yi = y[i]
            x[i] = new_xi
            y[i] = new_yi
            
            new_score, _, _, _ = score_configuration(x, y, w, h)
            
            if new_score > score:
                score = new_score
                if score > best_score:
                    best_score = score
                    best_x = x.copy()
                    best_y = y.copy()
                    _, _, _, best_r = score_configuration(x, y, w, h)
            else:
                x[i] = old_xi
                y[i] = old_yi
        
        return best_score, best_x, best_y, best_r

    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 1.3, 1.7, 0.5, 1.1, 1.4, 1.8, 0.7, 2.2, 2.5]
    
    for ar_idx, ar in enumerate(aspect_ratios):
        if time.time() > deadline - 10:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        # Grid resolution based on aspect ratio
        n_grid_x = max(20, int(40 * np.sqrt(w / (w + h))))
        n_grid_y = max(20, int(40 * np.sqrt(h / (w + h))))
        
        # Run simulated annealing with different seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            sa_score, sa_x, sa_y, sa_r = simulated_annealing_grid(
                w, h, n_grid_x, n_grid_y, max_iter=3000, 
                t_start=1.0, t_end=0.01, seed=42 + seed
            )
            
            # Refine with local search
            if time.time() > deadline - 3:
                score = sa_score
                x, y, r = sa_x, sa_y, sa_r
            else:
                score, x, y, r = refine_with_local_search(sa_x, sa_y, w, h, max_iter=500)
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r])

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
