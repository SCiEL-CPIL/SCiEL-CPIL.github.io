"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    """Pack N circles in unit square, maximizing sum of radii."""
    deadline = time.time() + 145  # safety margin before 150s limit
    
    def compute_score(circles):
        """Compute the objective: sum of radii."""
        return np.sum(circles[:, 2])
    
    def check_validity(circles, tol=1e-8):
        """Check if circles are valid (inside square, non-overlapping)."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dist < ri + rj - tol:
                    return False
        return True

    def build_contact_graph(positions, k=3):
        """
        Build a contact graph where each node is connected to its k nearest neighbors.
        Returns a list of (i, j) pairs.
        """
        n = len(positions)
        # Compute pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                dists[i, j] = d
                dists[j, i] = d
        
        edges = set()
        for i in range(n):
            # Get indices of k nearest neighbors (excluding self)
            neighbors = np.argsort(dists[i])[:k]
            for j in neighbors:
                if i != j:
                    edges.add((min(i, j), max(i, j)))
        
        return list(edges)

    def soft_min_gap(positions, radii, edges, beta=50.0):
        """
        Compute a differentiable approximation of the minimum gap between circles.
        gap_ij = dist_ij - (r_i + r_j)
        soft_min = -1/beta * log(sum(exp(-beta * gap_ij)))
        """
        gaps = []
        for i, j in edges:
            dist = np.linalg.norm(positions[i] - positions[j])
            gap = dist - (radii[i] + radii[j])
            gaps.append(gap)
        
        if not gaps:
            return 0.0, np.zeros(2 * len(positions))
        
        gaps = np.array(gaps)
        # Avoid overflow/underflow in log-sum-exp
        min_gap = np.min(gaps)
        log_sum_exp = np.log(np.sum(np.exp(-beta * (gaps - min_gap)))) + (-beta * min_gap)
        soft_min = -log_sum_exp / beta
        
        # Gradient with respect to positions
        grad = np.zeros(2 * len(positions))
        exp_terms = np.exp(-beta * (gaps - min_gap))
        sum_exp = np.sum(exp_terms)
        
        for idx, (i, j) in enumerate(edges):
            weight = -beta * exp_terms[idx] / sum_exp
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.linalg.norm([dx, dy])
            if dist > 1e-10:
                gx = dx / dist
                gy = dy / dist
            else:
                gx, gy = 1.0, 0.0
            
            grad[2*i] += weight * gx
            grad[2*i+1] += weight * gy
            grad[2*j] -= weight * gx
            grad[2*j+1] -= weight * gy
        
        return soft_min, grad

    def objective_and_grad(x, radii, edges, beta):
        """
        Objective for L-BFGS-B: minimize -soft_min_gap (i.e., maximize min gap).
        x is flattened positions [x0, y0, x1, y1, ...]
        """
        n = len(radii)
        positions = x.reshape(n, 2)
        
        # Add small penalty for going out of bounds to keep things stable
        # But we rely on bounds in L-BFGS-B primarily
        gap, gap_grad = soft_min_gap(positions, radii, edges, beta)
        
        # We want to maximize gap, so minimize -gap
        obj = -gap
        grad = -gap_grad
        
        return obj, grad

    def optimize_positions_for_radii(radii, initial_positions=None, beta=50.0, max_iter=200):
        """
        Use L-BFGS-B to find positions for given radii that maximize the minimum gap.
        Returns (positions, success, final_gap)
        """
        n = len(radii)
        
        if initial_positions is None:
            # Simple grid initialization
            side = int(np.ceil(np.sqrt(n)))
            positions = np.zeros((n, 2))
            for i in range(n):
                row = i // side
                col = i % side
                positions[i, 0] = (col + 0.5) / side
                positions[i, 1] = (row + 0.5) / side
        else:
            positions = initial_positions.copy()
        
        # Ensure initial positions are within [r_i, 1-r_i]
        for i in range(n):
            positions[i, 0] = np.clip(positions[i, 0], radii[i], 1 - radii[i])
            positions[i, 1] = np.clip(positions[i, 1], radii[i], 1 - radii[i])
        
        x0 = positions.flatten()
        
        # Bounds: each x_i in [r_i, 1-r_i], each y_i in [r_i, 1-r_i]
        bounds = []
        for i in range(n):
            bounds.append((radii[i], 1 - radii[i]))
            bounds.append((radii[i], 1 - radii[i]))
        
        # Build contact graph based on current positions
        edges = build_contact_graph(positions, k=4)
        
        # Define wrapper for minimize
        def fun(x):
            val, g = objective_and_grad(x, radii, edges, beta)
            return val, g
        
        res = minimize(
            fun, x0, method='L-BFGS-B', jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        final_positions = res.x.reshape(n, 2)
        # Re-check validity and compute actual min gap
        min_actual_gap = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(final_positions[i] - final_positions[j])
                gap = d - (radii[i] + radii[j])
                if gap < min_actual_gap:
                    min_actual_gap = gap
        
        return final_positions, res.success, min_actual_gap

    def grow_radii_iteratively(initial_radii, max_steps=100, growth_factor=1.01):
        """
        Iteratively increase radii and re-optimize positions.
        """
        current_radii = initial_radii.copy()
        current_positions = None
        best_radii = current_radii.copy()
        best_positions = None
        best_score = np.sum(current_radii)
        
        # Initial optimization to get a valid starting point
        if current_positions is None:
            current_positions, success, gap = optimize_positions_for_radii(current_radii, beta=50.0, max_iter=100)
            if not check_validity(np.column_stack([current_positions, current_radii]), tol=1e-6):
                # Fallback to simple grid if optimization fails badly
                side = int(np.ceil(np.sqrt(len(current_radii))))
                current_positions = np.zeros((len(current_radii), 2))
                for i in range(len(current_radii)):
                    row = i // side
                    col = i % side
                    current_positions[i, 0] = (col + 0.5) / side
                    current_positions[i, 1] = (row + 0.5) / side
                    # Clip to bounds
                    current_positions[i, 0] = np.clip(current_positions[i, 0], current_radii[i], 1 - current_radii[i])
                    current_positions[i, 1] = np.clip(current_positions[i, 1], current_radii[i], 1 - current_radii[i])
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            # Try to grow radii
            new_radii = current_radii * growth_factor
            if np.any(new_radii > 0.5):
                break
            
            # Optimize positions for new radii
            new_positions, success, gap = optimize_positions_for_radii(
                new_radii, initial_positions=current_positions, beta=50.0, max_iter=50
            )
            
            # Check validity
            circles = np.column_stack([new_positions, new_radii])
            if check_validity(circles, tol=1e-6):
                score = np.sum(new_radii)
                if score > best_score:
                    best_score = score
                    best_radii = new_radii.copy()
                    best_positions = new_positions.copy()
                current_radii = new_radii.copy()
                current_positions = new_positions.copy()
            else:
                # Try a smaller growth factor if this fails
                # Binary search for the max growth factor that works
                lo, hi = 1.0, growth_factor
                for _ in range(10):
                    mid = (lo + hi) / 2
                    test_radii = current_radii * mid
                    test_positions, _, _ = optimize_positions_for_radii(
                        test_radii, initial_positions=current_positions, beta=50.0, max_iter=30
                    )
                    test_circles = np.column_stack([test_positions, test_radii])
                    if check_validity(test_circles, tol=1e-6):
                        lo = mid
                    else:
                        hi = mid
                
                if lo > 1.0 + 1e-6:
                    new_radii = current_radii * lo
                    new_positions, _, _ = optimize_positions_for_radii(
                        new_radii, initial_positions=current_positions, beta=50.0, max_iter=30
                    )
                    circles = np.column_stack([new_positions, new_radii])
                    if check_validity(circles, tol=1e-6):
                        score = np.sum(new_radii)
                        if score > best_score:
                            best_score = score
                            best_radii = new_radii.copy()
                            best_positions = new_positions.copy()
                        current_radii = new_radii.copy()
                        current_positions = new_positions.copy()
                else:
                    break
        
        return best_positions, best_radii

    def multi_start_strategy():
        """
        Try multiple different initial radius distributions and return the best.
        """
        best_positions = None
        best_radii = None
        best_score = -np.inf
        
        strategies = [
            "uniform_small",
            "uniform_medium",
            "one_large",
            "random"
        ]
        
        for strat in strategies:
            if time.time() > deadline - 15:
                break
            
            radii = np.zeros(N)
            if strat == "uniform_small":
                radii[:] = 0.05
            elif strat == "uniform_medium":
                radii[:] = 0.08
            elif strat == "one_large":
                radii[0] = 0.2
                radii[1:] = 0.05
            elif strat == "random":
                radii[:] = np.random.uniform(0.04, 0.12, N)
            
            # Cap radii to ensure they can fit
            radii = np.clip(radii, 0.01, 0.45)
            
            positions, final_radii = grow_radii_iteratively(radii, max_steps=50, growth_factor=1.015)
            
            if positions is not None:
                circles = np.column_stack([positions, final_radii])
                if check_validity(circles, tol=1e-6):
                    score = np.sum(final_radii)
                    if score > best_score:
                        best_score = score
                        best_positions = positions
                        best_radii = final_radii
        
        return best_positions, best_radii

    # Main execution
    positions, radii = multi_start_strategy()
    
    if positions is None or radii is None:
        # Fallback to simple grid if everything failed
        side = int(np.ceil(np.sqrt(N)))
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            positions[i, 0] = (col + 0.5) / side
            positions[i, 1] = (row + 0.5) / side
            radii[i] = 0.5 / side - 1e-4
        # Clip positions to bounds
        for i in range(N):
            positions[i, 0] = np.clip(positions[i, 0], radii[i], 1 - radii[i])
            positions[i, 1] = np.clip(positions[i, 1], radii[i], 1 - radii[i])
    
    result = np.column_stack([positions, radii])
    
    # Final verification
    if not check_validity(result, tol=1e-6):
        # Aggressive fallback
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) / side
            result[i, 1] = (row + 0.5) / side
            result[i, 2] = 0.5 / side - 1e-4
    
    return result
# EVOLVE-BLOCK-END
