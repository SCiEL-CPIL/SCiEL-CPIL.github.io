import numpy as np
import time

N = 50
DEADLINE_SECONDS = 115  # Safety margin before 120s limit

def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    if points.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points > 0.5) or np.any(points < -0.5):
        return 0.0
        
    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: cube is [-0.5, 0.5]^3
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _fast_score(points, current_r):
    """Fast approximate score: only check if radius can be maintained.
    Returns the actual minimum distance/2 if >= current_r, else a lower value."""
    if np.any(points > 0.5) or np.any(points < -0.5):
        return 0.0
    
    # Check wall distances first (cheap)
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    if min_wall < current_r * 0.9:
        return min_wall
    
    # Check pair distances using a spatial hash for speed
    # For now, just do full check but with early exit
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Early exit: if any pair is too close
    min_dist_sq = np.min(dists_sq)
    if min_dist_sq < (2 * current_r * 0.95) ** 2:
        return np.sqrt(min_dist_sq) / 2.0
    
    # Full minimum
    min_pair = np.sqrt(min_dist_sq) / 2.0
    return min(min_pair, min_wall)

def _soft_min_energy(points, beta=50.0):
    """Compute soft-min energy using log-sum-exp for smooth optimization.
    Lower energy = better packing."""
    # Pair distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Soft min of pair distances (want to maximize this)
    # E_pair = -log(sum(exp(-beta * d_i))) / beta
    exp_vals = np.exp(-beta * dists)
    log_sum = np.log(np.sum(exp_vals))
    soft_min_pairs = -log_sum / beta
    
    # Wall distances
    wall_dists = 0.5 - np.abs(points)
    exp_vals_w = np.exp(-beta * wall_dists)
    log_sum_w = np.log(np.sum(exp_vals_w))
    soft_min_walls = -log_sum_w / beta
    
    # Total energy: we want both to be large, so minimize negative
    return -(soft_min_pairs + soft_min_walls)

def _soft_min_grad(points, beta=50.0):
    """Gradient of the soft-min energy."""
    # Pair distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e10)
    
    # Soft min gradient for pairs
    exp_vals = np.exp(-beta * dists)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # normalized weights
    
    # d/dx_i of -log(sum(exp(-beta*d_ij)))/beta for each pair
    # = sum_j weights_j * (beta * d/dx_i (-d_ij)) / beta
    # = sum_j weights_j * d/dx_i d_ij
    # d/dx_i d_ij = (x_i - x_j) / d_ij
    
    grad_pairs = np.zeros_like(points)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = dists[i, j]
                if d > 1e-10:
                    grad_pairs[i] += weights[i, j] * (points[i] - points[j]) / d
                    grad_pairs[j] += weights[j, i] * (points[j] - points[i]) / d
    
    # Wall distances gradient
    wall_dists = 0.5 - np.abs(points)
    exp_vals_w = np.exp(-beta * wall_dists)
    sum_exp_w = np.sum(exp_vals_w)
    weights_w = exp_vals_w / sum_exp_w
    
    # d/dx_i of wall_dist_i = -sign(x_i) for each component
    grad_walls = np.zeros_like(points)
    for i in range(N):
        for k in range(3):
            sign = np.sign(points[i, k])
            if sign == 0:
                sign = 1
            grad_walls[i, k] -= weights_w[i, k] * sign
    
    # Total gradient (we want to minimize energy = maximize soft min)
    # dE/dx = -d(soft_min)/dx
    grad = -(grad_pairs + grad_walls)
    
    return grad

def _generate_bcc_init(n):
    """
    Generate an initial configuration based on a Body-Centered Cubic (BCC) lattice.
    BCC has 2 points per unit cell. For 50 points, use a 3x3x3 grid with some cells.
    """
    # Create a 3x3x3 BCC lattice
    # Corner points and body-center points
    points = []
    a = 1.0 / 2.0  # lattice parameter
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner
                points.append([(i - 1) * a, (j - 1) * a, (k - 1) * a])
                # Body center
                points.append([(i - 0.5) * a, (j - 0.5) * a, (k - 0.5) * a])
    
    points = np.array(points)
    
    # We have 54 points, need 50. Remove 4 corner points from the outer shell
    # to keep the packing more symmetric
    # Remove corners at (±0.5, ±0.5, ±0.5)
    remove_idx = []
    for idx, p in enumerate(points):
        if all(abs(abs(c) - 0.5) < 1e-6 for c in p):
            remove_idx.append(idx)
            if len(remove_idx) == 4:
                break
    
    if len(remove_idx) < 4:
        # If not enough corners, remove randomly
        remaining = [i for i in range(len(points)) if i not in remove_idx]
        extra = np.random.choice(remaining, size=4 - len(remove_idx), replace=False)
        remove_idx.extend(extra)
    
    keep_idx = [i for i in range(len(points)) if i not in remove_idx]
    points = points[keep_idx[:n]]
    
    # Center and scale to fit nicely in [-0.48, 0.48]
    points -= np.mean(points, axis=0)
    max_val = np.max(np.abs(points))
    if max_val > 0:
        points *= (0.48 / max_val)
    
    # Add small random perturbation to break symmetry
    points += np.random.randn(n, 3) * 0.003
    
    return np.clip(points, -0.5, 0.5)

def _optimize_sa(points, max_time):
    """
    Simulated Annealing to maximize the minimum radius.
    Uses soft-min energy for smoother landscape.
    """
    start_time = time.time()
    current_points = points.copy()
    current_energy = _soft_min_energy(current_points, beta=30.0)
    best_points = current_points.copy()
    best_score = _score(current_points)
    
    T = 0.015  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.9998
    step_size = 0.04
    beta = 30.0
    
    while time.time() - start_time < max_time:
        if T < T_min:
            T = 0.015
            step_size = 0.04
            beta = max(15.0, beta * 0.9)
        
        # Propose a move: move one random point
        idx = np.random.randint(N)
        delta = np.random.randn(3) * step_size
        new_points = current_points.copy()
        new_points[idx] += delta
        new_points = np.clip(new_points, -0.5, 0.5)
        
        new_energy = _soft_min_energy(new_points, beta=beta)
        
        # Acceptance based on energy difference
        if new_energy < current_energy:
            current_points = new_points
            current_energy = new_energy
            # Check actual score periodically
            if np.random.random() < 0.05:
                s = _score(current_points)
                if s > best_score:
                    best_score = s
                    best_points = current_points.copy()
        else:
            delta_e = current_energy - new_energy
            if delta_e < 1e-10:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_e / T)
            
            if np.random.random() < accept_prob:
                current_points = new_points
                current_energy = new_energy
        
        T *= cooling_rate
        step_size *= 0.9995
    
    # Final score check
    s = _score(current_points)
    if s > best_score:
        best_score = s
        best_points = current_points.copy()
    
    return best_points, best_score

def _polish(points, max_time):
    """
    Local optimization using gradient ascent on the minimum distance.
    Uses the closest pair and closest wall to guide movement.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    lr = 0.005
    decay = 0.998
    
    while time.time() - start_time < max_time:
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        
        wall_dists = 0.5 - np.abs(current_points)
        min_wall_idx = np.argmin(np.min(wall_dists, axis=1))
        min_wall_val = np.min(wall_dists[min_wall_idx])
        
        min_pair_val = dists[i, j] / 2.0
        
        # Move to increase the minimum
        if min_pair_val < min_wall_val:
            # Move i and j apart
            direction = current_points[i] - current_points[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                current_points[i] += lr * direction
                current_points[j] -= lr * direction
        else:
            # Move the point with closest wall away from the wall
            close_wall = np.argmin(np.abs(current_points[min_wall_idx]))
            sign = np.sign(current_points[min_wall_idx, close_wall])
            if sign == 0:
                sign = 1
            current_points[min_wall_idx, close_wall] += lr * sign
        
        current_points = np.clip(current_points, -0.5, 0.5)
        
        new_score = _score(current_points)
        if new_score > best_score:
            best_score = new_score
            best_points = current_points.copy()
            
        lr *= decay
        if lr < 1e-7:
            break
            
    return best_points, best_score

def _gradient_polish(points, max_time):
    """
    Gradient-based polish using soft-min gradient for smoother convergence.
    """
    start_time = time.time()
    current_points = points.copy()
    best_score = _score(current_points)
    best_points = current_points.copy()
    
    lr = 0.003
    beta = 40.0
    decay = 0.997
    
    while time.time() - start_time < max_time:
        grad = _soft_min_grad(current_points, beta=beta)
        
        # Gradient ascent (we want to maximize soft min = minimize energy)
        current_points -= lr * grad
        current_points = np.clip(current_points, -0.5, 0.5)
        
        new_score = _score(current_points)
        if new_score > best_score:
            best_score = new_score
            best_points = current_points.copy()
        
        lr *= decay
        beta = max(20.0, beta * 0.995)
        
        if lr < 1e-7:
            break
    
    return best_points, best_score

def construct():
    """Construct the sphere centers."""
    start_time = time.time()
    best_points = None
    best_score = 0.0
    
    # Run multiple restarts with time budget
    restart_time_budget = 8.0  # seconds per restart cycle
    
    while time.time() - start_time < DEADLINE_SECONDS:
        remaining = DEADLINE_SECONDS - (time.time() - start_time)
        if remaining < restart_time_budget * 0.5:
            break
        
        # Initialize with BCC
        points = _generate_bcc_init(N)
        
        # Optimize: SA then gradient polish then local polish
        opt_points, opt_score = _optimize_sa(points, min(restart_time_budget * 0.6, remaining * 0.5))
        opt_points, opt_score = _gradient_polish(opt_points, min(restart_time_budget * 0.25, remaining * 0.3))
        opt_points, opt_score = _polish(opt_points, min(restart_time_budget * 0.15, remaining * 0.2))
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
    
    if best_points is None:
        best_points = _generate_bcc_init(N)
        
    return best_points

if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
