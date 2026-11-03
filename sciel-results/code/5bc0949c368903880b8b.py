"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def calculate_score(points):
    """
    Calculate the score (radius) for a given set of points.
    Points: array of shape (N, 2)
    Returns: float, the maximum possible radius for these centers.
    """
    if len(points) != N:
        return -1.0
    
    # Calculate pairwise distances
    # Use broadcasting for speed
    x = points[:, 0]
    y = points[:, 1]
    
    # Squared distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    
    # Get upper triangle indices (i < j)
    i_upper, j_upper = np.triu_indices(N, k=1)
    min_d2 = d2[i_upper, j_upper].min()
    min_d = np.sqrt(min_d2)
    
    # Distance to boundary
    radii = np.hypot(x, y)
    max_radius = radii.max()
    
    # Score is the minimum of half the min distance and the distance to the boundary
    return min(min_d / 2.0, 1.0 - max_radius)

def hex_lattice_init(n_points):
    """
    Generate an initial hexagonal lattice configuration.
    """
    # Approximate spacing for N points in unit circle
    # Area of unit circle = pi. Area of hex cell ~ sqrt(3)/2 * s^2.
    # N * sqrt(3)/2 * s^2 ~ pi => s ~ sqrt(2*pi / (N*sqrt(3)))
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    points = []
    # Generate points in a grid and filter
    # We need enough points, so generate a large grid
    max_coord = 1.5
    step = s
    x = -max_coord
    while x < max_coord:
        y = -max_coord
        offset = (s / 2) if (int(x / step) % 2 != 0) else 0
        while y < max_coord:
            px = x
            py = y + offset
            if px**2 + py**2 <= (1.0 - s/2)**2:
                points.append([px, py])
            y += step * np.sqrt(3) / 2
        x += step
    
    points = np.array(points)
    if len(points) < N:
        # Fallback to random if lattice is too sparse (shouldn't happen)
        return np.random.uniform(-0.8, 0.8, (N, 2))
    
    # Sort by distance from center to pick the "densest" center
    dists = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(dists)
    return points[indices[:N]]

def ring_init(n_points):
    """
    Generate an initial concentric ring configuration.
    """
    points = []
    # 1 center
    points.append([0.0, 0.0])
    remaining = n_points - 1
    
    # Ring 1: 6
    # Ring 2: 12
    # Ring 3: 18
    # Ring 4: 14 (total 50)
    
    radii = [0.15, 0.35, 0.55, 0.75] # Initial guesses, SA will optimize
    
    counts = [6, 12, 18, 14]
    
    for r, count in zip(radii, counts):
        if remaining <= 0:
            break
        take = min(count, remaining)
        angles = np.linspace(0, 2*np.pi, take, endpoint=False)
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
        remaining -= take
    
    # If we have fewer than N, add random points near the boundary? 
    # Or just pad with random if logic fails.
    if len(points) < N:
        while len(points) < N:
             points.append([np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8)])
             
    return np.array(points[:N])

def construct():
    start_time = time.time()
    # Deadline: 110 seconds to be safe (limit is 120s)
    deadline = start_time + 110.0
    
    # Try multiple initializations and keep the best one
    candidates = []
    
    # 1. Hex Lattice
    p_hex = hex_lattice_init(N)
    s_hex = calculate_score(p_hex)
    candidates.append((s_hex, p_hex))
    
    # 2. Ring Init
    p_ring = ring_init(N)
    s_ring = calculate_score(p_ring)
    candidates.append((s_ring, p_ring))
    
    # 3. Random Init
    p_rand = np.random.uniform(-0.8, 0.8, (N, 2))
    s_rand = calculate_score(p_rand)
    candidates.append((s_rand, p_rand))
    
    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score = candidates[0][0]
    best_points = candidates[0][1].copy()
    
    # Simulated Annealing
    # We will run SA on the best candidate.
    # If time permits, we could restart SA on the second best, etc.
    
    current_points = best_points.copy()
    current_score = best_score
    
    T = 0.05
    T_min = 1e-6
    alpha = 0.995
    step_size = 0.05
    
    # Main SA Loop
    # We need to be careful with the time.
    # calculate_score is O(N^2) ~ 2500 ops. Very fast.
    # SA step is O(N).
    
    while time.time() < deadline - 5.0:
        # Perturb
        new_points = current_points + np.random.normal(0, step_size, (N, 2))
        
        # Keep points inside container? 
        # The score function handles it (returns small r if outside).
        # But it's better to project back or penalize.
        # Let's just let the score handle it.
        
        new_score = calculate_score(new_points)
        
        # Acceptance
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = new_points.copy()
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if T > 0:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current_points = new_points
                    current_score = new_score
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # If we hit min T, we can restart with a small random kick or just stop
            # Let's do a small kick to escape local minima
            if random.random() < 0.01:
                 kick_idx = np.random.randint(N)
                 current_points[kick_idx] += np.random.normal(0, 0.1, 2)
                 current_score = calculate_score(current_points)
        
        # Adaptive step size
        # If we are stuck (no improvement for a while), reduce step size?
        # Or increase if we are accepting too many bad moves?
        # Simple: decay step size slowly
        step_size *= 0.999
        if step_size < 1e-4:
            step_size = 1e-4

    # Final Polish
    # Use local optimization on the best points found
    # We can use scipy.optimize.minimize with Nelder-Mead
    # But Nelder-Mead on 100 variables is slow.
    # Let's do a few more SA steps with very small step size and low T
    T = 1e-4
    step_size = 1e-3
    polish_deadline = time.time() + 5.0
    
    while time.time() < polish_deadline:
        new_points = current_points + np.random.normal(0, step_size, (N, 2))
        new_score = calculate_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = new_points.copy()
        
        T *= 0.99
        step_size *= 0.99
        if step_size < 1e-5:
            break

    return best_points

# EVOLVE-BLOCK-END
