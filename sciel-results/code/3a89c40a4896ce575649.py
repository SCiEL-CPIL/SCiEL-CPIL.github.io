"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

    # Initial positions: a rough hexagonal cluster
    # 1 center, 6 around, 4 more
    centers = []
    centers.append([0.0, 0.0])
    for i in range(6):
        angle = i * np.pi / 3
        centers.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
    for i in range(4):
        angle = (i * 2 + 1) * np.pi / 6
        centers.append([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)])
    
    current_positions = np.array(centers, dtype=np.float64)
    
    # Precompute angles for hexagon constraints
    angles = np.arange(6) * np.pi / 3
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    # Penalty weight for overlap
    mu = 10.0
    
    def calc_cost(pos):
        # Calculate L_min
        proj = pos[:, 0:1] * cos_angles[None, :] + pos[:, 1:2] * sin_angles[None, :]
        max_proj = np.max(proj)
        L_min = (2.0 / np.sqrt(3)) * max_proj + 1.0
        
        # Calculate overlap penalty
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        violations = np.maximum(0, np.sqrt(3) - dists)
        overlap_penalty = np.sum(violations**2)
        
        return L_min + mu * overlap_penalty

    current_cost = calc_cost(current_positions)
    best_positions = current_positions.copy()
    best_cost = current_cost
    
    T = 1.0
    cooling = 0.995
    step_size = 0.5
    
    while time.time() - start_time < time_limit and T > 0.01:
        new_positions = current_positions + np.random.normal(0, step_size, size=current_positions.shape)
        new_cost = calc_cost(new_positions)
        
        if new_cost < current_cost or np.random.random() < np.exp((current_cost - new_cost) / T):
            current_positions = new_positions
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = current_positions.copy()
        
        T *= cooling
        step_size *= 0.999

    # Calculate L for best_positions
    proj = best_positions[:, 0:1] * cos_angles[None, :] + best_positions[:, 1:2] * sin_angles[None, :]
    max_proj = np.max(proj)
    L = (2.0 / np.sqrt(3)) * max_proj + 1.0
    
    # Check overlaps
    diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist < np.sqrt(3) - 1e-9:
        # Fallback to a known valid configuration if SA failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for p in best_positions:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
