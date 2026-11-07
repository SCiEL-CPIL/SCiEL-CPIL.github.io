"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Use a softer distance penalty to allow ants to consider slightly farther nodes
    # with high prize values, potentially finding better routes within max_len.
    # The original used 1/(d+eps) which strongly penalizes distance.
    # Using 1/(d+eps)^0.5 or similar might help.
    
    # Let's try: prize / (distance + 1e-3)^0.5
    # Or maybe just prize * exp(-distance)
    
    # Let's stick close to the original but tune the epsilon or exponent.
    # The original score was 1.0. The child with alpha=2, beta=0.5 scored 0.956 (worse).
    # This suggests that amplifying prize (alpha>1) or softening distance (beta<1) might not be the right direction, 
    # OR that the specific combination was bad.
    
    # Let's try a different approach: 
    # Heuristic = prize / (distance + 1e-9) is the seed.
    # Maybe we should emphasize nodes that are closer to the start or have better return potential?
    # But the heuristic is static.
    
    # Let's try: (prize + 1e-9) / (distance_matrix + 1e-9) ** 0.8
    # Softening the distance penalty slightly.
    
    eps = 1e-9
    alpha = 1.0
    beta = 0.8
    
    prize_factor = np.power(prizes + eps, alpha)
    dist_factor = np.power(distance_matrix + eps, beta)
    
    heuristic = prize_factor[None, :] / dist_factor
    
    # Ensure positive finite values
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
