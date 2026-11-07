"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    # Convert to float array to ensure numerical stability
    d = np.array(distance_matrix, dtype=np.float64)
    
    # Identify the diagonal (distance 0) to handle separately
    n = d.shape[0]
    
    # Create a safe distance matrix for off-diagonal calculations
    # We replace 0 with a small epsilon to avoid division by zero
    # However, for the heuristic, we want the diagonal to be very low attractiveness
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    # So if we return 0 or negative, they become 1e-9.
    # If we return inf, it might be invalid? "non-finite values ... make the whole result invalid"
    # So we must avoid inf.
    
    # Strategy:
    # 1. Compute a strong heuristic based on inverse distance.
    # 2. The current best (1.3436) used alpha=3.0.
    # 3. Inspiration 3 (score 1.2501) used normalization: (d_min / d)^2.
    # 4. Inspiration 2 (score 1.2501) used alpha=2.0 without normalization.
    #
    # Let's combine: Use a sharp inverse power (like alpha=2 or 3) but normalize
    # so that the maximum heuristic value is 1.0. This prevents the heuristic
    # from overwhelming the pheromone (which starts at 1.0) in the early iterations.
    # In the current program (alpha=3, no norm), max eta is 1/(d_min^3).
    # If d_min ~ 0.05 (typical for 50 pts in unit square), 1/0.05^3 = 8000.
    # This makes the heuristic dominate completely, essentially greedy NN.
    # Normalizing to max 1.0 balances pheromone and heuristic better,
    # allowing the colony to explore slightly more while still favoring short edges.
    
    # Find min non-zero distance
    mask = ~np.eye(n, dtype=bool)
    non_zero_dists = d[mask]
    
    # Filter out any potential numerical noise zeros if present
    non_zero_dists = non_zero_dists[non_zero_dists > 1e-10]
    
    if len(non_zero_dists) == 0:
        # Fallback
        return np.ones_like(d)
        
    d_min = np.min(non_zero_dists)
    
    # Create safe distance matrix
    # Diagonal will be treated separately
    d_safe = d.copy()
    d_safe[d_safe < 1e-10] = d_min  # Prevent div by zero
    
    # Use alpha=2.0 as a balanced sharpness (Inspiration 2 scored well with this)
    # Normalization: eta_ij = (d_min / d_ij)^alpha
    alpha = 2.0
    eta = (d_min / d_safe) ** alpha
    
    # Ensure the diagonal is low (1e-9) as per problem statement for non-positive/zero
    # The diagonal in eta is (d_min/d_min)^alpha = 1.0.
    # We want the diagonal to be unattractive. The ACO loop masks visited nodes,
    # but the diagonal is the node itself. Usually, the probability calculation
    # excludes visited nodes. Node i is visited when at node i.
    # So diagonal values don't matter much, but let's set them to 1e-9 to be safe
    # and consistent with "non-positive become 1e-9" if we were to return 0.
    # However, 1.0 is positive. Let's explicitly set diagonal to a small value.
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure no NaNs or Infs
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
