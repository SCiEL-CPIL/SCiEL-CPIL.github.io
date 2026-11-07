"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = 300.0  # seconds

    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    dist = np.asarray(distance_matrix, dtype=np.float64)

    # Ensure diagonal is large (self-transitions not useful)
    d = dist.copy()
    np.fill_diagonal(d, np.inf)

    # Prize vector, node 0 has prize 0
    p = prizes.copy()
    p[0] = 0.0

    # Distance to start (for return constraint awareness)
    d_to_start = d[:, 0].copy()
    d_to_start[0] = 0.0

    # Dynamic Potential heuristic:
    # Entry (i, j) = prize[j] * normalized_slack_after_visiting_j
    # where slack = max_len - (d[i,0] + d[i,j] + d[j,0])
    # This encodes future feasibility directly into the edge weight.

    eps = 1e-9

    # Compute the slack matrix: for each (i, j), how much length remains
    # after going from i to j and then back to start
    # slack[i,j] = max_len - d[i,0] - d[i,j] - d[j,0]
    slack = max_len - d_to_start[:, None] - d - d_to_start[None, :]

    # Normalize slack to [0, 1] range for better conditioning
    # Only consider positive slack (feasible moves)
    slack = np.maximum(slack, 0.0)
    
    # Scale: divide by max_len to normalize
    slack_norm = slack / max_len

    # Heuristic: prize[j] * (slack_norm[i,j])^beta
    # beta > 1 makes the heuristic more selective about tight routes
    beta = 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        base = p[None, :] * (slack_norm + eps) ** beta

    # Replace non-finite with small value
    base = np.where(np.isfinite(base), base, 1e-9)

    # Ensure all values are positive
    base = np.maximum(base, 1e-9)

    # Add a secondary term: direct prize attractiveness with distance decay
    # This helps when slack is similar across options
    alpha = 1.2
    with np.errstate(divide='ignore', invalid='ignore'):
        direct = p[None, :] / (d + eps) ** alpha

    direct = np.where(np.isfinite(direct), direct, 1e-9)
    direct = np.maximum(direct, 1e-9)

    # Combine: primary is dynamic potential, secondary is direct attractiveness
    # Weight them to balance feasibility vs. immediate prize
    w1 = 0.7  # weight for dynamic potential
    w2 = 0.3  # weight for direct attractiveness

    heuristic = w1 * base + w2 * direct

    # Ensure diagonal is minimal (self-loops not attractive)
    np.fill_diagonal(heuristic, 1e-9)

    # Final safety: ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Return whatever we have; it's already computed
        pass

    return heuristic
# EVOLVE-BLOCK-END
