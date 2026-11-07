"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix based on Clarke-Worth savings.
    
    The standard ACO heuristic is 1/distance. This approach uses the 'savings'
    metric: s(i, j) = d(i, depot) + d(depot, j) - d(i, j).
    If s(i, j) is large, connecting i and j directly is much better than
    returning to the depot between them. We map this savings value to a
    heuristic attractiveness.
    
    To ensure the ACO still prefers shorter edges generally (as savings can be
    noisy or negative for distant nodes), we combine savings with a distance
    penalty.
    
    Heuristic[i, j] = exp(savings[i, j] / scale) if savings[i, j] > 0 else 1.0
    But to keep it simple and robust:
    We define attractiveness as proportional to the savings, normalized.
    
    However, the ACO uses heuristic[i, j] * pheromone[i, j].
    If we set heuristic[i, j] high for high savings, the ant is more likely
    to pick j after i, forming a cluster.
    
    Let's calculate savings for all pairs i, j (i != j).
    s[i, j] = dist[i, 0] + dist[0, j] - dist[i, j]
    
    We want larger values for more attractive edges.
    We will use: heuristic[i, j] = 1 + max(0, s[i, j]) / max_savings
    This ensures all values are positive and >= 1.
    
    Note: The problem states "Heuristic entries that are not positive become 1e-9".
    Our values are strictly positive.
    
    We also need to consider that the ACO selects the *next* node.
    If we are at node i, we look at heuristic[i, j] for candidate j.
    High heuristic[i, j] means "It is good to go from i to j".
    This aligns with savings if the savings are calculated for the pair (i, j).
    
    One nuance: Savings are symmetric. d(i, 0) + d(0, j) - d(i, j) is same as d(j, 0) + d(0, i) - d(j, i).
    So the matrix will be symmetric.
    """
    n = len(demands)
    # Ensure distance_matrix is numpy array
    D = np.asarray(distance_matrix, dtype=float)
    
    # Calculate savings matrix
    # s[i, j] = D[i, 0] + D[0, j] - D[i, j]
    # We can do this vectorized.
    # D[:, 0] is column 0 (dist to depot from all nodes)
    # D[0, :] is row 0 (dist from depot to all nodes)
    
    # s[i, j] = D[i, 0] + D[0, j] - D[i, j]
    savings = D[:, 0][:, np.newaxis] + D[0, :][np.newaxis, :] - D
    
    # Diagonal is 0,0,0 -> 0. We don't care about diagonal much as ants don't pick visited.
    # But we must ensure positive values.
    
    # Normalize savings to [0, 1] range roughly, then scale up.
    # Max savings might be large.
    max_s = np.max(savings[savings > 0]) if np.any(savings > 0) else 1.0
    
    # We want high attractiveness for high savings.
    # Let's map savings to a range [1, 10] or similar.
    # If savings is 0, attractiveness is 1.
    # If savings is max, attractiveness is 10.
    
    # normalized_s = savings / max_s
    # heuristic = 1 + 9 * normalized_s
    
    # However, we must be careful: if savings is negative (i.e., going directly is *more* expensive
    # than going via depot), the attractiveness should be low.
    # The formula 1 + 9*(s/max) would give < 1 for negative s.
    # The task says "Heuristic entries that are not positive become 1e-9".
    # If we allow values < 1 but > 0, they are valid but less attractive than 1.
    # If we allow values <= 0, they become 1e-9.
    
    # Let's stick to strictly positive values to avoid the 1e-9 clamp if possible,
    # or use the clamp to punish bad edges.
    
    # Strategy:
    # heuristic[i, j] = 1.0 + (s[i, j] / max_s) * 5.0
    # This makes good edges (high savings) up to 6x more attractive than bad edges (0 savings).
    # Edges with negative savings will have heuristic < 1.0.
    # If s is very negative, heuristic can be < 0, triggering 1e-9.
    
    heuristic = 1.0 + (savings / max_s) * 5.0
    
    # Ensure no NaNs or infs
    heuristic = np.nan_to_num(heuristic, nan=1.0, posinf=10.0, neginf=1.0)
    
    # The task requires returning a matrix of the same shape.
    # It also says "Heuristic entries that are not positive become 1e-9".
    # This is handled by the evaluator, but we should probably ensure our output is clean.
    # The evaluator does: `heuristics[heuristics <= 0] = 1e-9`.
    # So we can just return the raw calculated values.
    
    return heuristic

# EVOLVE-BLOCK-END
