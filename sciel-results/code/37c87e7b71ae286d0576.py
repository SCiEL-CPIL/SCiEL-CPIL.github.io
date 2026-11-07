"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for ACO orienteering.
    
    Strategy: Prize-to-Total-Cost Ratio weighted by Feasibility Slack.
    The heuristic for edge (i, j) is based on the prize of node j divided by the 
    total cost of visiting j from i (distance i->j plus return j->0), multiplied 
    by the remaining slack in the path budget. This prioritizes high-value nodes 
    that are "cheap" to visit in terms of total tour length increase and that 
    leave sufficient budget for further exploration.
    
    The result is squared to sharpen the distinction between good and bad edges, 
    guiding ants more decisively towards high-efficiency moves.
    """
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Ensure finite values
    distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, 1e6)
    prizes = np.where(np.isfinite(prizes), prizes, 0.0)
    
    eps = 1e-9
    
    # d[i, j] for all i, j
    dij = distance_matrix  # shape (n, n)
    
    # d[j, 0] for all j (return cost to home)
    d_j_0 = distance_matrix[:, 0]  # shape (n,)
    
    # Total cost of visiting j from i: d[i, j] + d[j, 0]
    # This represents the increase in tour length if we go i -> j and then eventually j -> 0
    total_cost = dij + d_j_0[None, :]  # shape (n, n)
    
    # Slack: remaining budget after visiting j and returning home
    # If total_cost > max_len, slack is 0 (infeasible)
    slack = max_len - total_cost
    slack = np.maximum(slack, 0.0)
    
    # Prize of node j, broadcast over rows
    prize_j = prizes[None, :]  # shape (1, n)
    
    # Base heuristic: (Prize / Total Cost) * Slack
    # This rewards high prize, low total cost, and high remaining budget
    # We add eps to total_cost to avoid division by zero
    base_heuristic = (prize_j / (total_cost + eps)) * slack
    
    # Handle diagonal: no self-loops
    np.fill_diagonal(base_heuristic, 0.0)
    
    # Ensure non-negative and finite
    base_heuristic = np.where(np.isfinite(base_heuristic) & (base_heuristic > 0), base_heuristic, 0.0)
    
    # Sharpen the heuristic to amplify differences between good and bad edges
    # Squaring emphasizes high-value, low-cost edges significantly
    heuristic = base_heuristic ** 2.0
    
    # Final safety: ensure positive finite values for ACO
    # ACO requires positive heuristics. We map 0 to a very small positive number.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
