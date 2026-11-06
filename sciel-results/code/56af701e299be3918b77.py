"""Rewritten: dynamic marginal-contribution scoring via multi-heuristic greedy simulation."""
import numpy as np
import time


def _greedy_sim(adjacency, scores, n):
    """Run one greedy pass. Returns (selected_mask, order_list)."""
    remaining = np.ones(n, dtype=bool)
    selected = np.zeros(n, dtype=bool)
    order = []
    # Precompute neighbor masks for fast removal
    # adjacency is bool matrix; neighbors of i = adjacency[i]
    while remaining.any():
        # Among remaining, pick argmax of scores (tie-break by lower degree)
        # Mask scores for remaining
        s = np.where(remaining, scores, -np.inf)
        v = int(np.argmax(s))
        if not remaining[v]:
            break
        selected[v] = True
        order.append(v)
        # Remove v and its neighbors from remaining
        remaining[v] = False
        remaining[adjacency[v]] = False
    return selected, order


def priority(adjacency):
    """
    Score each vertex by its marginal contribution to the independent set.
    
    Structural approach: simulate the greedy decoder under several different
    scoring heuristics. For each vertex, compute:
      - how frequently it is selected across simulations (robustness)
      - how early it is selected (lower index = more valuable)
    
    The final score combines these: vertices that are consistently selected
    early across diverse heuristics get high priority, guiding the actual
    decoder to pick them first.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Precompute basic features
    degrees = adjacency.sum(axis=1).astype(float)
    nbr_deg_sum = adjacency @ degrees
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_nbr_deg = np.where(degrees > 0, nbr_deg_sum / degrees, 0.0)
    
    # Build a set of diverse scoring heuristics to simulate
    # Each heuristic is a vector of scores per vertex
    heuristics = []
    
    # H1: min-degree (negative degree)
    heuristics.append(-degrees)
    
    # H2: negative degree + mean neighbor degree
    heuristics.append(-degrees + mean_nbr_deg)
    
    # H3: pure mean neighbor degree
    heuristics.append(mean_nbr_deg)
    
    # H4: -degree^1.5 (stronger penalty for high degree)
    heuristics.append(-np.power(degrees + 1e-9, 1.5))
    
    # H5: -degree + 2*mean_nbr_deg
    heuristics.append(-degrees + 2.0 * mean_nbr_deg)
    
    # H6: -sqrt(degree)
    heuristics.append(-np.sqrt(degrees + 1e-9))
    
    # H7: -degree + mean_nbr_deg - clustering bonus
    common_neighbors = adjacency @ adjacency
    triangles_v = 0.5 * (common_neighbors * adjacency).sum(axis=1)
    max_triangles = 0.5 * degrees * (degrees - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        clustering_coeff = np.where(max_triangles > 0, triangles_v / max_triangles, 0.0)
    heuristics.append(-degrees + mean_nbr_deg + 0.3 * clustering_coeff * degrees.mean())
    
    # H8: -degree^0.5 + 0.5 * mean_nbr_deg^0.5
    heuristics.append(-np.sqrt(degrees + 1e-9) + 0.5 * np.sqrt(mean_nbr_deg + 1e-9))
    
    # H9: -degree * (1 - mean_nbr_deg / (degrees.mean() + 1e-9))
    heuristics.append(-degrees * (1.0 - mean_nbr_deg / (degrees.mean() + 1e-9)))
    
    # H10: -degree + 0.5 * mean_nbr_deg - 0.2 * degree * (degree / (n + 1e-9))
    heuristics.append(-degrees + 0.5 * mean_nbr_deg - 0.2 * degrees * (degrees / (n + 1e-9)))
    
    # Accumulate: for each vertex, track selection frequency and average rank
    selection_count = np.zeros(n, dtype=float)
    rank_sum = np.zeros(n, dtype=float)  # lower rank = picked earlier
    num_sims = 0
    
    for h in heuristics:
        if time.time() > deadline:
            break
        try:
            selected, order = _greedy_sim(adjacency, h, n)
            num_sims += 1
            # Mark selected vertices
            for idx, v in enumerate(order):
                selection_count[v] += 1.0
                rank_sum[v] += float(idx)  # 0-indexed rank
        except Exception:
            continue
    
    # Also add a few random perturbations of the best heuristic for diversity
    best_h = heuristics[1]  # -degrees + mean_nbr_deg
    rng = np.random.default_rng(42)
    n_random = 5
    for _ in range(n_random):
        if time.time() > deadline:
            break
        noise = rng.normal(0, 0.5, size=n)
        h_perturbed = best_h + noise
        try:
            selected, order = _greedy_sim(adjacency, h_perturbed, n)
            num_sims += 1
            for idx, v in enumerate(order):
                selection_count[v] += 1.0
                rank_sum[v] += float(idx)
        except Exception:
            continue
    
    # Final score: combine frequency and early-selection tendency
    # Normalize selection_count: fraction of simulations where vertex was picked
    if num_sims > 0:
        freq = selection_count / num_sims  # 0 to 1
    else:
        freq = np.zeros(n)
    
    # Average rank (only for vertices that were selected at least once)
    # For unselected vertices, assign a high rank (n)
    selected_at_least_once = selection_count > 0
    avg_rank = np.where(selected_at_least_once, rank_sum / np.maximum(selection_count, 1), float(n))
    
    # Normalize avg_rank to [0,1] where 0 = always first, 1 = always last
    rank_min = avg_rank.min() if n > 0 else 0
    rank_max = avg_rank.max() if n > 0 else 1
    if rank_max > rank_min:
        norm_rank = (avg_rank - rank_min) / (rank_max - rank_min)
    else:
        norm_rank = np.zeros(n)
    
    # Score: high frequency + early rank (low norm_rank) is good
    # score = freq * (1 - norm_rank)
    # But we also want to incorporate the original structural signal as a tiebreaker
    # to avoid degenerate cases where many vertices have similar freq/rank.
    
    # Primary: weighted combination
    score = 10.0 * freq * (1.0 - norm_rank)
    
    # Secondary: add a small structural component to break ties
    # Normalize degrees to [0,1]
    deg_min, deg_max = degrees.min(), degrees.max()
    if deg_max > deg_min:
        norm_deg = (degrees - deg_min) / (deg_max - deg_min)
    else:
        norm_deg = np.zeros(n)
    
    # Low degree is good
    score += 0.1 * (1.0 - norm_deg)
    
    # Tertiary: tiny random tiebreaker to avoid exact ties
    rng2 = np.random.default_rng(123)
    score += rng2.uniform(0, 1e-6, size=n)
    
    return score
