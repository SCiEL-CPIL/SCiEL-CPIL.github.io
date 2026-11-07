"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Competitive exclusion score based on opportunity cost.
    For each item, we estimate its marginal utility relative to the competition
    from other items that share its binding constraints. Items that are efficient
    across all constraints (especially the most binding ones) get higher scores.
    
    The heuristic combines:
    1. Per-constraint efficiency ratios (prize / weight per constraint)
    2. Harmonic mean to emphasize the bottleneck constraint
    3. Competitive exclusion: penalize items whose selection would crowd out
       high-value alternatives in the same constraints
    4. Power amplification to sharpen selection
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    
    # Per-constraint efficiency: prize / (weight + epsilon)
    eps = 1e-9
    efficiency = prizes / (weights + eps)  # shape: (5, 100)
    
    # Harmonic mean across constraints to emphasize the bottleneck
    # harmonic_mean = n / sum(1/x)
    inv_eff = 1.0 / (efficiency + eps)
    harmonic_mean = n_constraints / (inv_eff.sum(axis=0) + eps)  # shape: (100,)
    
    # Competitive exclusion score:
    # For each item i, estimate the opportunity cost by looking at other items
    # that would compete for the same constraint resources.
    # 
    # For constraint c, item i has weight w[i,c]. The "competition" is the sum
    # of prizes of other items j that have significant weight in constraint c.
    # We normalize this to get a relative opportunity cost.
    
    # Compute for each constraint, the total prize "pressure" from all items
    # that use that constraint (weighted by their weight in that constraint)
    # This gives a sense of how "crowded" each constraint is.
    
    # For each constraint c, compute a competitive score for each item i:
    # comp[i, c] = prizes[i] / (sum of prizes[j] * weights[j,c] for all j with weights[j,c] > 0)
    # This normalizes the item's prize by the total "value density" competing in that constraint.
    
    competitive_scores = np.zeros_like(harmonic_mean)
    
    for c in range(n_constraints):
        # Total prize-weight product for constraint c (proxy for competition intensity)
        # Items with high weight in this constraint compete more intensely
        w_c = weights[c, :]  # shape: (100,)
        p_c = prizes[c, :]   # shape: (100,)
        
        # Competition intensity: sum of prizes weighted by weights in this constraint
        # This represents the "crowding" in this constraint
        competition = np.dot(p_c, w_c) + eps  # scalar
        
        # For each item, its competitive score in this constraint is its prize
        # normalized by the competition and its own weight (so lighter items in
        # crowded constraints get penalized less)
        # score[i, c] = p_c[i] / (competition * (w_c[i] + eps))
        # But we want this to reflect opportunity cost: how much value we give up
        # by NOT selecting this item vs selecting competitors.
        
        # Alternative: rank-based competitive exclusion
        # For each constraint, rank items by prize/weight ratio
        # Items that are high-ranked in many constraints are less "crowded out"
        
        # Simpler approach: normalized prize contribution
        # Each item's share of the total prize in constraint c, adjusted by weight
        share = p_c / (competition + eps)  # shape: (100,)
        # Penalize items with high weight in this constraint (they consume more)
        share_adjusted = share / (w_c + eps)  # shape: (100,)
        
        competitive_scores += share_adjusted
    
    # Normalize competitive_scores to [0, 1] range approximately
    cs_max = competitive_scores.max()
    cs_min = competitive_scores.min()
    if cs_max > cs_min:
        competitive_scores = (competitive_scores - cs_min) / (cs_max - cs_min)
    else:
        competitive_scores = np.ones_like(competitive_scores)
    
    # Combine harmonic mean efficiency with competitive exclusion score
    # Use geometric combination to balance both aspects
    # Normalize harmonic_mean to [0, 1]
    hm_max = harmonic_mean.max()
    hm_min = harmonic_mean.min()
    if hm_max > hm_min:
        hm_normalized = (harmonic_mean - hm_min) / (hm_max - hm_min)
    else:
        hm_normalized = np.ones_like(harmonic_mean)
    
    # Geometric mean of the two normalized scores
    combined = np.sqrt(hm_normalized * competitive_scores)
    
    # Power amplification to sharpen selection (higher power = more selective)
    power = 1.5
    scores = combined ** power
    
    # Ensure all scores are positive
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
