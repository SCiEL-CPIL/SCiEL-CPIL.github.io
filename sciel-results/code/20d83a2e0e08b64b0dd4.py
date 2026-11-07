"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a hybrid approach:
    - Primary: Cost-normalized Expected Improvement with a small xi offset
    - Secondary: Uncertainty-driven exploration for multimodal landscapes
    - Time-guarded to stay within wall-clock limits
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    start_time = time.time()
    DEADLINE = 295.0  # Leave 5s margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Safety: handle degenerate sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # --- Expected Improvement with small offset (xi) for aggressive improvement ---
    xi = 0.01
    z = (f_best - xi - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    EI = (f_best - xi - mu) * Phi_z + sigma_safe * phi_z
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware: normalize by cost
    cost_safe = np.maximum(cost, 1e-10)
    cost_ei = EI / cost_safe
    
    # --- Exploration term: encourage visiting uncertain regions ---
    # Weight decays if EI is already high (exploitation dominates)
    # Use a sigmoid-like gate: exploration matters most when EI is small
    ei_scale = np.maximum(np.max(EI), 1e-8)
    ei_relative = EI / ei_scale  # in [0, 1]
    
    # Exploration weight: higher when EI is low
    exploration_gate = 1.0 - 0.8 * ei_relative  # ranges from 0.2 to 1.0
    ucb_term = sigma_safe / cost_safe
    exploration_weight = 0.15
    score = cost_ei + exploration_weight * exploration_gate * ucb_term
    
    # --- Check time, if over budget return current result ---
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return score
    
    # --- Additional refinement: rank-based diversification for multimodality ---
    # Identify top candidates by cost_ei and add a small diversity bonus
    # to avoid clustering on a single promising region
    top_k = min(200, n)
    top_idx = np.argpartition(-cost_ei, top_k)[:top_k]
    
    # Compute a simple diversity score: distance in (mu, sigma) space from other top candidates
    # Use a lightweight approximation: penalize candidates very close (in score) to already-selected ones
    # This is a heuristic to encourage spread among top candidates
    
    # Simple approach: add a small bonus based on the rank in sorted order of cost_ei
    # among the top-k, so that not just the single best but a spread of near-best are considered
    rank_in_top = np.full(n, -1, dtype=np.float64)
    sorted_top = np.argsort(-cost_ei[top_idx])
    for i, idx in enumerate(top_idx):
        rank_in_top[idx] = sorted_top[i]
    
    # Small spread bonus for candidates in top-k (max bonus for rank 0, min for rank top_k-1)
    spread_bonus = np.zeros(n, dtype=np.float64)
    valid_top = rank_in_top >= 0
    if np.any(valid_top):
        max_rank = max(rank_in_top[valid_top]) + 1
        spread_bonus[valid_top] = (1.0 - rank_in_top[valid_top] / max_rank) * 0.02 / cost_safe[valid_top]
    
    score += spread_bonus
    
    # Final safety: ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
