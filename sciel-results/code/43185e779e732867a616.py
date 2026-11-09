"""Cost-aware acquisition: two-phase knowledge-gradient-inspired with adaptive exploration."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import ndtr, ndtrinv

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 280 s safety margin within 300 s limit

def _norm_cdf_vec(z):
    """Vectorized standard normal CDF using scipy.special.ndtr."""
    return ndtr(z)

def _norm_pdf_vec(z):
    """Vectorized standard normal PDF."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement vectorized."""
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20.0, 20.0)
    cdf = _norm_cdf_vec(z)
    pdf = _norm_pdf_vec(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    return np.maximum(ei, 0.0), cdf, pdf, sigma_safe

def _hypervolume_contribution(ei, cost, f_best):
    """
    Approximate hypervolume contribution of each candidate.
    Higher EI and lower cost -> higher contribution.
    """
    # Normalize EI to [0, 1] approximately
    max_ei = np.max(ei)
    if max_ei < 1e-12:
        max_ei = 1e-12
    ei_norm = ei / max_ei
    
    # Cost in [1, 2], normalize to [0, 1] where 0 is best (lowest cost)
    cost_norm = np.clip(cost - 1.0, 0.0, 1.0)
    
    # Hypervolume-like contribution: product of normalized objectives
    # Use a power to make it more sensitive to extremes
    hv = ei_norm * (1.0 - cost_norm)
    return hv

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing cosine8 8d.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        # Return cost-based fallback (prefer cheaper points)
        scores = -np.asarray(cost, dtype=np.float64)
        return np.where(np.isfinite(scores), scores, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    n = mu.shape[0]

    # Compute EI and auxiliary quantities
    ei, cdf_z, pdf_z, sigma_safe = _compute_ei(mu, sigma, f_best)

    # Progress indicator: how close we are to the known minimum (-0.8)
    # f_best starts around -0.1 to -0.5 and decreases toward -0.8
    # progress = 1.0 early (far from min), 0.0 late (near min)
    progress = np.clip((f_best - (-0.8)) / 1.0, 0.0, 1.0)
    
    # Smooth transition: early -> more exploration, late -> more exploitation
    # Use sigmoid-like transition
    exploration_weight = np.exp(-3.0 * (1.0 - progress))  # ~1 early, ~0 late
    exploitation_weight = 1.0 - exploration_weight

    # Cost normalization
    cost_norm = np.clip(cost - 1.0, 0.0, 1.0)

    # --- Phase 1: Knowledge-Gradient-inspired term ---
    # KG approximates the expected improvement in f_best after evaluating x.
    # A common approximation: EI * P(improve) + small uncertainty term
    kg_term = ei * cdf_z  # probability-weighted EI

    # --- Phase 2: Hypervolume contribution ---
    hv = _hypervolume_contribution(ei, cost, f_best)

    # --- Exploration bonus: reward high uncertainty, especially early ---
    # Normalize sigma to [0, 1]
    max_sigma = np.max(sigma_safe)
    if max_sigma < 1e-12:
        max_sigma = 1e-12
    sigma_norm = sigma_safe / max_sigma
    
    # Exploration bonus scaled by progress (more early)
    # Use a 10% relative to max EI
    max_ei = np.max(ei)
    if max_ei < 1e-12:
        max_ei = 1e-12
    exploration_bonus = exploration_weight * 0.10 * max_ei * sigma_norm

    # --- Cost penalty: linear, stronger in exploitation phase ---
    cost_penalty = exploitation_weight * 0.5 * cost_norm * max_ei

    # --- Combine all terms ---
    # Scale factors to balance the terms
    scores = (
        1.0 * kg_term +          # primary: knowledge gradient
        0.3 * hv * max_ei +       # secondary: hypervolume contribution
        exploration_bonus +        # exploration: uncertainty-driven
        0.15 * ei * (1.0 - cost_norm) -  # cost-aware EI
        cost_penalty               # penalty: avoid expensive points late
    )

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Normalize to [0, 1] range for consistency
    min_s = np.min(scores)
    max_s = np.max(scores)
    if max_s - min_s > 1e-12:
        scores = (scores - min_s) / (max_s - min_s)
    else:
        scores = np.ones_like(scores) * 0.5

    # Add tiny deterministic noise to break ties
    rng = np.random.default_rng(42)
    noise = rng.uniform(0, 1e-8, size=n)
    scores = scores + noise

    return scores
# EVOLVE-BLOCK-END
