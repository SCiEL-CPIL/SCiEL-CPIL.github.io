"""
Cost-aware Knowledge Gradient (KG) acquisition for Bayesian optimization.
Structural change from LCB/EI blend: uses a proper KG formulation that estimates
the expected reduction in f_best per unit cost, with adaptive exploration via
a quantile-based sigma floor to avoid over-exploration in high-variance regions.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 300s limit + 5s margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Knowledge Gradient acquisition function.
    
    KG estimates the expected improvement in f_best from evaluating each point.
    For a Gaussian posterior with mean mu_i and std sigma_i, the expected value
    of the minimum of f_best and the new observation is:
    
    E[min(f_best, y_i)] where y_i ~ N(mu_i, sigma_i)
    
    The KG value is: f_best - E[min(f_best, y_i)]
    
    This is computed as:
    KG_i = sigma_i * [phi(z_i) - z_i * Phi(-z_i)]
    where z_i = (f_best - mu_i) / sigma_i
    
    We then divide by cost to get cost-aware KG, and add a small exploration
    bonus for points with moderate uncertainty (not too high, not too low)
    to handle multimodality in 8D cosine8.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety fallback: simple cost-aware EI
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Knowledge Gradient computation ---
    # z_i = (f_best - mu_i) / sigma_i
    z = (f_best - mu) / sigma_safe
    
    # KG = sigma * [phi(z) - z * Phi(-z)]
    # where phi is the standard normal PDF and Phi is the CDF
    phi_z = norm.pdf(z)
    Phi_neg_z = norm.cdf(-z)
    
    kg = sigma_safe * (phi_z - z * Phi_neg_z)
    kg = np.maximum(kg, 0.0)
    
    # --- Adaptive exploration bonus ---
    # In 8D cosine8, the landscape is multimodal. We want to encourage
    # exploration of regions with moderate uncertainty, but not chase
    # pure noise (very high sigma) or exploit already-certain regions (very low sigma).
    # 
    # Use a quantile-based sigma floor: points with sigma below the 25th percentile
    # get reduced exploration weight, points above get full weight.
    # This prevents the GP from wasting budget on the most uncertain (and likely
    # uninformative) regions while still allowing exploration.
    
    sigma_sorted = np.sort(sigma_safe)
    n = len(sigma_sorted)
    q25 = sigma_sorted[int(0.25 * n)] if n > 4 else np.min(sigma_sorted)
    q75 = sigma_sorted[int(0.75 * n)] if n > 4 else np.max(sigma_sorted)
    
    # Smooth transition: full weight for sigma in [q25, q75], reduced outside
    # Below q25: linearly reduce to 0.3x
    # Above q75: linearly reduce to 0.5x (still some exploration)
    below_q25 = np.clip((q25 - sigma_safe) / np.maximum(q25, 1e-9), 0, 1)
    above_q75 = np.clip((sigma_safe - q75) / np.maximum(sigma_safe - q75, 1e-9), 0, 1)
    
    exploration_weight = 1.0 - 0.7 * below_q25 - 0.5 * above_q75
    exploration_weight = np.clip(exploration_weight, 0.3, 1.0)
    
    # --- Cost-aware scoring ---
    # Primary signal: KG / cost
    # Secondary signal: small EI component for exploitation
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # Normalize KG and EI to comparable scales
    kg_max = np.max(kg) if np.max(kg) > 1e-9 else 1e-9
    ei_max = np.max(ei) if np.max(ei) > 1e-9 else 1e-9
    
    kg_norm = kg / kg_max
    ei_norm = ei / ei_max
    
    # Blend: KG is primary, EI is secondary
    # Weight exploration weight into the KG term
    kg_weighted = kg_norm * exploration_weight
    
    # Combine
    combined = 0.8 * kg_weighted + 0.2 * ei_norm
    
    # Cost-aware: divide by cost
    # Use a softened cost to avoid extreme penalties for very cheap points
    cost_softened = cost_safe * (1.0 + 0.05 * np.exp(-cost_safe))
    
    score = combined / cost_softened
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
