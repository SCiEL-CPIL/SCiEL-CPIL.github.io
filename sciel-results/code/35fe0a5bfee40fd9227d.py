"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Banded sparse assimilation for Lorenz-96.
    
    Uses a local nudging approach: observed variables are nudged toward
    observations, and unobserved variables receive contributions only from
    their immediate neighbors (ring distance 1) that were observed.
    This avoids spurious long-range correlations from small ensembles.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of observed_indices
    - observed_indices: array of indices that were observed (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    start_time = time.time()
    DEADLINE = 140.0
    
    n_members, n_vars = ensemble.shape
    ens = ensemble.copy().astype(np.float64)
    
    obs_idx = np.array(sorted(observed_indices))
    all_idx = np.arange(n_vars)
    unobs_idx = np.array([i for i in all_idx if i not in set(obs_idx.tolist())])
    
    # --- Step 1: Compute ensemble spread per variable ---
    mean = ens.mean(axis=0)
    spread = ens.std(axis=0)
    
    # --- Step 2: Compute adaptive nudging weights ---
    # For observed: weight based on relative observation noise vs model uncertainty
    # For unobserved: weight based on neighbor correlation
    
    # Observation noise std
    obs_std = np.sqrt(observation_variance)
    
    # For observed variables: blend weight
    # Higher weight when observation is more reliable (lower obs_std relative to spread)
    obs_weights = np.zeros(n_vars)
    for i, idx in enumerate(obs_idx):
        # Adaptive: if spread is large relative to obs noise, trust obs more
        rel = obs_std / (spread[idx] + 1e-6)
        # Weight in [0.3, 0.95]: more aggressive when obs is reliable
        w = 0.3 + 0.65 * np.exp(-0.5 * rel ** 2)
        obs_weights[idx] = w
    
    # For unobserved variables: get contribution from nearest observed neighbors
    # In Lorenz-96 ring, unobserved (odd) indices have observed (even) neighbors at distance 1
    unobs_weights = np.zeros(n_vars)
    for idx in unobs_idx:
        # Neighbors on ring: (idx-1) and (idx+1) mod 20
        left = (idx - 1) % n_vars
        right = (idx + 1) % n_vars
        
        # Check if neighbors are observed
        left_obs = left in set(obs_idx.tolist())
        right_obs = right in set(obs_idx.tolist())
        
        # Weight for unobserved: fraction of info from observed neighbors
        # Use spread ratio to modulate
        if left_obs and right_obs:
            # Both neighbors observed: moderate confidence
            base_w = 0.45
        elif left_obs or right_obs:
            # One neighbor observed: lower confidence
            base_w = 0.25
        else:
            base_w = 0.1
        
        # Modulate by relative spread (if unobserved spread is large, be more conservative)
        spread_ratio = spread[idx] / (spread[left] + spread[right] + 1e-6)
        # If unobserved spread is much larger, reduce weight
        w = base_w * np.exp(-0.3 * max(0, spread_ratio - 1.0))
        unobs_weights[idx] = w
    
    # --- Step 3: Apply nudging ---
    # For observed: nudge toward observation
    for i in range(n_members):
        for idx in obs_idx:
            innovation = observations[list(obs_idx).index(idx)] - ens[i, idx]
            ens[i, idx] += obs_weights[idx] * innovation
    
    # For unobserved: nudge toward mean of observed neighbors
    for i in range(n_members):
        for idx in unobs_idx:
            left = (idx - 1) % n_vars
            right = (idx + 1) % n_vars
            
            # Target: weighted average of observed neighbors' values
            left_obs = left in set(obs_idx.tolist())
            right_obs = right in set(obs_idx.tolist())
            
            if left_obs and right_obs:
                target = 0.5 * ens[i, left] + 0.5 * ens[i, right]
            elif left_obs:
                target = ens[i, left]
            elif right_obs:
                target = ens[i, right]
            else:
                target = mean[idx]  # fallback to ensemble mean
            
            innovation = target - ens[i, idx]
            ens[i, idx] += unobs_weights[idx] * innovation
    
    # --- Step 4: Variance inflation to prevent collapse ---
    # Mild multiplicative inflation
    new_mean = ens.mean(axis=0)
    current_spread = ens.std(axis=0)
    pre_spread = spread  # from before update
    
    for j in range(n_vars):
        if current_spread[j] > 1e-8:
            # Inflate to maintain at least 95% of pre-update spread
            target = pre_spread[j] * 0.95
            if current_spread[j] < target:
                scale = target / current_spread[j]
                scale = min(scale, 1.3)  # cap inflation
                ens[:, j] = new_mean[j] + (ens[:, j] - new_mean[j]) * scale
        else:
            # Add small perturbation if spread collapsed
            ens[:, j] = new_mean[j] + np.random.normal(0, 0.05, n_members)
    
    # --- Step 5: Safety checks ---
    if not np.all(np.isfinite(ens)):
        ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    ens = np.clip(ens, -100.0, 100.0)
    
    # Deadline guard
    if time.time() - start_time > DEADLINE:
        pass
    
    return ens

# EVOLVE-BLOCK-END
