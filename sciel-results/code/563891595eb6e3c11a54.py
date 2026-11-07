"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Quantile-binned exploration acquisition.
    
    Partitions candidates into disjoint quantile bins based on the posterior mean (mu).
    Within each bin, the candidate with the highest standard deviation (sigma) is assigned
    a high score, while others are suppressed. This forces the selection of the most 
    uncertain candidate within each region of the posterior mean distribution, 
    promoting systematic coverage and preventing clustering in high-improvement/high-uncertainty 
    regions.
    """
    n = len(mu)
    sigma = np.maximum(sigma, 1e-9)
    
    # Number of bins
    n_bins = 20
    
    # Ensure n_bins is not greater than n
    n_bins = min(n_bins, n)
    
    # Sort indices by mu
    sorted_indices = np.argsort(mu)
    sorted_mu = mu[sorted_indices]
    sorted_sigma = sigma[sorted_indices]
    
    # Create bin assignments
    # Each bin has size n // n_bins
    bin_size = n // n_bins
    # Handle remainder by distributing extra points to the last bins or first bins?
    # Simpler: use np.linspace to define bin edges in the sorted array
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    
    scores = np.zeros(n)
    
    # For each bin, find the index with max sigma
    for i in range(n_bins):
        start = bin_edges[i]
        end = bin_edges[i+1]
        
        if start >= end:
            continue
            
        # Indices in this bin
        bin_indices = sorted_indices[start:end]
        bin_sigmas = sorted_sigma[start:end]
        
        # Find the index in the original array that has max sigma in this bin
        local_max_idx_in_bin = np.argmax(bin_sigmas)
        global_idx = bin_indices[local_max_idx_in_bin]
        
        # Assign high score to this candidate
        # Use sigma as the score to differentiate between bins
        scores[global_idx] = sorted_sigma[start:end][local_max_idx_in_bin]
        
    # Suppress scores for non-selected candidates
    # We want the selected candidates to be the top ones.
    # If we just set others to 0, the max score will be one of the selected ones.
    # To ensure "higher meaning evaluate next", and we want to pick the best among the selected,
    # the current logic works: argmax(scores) will pick the selected candidate with the highest sigma.
    
    # However, if we want to be safer, we can add a small epsilon to the selected ones 
    # and set others to -infinity or a very low number.
    # But setting others to 0 is fine if all sigmas are positive (which they are).
    
    # One issue: if multiple bins have the same max sigma, it's a tie.
    # We can break ties by adding a small random noise or just letting argmax pick the first one.
    # To be deterministic and robust, we can just return the scores.
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
