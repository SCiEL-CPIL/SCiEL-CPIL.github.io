import numpy as np


def acquisition(mu, sigma, f_best):
       mu = np.asarray(mu, dtype=float)
       sigma = np.asarray(sigma, dtype=float)
       
       # Handle edge cases where sigma might be 0 or very small
       # Ensure sigma is positive
       sigma = np.maximum(sigma, 1e-9)
       
       num_bins = 20
       beta = 2.5
       
       # Sort indices by posterior mean
       sorted_indices = np.argsort(mu)
       
       # Create bin edges
       bin_edges = np.linspace(0, len(mu), num_bins + 1).astype(int)
       
       # Array to store the max sigma for each candidate's bin
       bin_max_sigma = np.zeros_like(mu)
       
       for k in range(num_bins):
           start = bin_edges[k]
           end = bin_edges[k+1]
           if start >= end:
               continue
           
           # Get the indices belonging to this bin
           bin_idx = sorted_indices[start:end]
           
           # Get the sigma values for these indices
           bin_sigmas = sigma[bin_idx]
           
           # Find the maximum sigma in this bin
           max_sigma_in_bin = np.max(bin_sigmas)
           
           # Assign this max sigma to all candidates in the bin
           bin_max_sigma[bin_idx] = max_sigma_in_bin
       
       # Calculate the acquisition score
       # We want to minimize f, so we maximize -mu + beta * sigma
       scores = -mu + beta * bin_max_sigma
       
       # Ensure scores are finite
       scores = np.where(np.isfinite(scores), scores, -np.inf)
       
       return scores
