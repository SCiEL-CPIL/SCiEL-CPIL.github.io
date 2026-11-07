import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 25.0
    
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    if n == 0:
        return np.empty((0, 0))
    
    # Calculate pair sums and validity
    sums = demands[:, None] + demands[None, :]
    valid = sums <= capacity
    
    # Initialize heuristic with a very small value
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Extract indices for valid pairs
    i_idx, j_idx = np.where(valid)
    if len(i_idx) == 0:
        np.fill_diagonal(eta, 1e-9)
        return eta
        
    d_j = demands[j_idx]
    d_i = demands[i_idx]
    waste = capacity - sums[valid]
    
    # Calculate a local score based on waste and item size
    # Using a rational function that rewards tight fits and larger items
    # The scale is based on the minimum demand to normalize waste
    d_min = np.min(demands)
    if d_min < 1e-9:
        d_min = 1e-9
        
    scaled_waste = waste / d_min
    # Stronger penalty for waste
    local_score = d_j / (1.0 + (scaled_waste * scaled_waste) * 2.0)
    
    # Build a local affinity matrix for spectral embedding
    # We want to capture global structure.
    # Create a sparse-like matrix (dense since n=120 is small)
    local_affinity = np.zeros((n, n))
    local_affinity[valid] = local_score
    
    # Spectral embedding:
    # 1. Construct a similarity matrix S (local_score, symmetric? no, directional in ACO but let's use symmetric part for structure)
    #    Actually, ACO is directional. But community structure is often symmetric.
    #    Let's use the symmetric part: (S + S^T) / 2
    sym_affinity = (local_affinity + local_affinity.T) / 2.0
    
    # 2. Compute eigenvectors
    #    We want to project items into a lower-dimensional space where
    #    similar items (those that fit well together) are close.
    #    Using the largest few eigenvectors of the similarity matrix.
    
    # Check time
    if time.time() > deadline - 5.0:
         # Fallback to simple local score if too slow
         eta[valid] = local_score
         np.fill_diagonal(eta, 1e-9)
         eta[~np.isfinite(eta)] = 1e-9
         return eta
         
    try:
        # Eigen decomposition
        # sym_affinity is symmetric real
        vals, vecs = np.linalg.eigh(sym_affinity)
        
        # Sort by eigenvalue descending
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]
        vals = vals[idx]
        
        # Take top k components
        k = min(10, n)
        embedding = vecs[:, :k]
        
        # Check time
        if time.time() > deadline - 2.0:
             eta[valid] = local_score
             np.fill_diagonal(eta, 1e-9)
             eta[~np.isfinite(eta)] = 1e-9
             return eta
             
        # Calculate global affinity based on Euclidean distance in embedding space
        # Items that are close in embedding space are likely to belong to the same "community"
        # and should be encouraged to be placed together.
        
        # Pairwise distances in embedding space
        # D[i, j] = ||embedding[i] - embedding[j]||^2
        # We want a kernel: K[i, j] = exp(-beta * D[i, j])
        
        # Compute squared distances efficiently
        # ||a||^2 + ||b||^2 - 2 a.b
        sq_norms = np.sum(embedding ** 2, axis=1)
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (embedding @ embedding.T)
        dist_sq = np.maximum(dist_sq, 0) # Numerical safety
        
        # Choose beta to scale distances appropriately
        # Average distance
        avg_dist = np.mean(dist_sq[dist_sq > 0]) if np.any(dist_sq > 0) else 1.0
        if avg_dist < 1e-9:
             avg_dist = 1.0
        beta = 1.0 / (avg_dist * 0.5) # Heuristic scaling
        
        global_affinity = np.exp(-beta * dist_sq)
        
        # Combine local and global affinity
        # We want to boost local scores if the global affinity is high.
        # Or use global affinity to break ties?
        # Let's blend them: eta_final = local_score * (1 + alpha * global_affinity)
        # Or eta_final = local_score + alpha * global_affinity * local_score
        
        # The global affinity is symmetric and smooth.
        # The local score is the primary driver.
        # Let's multiply local score by a factor derived from global affinity.
        # If two items are in the same community, they should fit well.
        
        # Normalize global affinity to [0, 1] range roughly
        # Max global affinity is 1 (distance 0).
        
        # Blend factor
        alpha = 0.5
        
        # We only care about valid pairs
        # The global affinity is computed for all pairs, including invalid ones.
        # We will only use it to modulate the valid ones.
        
        # Final score for valid pairs
        # We want to preserve the relative order of local scores but boost
        # pairs that are also close in the spectral embedding.
        
        # One issue: The spectral embedding is based on the local scores themselves.
        # So items that have high local scores with each other will be close in embedding.
        # This creates a feedback loop.
        # Is this helpful? It might reinforce strong communities.
        
        # Let's try: eta[valid] = local_score * (1 + alpha * global_affinity[valid])
        
        global_affinity_valid = global_affinity[valid]
        
        # Ensure global_affinity_valid is not too large or small
        # It's in (0, 1]
        
        scores = local_score * (1.0 + alpha * global_affinity_valid)
        
    except Exception:
        # Fallback to local score if spectral fails
        scores = local_score
        
    # Assign to matrix
    eta[valid] = scores
    
    # Diagonal: set to 1e-9
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure no non-finite values
    eta[~np.isfinite(eta)] = 1e-9
    
    # Ensure positive
    eta[eta <= 0] = 1e-9
    
    return eta
