import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    # 1. Compute Degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # 2. Compute Average Neighbor Degree
    # Sum of degrees of neighbors
    nbr_deg_sums = adj @ degrees
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degrees > 0, nbr_deg_sums / degrees, 0.0)
        
    # 3. Compute 2nd Eigenvector (Spectral Component)
    # For small N, this is fast.
    try:
        if time.time() < deadline - 10:
            # Symmetric matrix
            eigvals, eigvecs = np.linalg.eigh(adj)
            # eigh returns eigenvalues in ascending order
            # We want the 2nd largest, which is the second to last
            # Or the largest magnitude? 
            # For connected graphs, largest is Perron. Second largest is Fiedler.
            # If graph is disconnected, there might be multiple 1s.
            # Let's take the eigenvector corresponding to the 2nd largest eigenvalue.
            idx = -2
            v2 = eigvecs[:, idx]
            # Normalize
            max_val = np.max(np.abs(v2))
            if max_val > 0:
                v2 = v2 / max_val
            else:
                v2 = np.zeros(n)
        else:
            v2 = np.zeros(n)
    except Exception:
        v2 = np.zeros(n)
        
    # 4. Combine Scores
    # Base: Low degree is good
    # Bonus: High avg neighbor degree is good (removes hubs)
    # Spectral: Align with Fiedler vector (bipartition heuristic)
    
    # Weights
    w_deg = 1.0
    w_nbr = 1.0
    w_spec = 0.1  # Small weight for spectral
    
    scores = -w_deg * degrees + w_nbr * avg_nbr_deg + w_spec * v2
    
    # Optional: Add Core Number penalty if time permits
    # Core number is hard to compute quickly without a library, but for N=200 it's fine.
    # However, the degree/nbr degree combo is already strong.
    
    return scores
