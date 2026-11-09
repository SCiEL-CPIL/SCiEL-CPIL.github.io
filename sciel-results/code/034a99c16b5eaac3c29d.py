import numpy as np
import time
import random

# Pre-compute indices for fast evaluation
N = 10
y_idx, x_idx = np.indices((N, N))
DIFF_IDX = (x_idx - y_idx) + 9  # 0..18
X_IDX = x_idx                   # 0..9
SUM_IDX = x_idx + y_idx         # 0..18
SUM2_IDX = x_idx + 2 * y_idx    # 0..27

def _compute_ratio(P):
    """
    Compute D/M ratio.
    P is a 10x10 array of non-negative weights.
    """
    total = np.sum(P)
    if total <= 1e-9:
        return 0.0
    
    # Normalize P to probabilities for entropy calculation
    # Note: Entropy is scale invariant, but we need probs for the formula
    # H = - sum (p_i) log2(p_i)
    
    # H_{-1} : distribution of x - y (19 bins)
    dist_diff = np.zeros(19)
    np.add.at(dist_diff, DIFF_IDX.ravel(), P.ravel())
    
    # H_0 : distribution of x (10 bins)
    dist_x = np.zeros(10)
    np.add.at(dist_x, X_IDX.ravel(), P.ravel())
    
    # H_1 : distribution of x + y (19 bins)
    dist_sum = np.zeros(19)
    np.add.at(dist_sum, SUM_IDX.ravel(), P.ravel())
    
    # H_2 : distribution of x + 2y (28 bins)
    dist_sum2 = np.zeros(28)
    np.add.at(dist_sum2, SUM2_IDX.ravel(), P.ravel())
    
    def entropy_from_counts(counts):
        c_total = counts.sum()
        if c_total <= 1e-15:
            return 0.0
        probs = counts / c_total
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
        
    H_diff = entropy_from_counts(dist_diff)
    H_x = entropy_from_counts(dist_x)
    H_sum = entropy_from_counts(dist_sum)
    H_sum2 = entropy_from_counts(dist_sum2)
    
    M = max(H_x, H_sum, H_sum2)
    
    if M <= 1e-9:
        return 0.0
        
    return H_diff / M

def _generate_candidate_P(n_points, scale_var=1.0):
    """
    Generate a candidate distribution by placing n_points Gaussian blobs
    on the 10x10 grid. This reduces the search space from 100 continuous
    variables (with sum constraint) to ~3*n_points continuous variables.
    """
    P = np.zeros((N, N))
    for _ in range(n_points):
        # Center of the blob
        cx = random.uniform(0, N-1)
        cy = random.uniform(0, N-1)
        
        # Weights for the blob
        weight = random.uniform(0.1, 1.0)
        
        # Spread (sigma) - keep it small to allow for sharp peaks, 
        # but large enough to cover neighbors if needed
        sigma = random.uniform(0.5, 2.5)
        
        # Create a local grid around the center
        # To be fast, we only add to cells within 3*sigma
        r = int(np.ceil(3 * sigma))
        cx_min = max(0, int(cx) - r)
        cx_max = min(N, int(cx) + r + 1)
        cy_min = max(0, int(cy) - r)
        cy_max = min(N, int(cy) + r + 1)
        
        for y in range(cy_min, cy_max):
            for x in range(cx_min, cx_max):
                dist_sq = (x - cx)**2 + (y - cy)**2
                val = weight * np.exp(-dist_sq / (2 * sigma**2))
                P[y, x] += val
                
    # Normalize
    total = np.sum(P)
    if total > 0:
        P /= total
    else:
        # Fallback to uniform
        P = np.ones((N, N)) / 100.0
    return P

def _refine_P(P, steps=1000, lr=0.05):
    """
    Perform a simple gradient-free local refinement using coordinate descent
    on the log-probabilities of the non-zero entries, or simply perturb and keep.
    Given the low dimensionality of the 'blob' representation is lost after generation,
    we refine the raw P vector directly using a random walk that preserves positivity.
    """
    best_P = P.copy()
    best_score = _compute_ratio(best_P)
    
    # We will use a simple random walk on the log-scale of the probabilities
    # to allow for multiplicative changes.
    # log_P = log(P + eps)
    # Update log_P, then exp back.
    
    # To make it faster and more robust, we'll do a "shrink and grow" on random subsets
    for _ in range(steps):
        # Pick a random subset of cells to perturb
        mask = np.random.rand(N, N) < 0.1  # 10% of cells
        if not np.any(mask):
            continue
            
        # Perturb the values in these cells
        # Multiplicative noise
        noise = np.random.normal(1.0, 0.1, size=(N, N))
        new_P = best_P.copy()
        new_P[mask] *= noise[mask]
        
        # Ensure non-negative
        new_P = np.maximum(new_P, 1e-12)
        
        # Normalize
        total = np.sum(new_P)
        if total < 1e-9:
            continue
        new_P /= total
        
        score = _compute_ratio(new_P)
        if score > best_score:
            best_score = score
            best_P = new_P
            
    return best_P, best_score

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600 - 10  # 10s safety margin
    
    best_P = None
    best_score = -np.inf
    
    # Strategy:
    # 1. Try a few specific structural heuristics (e.g., uniform on diagonals)
    # 2. Use random blob generation + local refinement
    
    # Heuristic 1: Uniform on main diagonal (x=y)
    P_diag = np.zeros((N, N))
    for i in range(N):
        P_diag[i, i] = 1.0
    P_diag /= N
    score_diag = _compute_ratio(P_diag)
    if score_diag > best_score:
        best_score = score_diag
        best_P = P_diag.copy()
        
    # Heuristic 2: Uniform on anti-diagonal (x+y=9)
    P_adiag = np.zeros((N, N))
    for i in range(N):
        P_adiag[i, N-1-i] = 1.0
    P_adiag /= N
    score_adiag = _compute_ratio(P_adiag)
    if score_adiag > best_score:
        best_score = score_adiag
        best_P = P_adiag.copy()
        
    # Heuristic 3: Uniform on rows (x is uniform, y is concentrated?)
    # Let's just rely on the random search with blobs which is more general.
    
    # Random Search with Blobs
    # We vary the number of blobs to explore different sparsities
    blob_counts = [3, 5, 8, 12, 20]
    
    # Time budget allocation
    # We don't know how many iterations we can do, so we loop until deadline.
    # We alternate between different blob counts to maintain diversity.
    
    idx = 0
    while time.time() < deadline:
        n_blobs = blob_counts[idx % len(blob_counts)]
        idx += 1
        
        # Generate candidate
        P_cand = _generate_candidate_P(n_blobs)
        
        # Refine
        # Adjust refinement steps based on remaining time
        remaining = deadline - time.time()
        if remaining < 1:
            break
            
        # Estimate time per refinement step. 
        # _compute_ratio is fast (~ms). 
        # Let's do a fixed number of refinement steps per candidate, 
        # but cap it if we are running out of time.
        # If we have > 100s left, do more steps.
        steps = 500 if remaining > 100 else 100 if remaining > 20 else 10
        
        P_ref, score_ref = _refine_P(P_cand, steps=steps)
        
        if score_ref > best_score:
            best_score = score_ref
            best_P = P_ref.copy()
            
            # If we find a significantly better score, we might want to 
            # do a more intensive refinement on this specific P.
            # But for now, the loop will naturally pick up on good regions
            # if we re-seed or if the random walk stays in the basin.
            
    # If no valid P was found (shouldn't happen), return uniform
    if best_P is None:
        return np.ones((N, N)) / 100.0
        
    # Final validation and normalization
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total <= 1e-9:
        return np.ones((N, N)) / 100.0
    best_P /= total
    
    return best_P
