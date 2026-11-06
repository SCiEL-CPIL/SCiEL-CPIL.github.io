import numpy as np
import time
import sys

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 135.0  # seconds, safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    degree = A.sum(axis=1)
    
    # Handle edge case: no edges
    if degree.sum() == 0:
        return np.ones(n)
    
    # --- Compute structural features ---
    # 1. Degree
    # 2. Sum of neighbor degrees (SND)
    snd = A @ degree
    # 3. Mean neighbor degree (MND) - avoid div by zero
    mnd = snd / np.maximum(degree, 1.0)
    # 4. Second-order degree: sum of neighbors' neighbor degrees
    #    This captures how "central" the neighbors are
    snd2 = A @ snd
    mnd2 = snd2 / np.maximum(degree, 1.0)
    
    # 5. Spectral feature: Fiedler vector (2nd smallest eigenvector)
    #    This captures peripheral vs central structure
    try:
        # Use sparse if available for speed, but for n=200 dense is fine
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        # Ensure symmetry
        A_sym = (A + A.T) / 2.0
        A_csr = csr_matrix(A_sym)
        
        # Get smallest eigenvalues (k=2 for Fiedler)
        # sigma=0, which='LM' (largest magnitude) with shift-invert is tricky for 0
        # Better: use which='SM' (smallest magnitude)
        vals, vecs = eigsh(A_csr, k=2, which='SM')
        
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
        
        # Normalize Fiedler vector to zero mean, unit variance for stability
        fiedler = fiedler - fiedler.mean()
        std = fiedler.std()
        if std > 1e-10:
            fiedler = fiedler / std
        else:
            fiedler = np.zeros(n)
    except Exception:
        # Fallback if spectral fails
        fiedler = np.zeros(n)
    
    # --- Base score: blend of structural features ---
    # We want to prefer vertices that are:
    # - Low degree (easier to keep neighbors out)
    # - High MND (neighbors are hubs, so if we pick us, we block many other candidates)
    # - Low MND2 (neighbors' neighbors are not too central, less "contested")
    # - Peripheral in spectral sense (Fiedler vector magnitude)
    
    # Z-score each feature for normalization
    def zscore(x):
        std = x.std()
        if std < 1e-10:
            return np.zeros_like(x)
        return (x - x.mean()) / std
    
    z_deg = zscore(degree)
    z_mnd = zscore(mnd)
    z_mnd2 = zscore(mnd2)
    z_fiedler = zscore(np.abs(fiedler))  # Use magnitude
    
    # Weights: tune to favor the combination that works best
    # Current best: -degree + 2.0 * mnd
    # Let's try a more balanced approach with z-scores
    # Negative weight for degree (prefer low)
    # Positive weight for MND (prefer high)
    # Negative weight for MND2 (prefer low, less contested)
    # Positive weight for Fiedler magnitude (prefer peripheral)
    
    w_deg = -1.5
    w_mnd = 2.0
    w_mnd2 = -0.5
    w_fiedler = 1.0
    
    base_score = (w_deg * z_deg + w_mnd * z_mnd + 
                  w_mnd2 * z_mnd2 + w_fiedler * z_fiedler)
    
    # --- Local neighborhood refinement ---
    # Penalize vertices that are adjacent to high-scoring neighbors
    # This reduces "contestation" - if a neighbor has a high score, 
    # they might get picked first, blocking us
    
    # Iterate a few times to refine scores based on neighbor scores
    scores = base_score.copy()
    
    for iteration in range(5):
        if time.time() > DEADLINE:
            break
        
        # For each vertex, compute the mean score of its neighbors
        # High mean neighbor score means more competition
        neighbor_score_sum = A @ scores
        neighbor_count = np.maximum(degree, 1.0)
        mean_neighbor_score = neighbor_score_sum / neighbor_count
        
        # Penalize vertices with high mean neighbor score
        # This is a soft penalty: vertices surrounded by high-score vertices
        # are less likely to be selected (their neighbors will block them)
        
        # Update score with damping
        damping = 0.3
        scores = (1 - damping) * scores - damping * 0.5 * zscore(mean_neighbor_score)
    
    # --- Final polish: adjust for exact greedy simulation ---
    # Simulate the greedy process with the current scores
    # and adjust scores based on actual selection outcomes
    
    if time.time() < DEADLINE - 5:
        # Try a few different weight combinations and pick the best
        # based on simulated greedy independent set size
        
        best_score = None
        best_iset_size = -1
        
        # Weight variations to try
        weight_configs = [
            (-1.5, 2.0, -0.5, 1.0),
            (-1.0, 2.5, -0.3, 0.8),
            (-2.0, 1.8, -0.7, 1.2),
            (-1.2, 2.2, -0.4, 1.0),
            (-1.8, 2.0, -0.6, 0.9),
        ]
        
        for w_deg_c, w_mnd_c, w_mnd2_c, w_fiedler_c in weight_configs:
            if time.time() > DEADLINE - 2:
                break
            
            candidate_scores = (w_deg_c * z_deg + w_mnd_c * z_mnd + 
                               w_mnd2_c * z_mnd2 + w_fiedler_c * z_fiedler)
            
            # Apply local refinement
            ref_scores = candidate_scores.copy()
            for _ in range(3):
                neighbor_score_sum = A @ ref_scores
                mean_neighbor_score = neighbor_score_sum / np.maximum(degree, 1.0)
                ref_scores = 0.7 * ref_scores - 0.3 * 0.5 * zscore(mean_neighbor_score)
            
            # Simulate greedy selection
            selected = set()
            remaining = set(range(n))
            
            # Sort by score descending
            order = np.argsort(-ref_scores)
            
            for v in order:
                if v in selected:
                    continue
                # Check if any neighbor is selected
                if any(u in selected for u in range(n) if A[v, u] > 0):
                    continue
                selected.add(v)
            
            iset_size = len(selected)
            
            if iset_size > best_iset_size:
                best_iset_size = iset_size
                best_score = ref_scores
        
        if best_score is not None:
            scores = best_score
    
    return scores
