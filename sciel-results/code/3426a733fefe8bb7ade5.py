import time
import numpy as np

def priority(adjacency):
    """
    Score vertices using a refined spectral and local neighborhood heuristic
    for Maximum Independent Set.
    
    Strategy:
    1. Use the Fiedler vector magnitude as a base "peripherality" score.
    2. Penalize high-degree vertices.
    3. Penalize vertices connected to many low-degree vertices (Inverse Degree Sum),
       which helps preserve low-degree candidates for the greedy decoder.
    4. Use an iterative damped fixed-point scheme to propagate "competition"
       information (neighbors' scores) through the graph.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]

    if n == 0:
        return np.array([], dtype=np.float64)
    
    if n == 1:
        return np.array([0.0], dtype=np.float64)

    # Ensure symmetry
    A = (A + A.T) / 2.0

    # Compute degree vector
    deg = A.sum(axis=1)
    
    # Handle isolated vertices (deg=0) safely for divisions
    # For inverse degree sum, we use 1/(deg+1) to avoid division by zero
    
    # Build Laplacian L = D - A
    L = np.diag(deg) - A

    try:
        # Dense eigendecomposition
        vals, vecs = np.linalg.eigh(L)
        
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # If graph is disconnected, vals[1] is 0.
        if vals[1] < 1e-5:
            # Fallback for disconnected graphs: use degree-based with inverse neighbor sum
            # This is a heuristic, but better than random
            inv_deg_neighbors = A @ (1.0 / (deg + 1.0))
            max_inv = inv_deg_neighbors.max()
            if max_inv > 0:
                inv_norm = inv_deg_neighbors / max_inv
            else:
                inv_norm = np.zeros(n)
            
            max_deg = deg.max()
            if max_deg > 0:
                deg_norm = deg / max_deg
            else:
                deg_norm = np.zeros(n)
                
            scores = -0.5 * deg_norm - 0.5 * inv_norm
            return scores.astype(np.float64)

        fiedler = vecs[:, 1]
        
        # Base score: Magnitude of Fiedler vector
        base_scores = np.abs(fiedler)
        fiedler_min = base_scores.min()
        fiedler_max = base_scores.max()
        if fiedler_max > fiedler_min:
            S_spec = (base_scores - fiedler_min) / (fiedler_max - fiedler_min)
        else:
            S_spec = np.ones(n, dtype=np.float64) / n # Uniform if flat

        # Normalize Degree
        max_deg = deg.max()
        if max_deg > 0:
            D_norm = deg / max_deg
        else:
            D_norm = np.zeros(n)

        # Compute Inverse Degree Neighbor Sum
        # I(u) = sum_{v in N(u)} 1 / (deg(v) + 1)
        inv_deg = 1.0 / (deg + 1.0)
        I_raw = A @ inv_deg
        
        I_min = I_raw.min()
        I_max = I_raw.max()
        if I_max > I_min:
            I_norm = (I_raw - I_min) / (I_max - I_min)
        else:
            I_norm = np.zeros(n)

        # Initialize scores
        # Weights: 
        # w_spec: weight for spectral peripherality
        # w_deg: weight for degree penalty
        # w_inv: weight for inverse degree neighbor penalty
        # w_comp: weight for competition (neighbor scores)
        
        w_spec = 1.0
        w_deg = 0.3
        w_inv = 0.5
        w_comp = 0.4
        
        alpha = 0.6 # Damping factor for fixed point
        
        # Initial scores
        scores = S_spec.copy()
        
        # Iterative refinement
        # We want to update scores based on current neighbor scores
        # Target(u) = S_spec(u) - w_deg * D_norm(u) - w_inv * I_norm(u) - w_comp * AvgNeighborScore(u)
        
        max_iters = 20
        for k in range(max_iters):
            if time.time() > DEADLINE:
                break
                
            # Compute average neighbor score
            # Sum of neighbor scores
            neighbor_score_sum = A @ scores
            # Average (handle deg=0)
            avg_neighbor_score = neighbor_score_sum / (deg + 1.0)
            
            # Normalize avg_neighbor_score to [0,1] for consistent scaling with other terms
            # It might be helpful to normalize it dynamically or use a fixed scale
            # Since scores are roughly in [0,1] range initially, avg_neighbor_score is also in [0,1]
            # We can just use it directly or normalize it to be safe
            ans_min = avg_neighbor_score.min()
            ans_max = avg_neighbor_score.max()
            if ans_max > ans_min:
                ans_norm = (avg_neighbor_score - ans_min) / (ans_max - ans_min)
            else:
                ans_norm = np.zeros(n)
                
            # Compute target score
            target = w_spec * S_spec - w_deg * D_norm - w_inv * I_norm - w_comp * ans_norm
            
            # Damped update
            scores = (1 - alpha) * scores + alpha * target
            
            # Optional: Re-normalize scores to [0,1] to prevent drift
            # This helps keep the competition term scaled correctly
            s_min = scores.min()
            s_max = scores.max()
            if s_max > s_min:
                scores = (scores - s_min) / (s_max - s_min)
            else:
                scores = np.zeros(n)

        return scores.astype(np.float64)

    except Exception:
        # Ultimate fallback
        return -deg.astype(np.float64)
