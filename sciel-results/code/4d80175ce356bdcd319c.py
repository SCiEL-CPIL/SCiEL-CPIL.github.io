import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Compute the principal eigenvector of the adjacency matrix using power iteration.
    2. The principal eigenvector captures the "coreness" or centrality of each node in the graph.
       Nodes with high eigenvector centrality are deeply embedded in the dense core of the graph.
       Nodes with low eigenvector centrality are on the periphery.
    3. For Maximum Independent Set, we generally want to pick peripheral nodes first because:
       a) They are less connected, so picking them blocks fewer other nodes.
       b) They are less "essential" to the graph's connectivity, so picking them doesn't
          disrupt the structure of the core as much (though we are just picking a set, not
          removing, the heuristic is that peripheral nodes are "cheaper" to include).
    4. We use a combination of -degree and -eigenvector_centrality.
       -degree is the standard LDF heuristic.
       -eigenvector_centrality adds a global structural signal.
       
    Score = -deg(u) - alpha * eigenvector_centrality(u)
    
    We normalize the eigenvector centrality to be comparable to degree.
    """
    start_time = time.time()
    
    # Convert to float array for fast vectorized operations
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # 1. Calculate degrees
    deg = adj.sum(axis=1)
    
    # 2. Power iteration for principal eigenvector
    # Initialize with random vector (deterministic seed for reproducibility if needed, but random is fine)
    # Using a fixed random seed might be safer to avoid variance, but let's try standard random first.
    # Actually, let's use a deterministic initialization based on degree to avoid randomness issues
    # or just use ones. Ones is a good start for connected graphs.
    v = np.ones(n)
    
    # Normalize to unit length
    v_norm = np.linalg.norm(v)
    if v_norm > 0:
        v = v / v_norm
    
    # Iterate
    # Convergence check or fixed steps? Fixed steps is safer for time.
    # 200 nodes is small. 50-100 iterations should be plenty.
    # Let's do 50 iterations.
    for _ in range(50):
        if time.time() - start_time > 140:
            break
        v_new = adj @ v
        v_norm = np.linalg.norm(v_new)
        if v_norm < 1e-12:
            break
        v = v_new / v_norm
    
    # v is now the principal eigenvector (up to sign)
    # Eigenvector centrality is the absolute value, but since it's the principal eigenvector of a 
    # non-negative matrix, it's non-negative (Perron-Frobenius).
    # Let's take absolute values just in case of numerical noise or disconnected components.
    eig_centrality = np.abs(v)
    
    # Normalize eig_centrality to have similar scale to degree
    # Max degree is at most n-1 = 199.
    # Max eig_centrality is at most 1 (since v is normalized).
    # We want the eig term to be a secondary or tertiary signal.
    # Let's scale eig_centrality by max_degree to make it comparable.
    max_deg = np.max(deg) if n > 0 else 1
    if max_deg == 0:
        max_deg = 1
        
    # Scale eig_centrality to [0, 1] relative to max possible centrality
    # Or just use it as is with a small weight.
    # Let's try: score = -deg - 0.5 * eig_centrality * (max_deg / 10)
    # This makes the eig term potentially up to 0.5 * 1 * 20 = 10, while deg is up to 200.
    # This seems reasonable. The eig term is a tie-breaker or a subtle adjustment.
    
    # Alternative: Normalize eig_centrality to mean 0, std 1, and add a small weight.
    # But the prompt suggests "prioritize vertices with lower eigenvector centrality".
    # So score should be lower for higher centrality.
    # score = -deg - beta * eig_centrality
    
    # Let's use a beta that makes the eig term significant but not dominant.
    # If we scale eig_centrality by max_deg, the range is [0, max_deg].
    # Then beta * max_deg should be small compared to max_deg.
    # Let's use beta = 0.1.
    # score = -deg - 0.1 * eig_centrality * max_deg
    
    # Wait, if max_deg is 100, and eig_centrality is 1, the penalty is 10.
    # If deg is 100, the primary score is -100.
    # So the eig term is a 10% adjustment. This seems good.
    
    scaled_eig = eig_centrality * max_deg
    beta = 0.1
    score = -deg - beta * scaled_eig
    
    # Safety deadline check
    if time.time() - start_time > 145:
        # Fallback to simple LDF
        return -deg
        
    return score
