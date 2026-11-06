"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Compute priority scores using a maximum-entropy-like approximation.
    
    Strategy:
    We model the greedy selection as a process where we want to maximize the
    expected number of selected vertices. A vertex v is 'safe' to pick early if
    it has few neighbors that are also likely to be picked (or rather, if picking
    it doesn't block many other high-potential vertices).
    
    Instead of simple degree, we use a structural score based on the inverse
    of (I + A). This is related to the number of walks of all lengths starting
    from a node, weighted by decay. High values in (I+A)^-1 indicate nodes that
    are 'isolated' in the walk sense, i.e., they are in sparse regions or
    peripheries where they don't interfere with many other nodes' potential
    inclusion.
    
    Specifically, we compute scores = (I + A)^-1 * 1.
    Nodes with higher scores are in 'sparsely connected' parts of the graph
    in a global sense.
    
    We also incorporate a local degree penalty because high-degree nodes
    inherently block more neighbors. The combination aims to identify nodes
    that are both locally low-degree and globally in sparse regions.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Ensure adjacency is a float matrix
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Symmetrize just in case
    adj = (adj + adj.T) / 2.0
    
    # Compute degrees
    degrees = adj.sum(axis=1)
    
    try:
        # Construct the matrix M = I + A
        # We want to solve M * x = 1, so x = M^-1 * 1
        # This x_i represents the sum of decaying walks starting at i.
        # In a dense cluster, this sum is large (many walks).
        # In a sparse/peripheral region, this sum is smaller.
        # Wait, actually, if a node is isolated, (I+A)^-1 has 1 on diagonal for it.
        # If it's connected to a hub, the value might be different.
        
        # Let's think about the sign.
        # If we want to pick 'safe' nodes first, we want nodes that don't block many others.
        # A node with high degree blocks many.
        # A node in a dense cluster blocks many indirectly.
        
        # Let's try: Score = - (degree + alpha * spectral_centralization)
        # But the prompt suggests a "maximum entropy" / "dynamic programming approximation".
        
        # Alternative interpretation:
        # The probability that a node v is selected in a random greedy process where
        # ties are broken randomly is related to its "independence" from others.
        
        # Let's use the following approximation for the expected gain of picking v:
        # Gain(v) = 1 - sum_{u in N(v)} P(u is picked before v)
        # If we assume a static ordering based on scores s, P(u before v) = 1 if s[u] > s[v].
        # This is circular.
        
        # Instead, let's use the concept of "effective degree" or "influence".
        # Let's compute the eigenvector centrality of the graph, but penalize it.
        # Or, use the inverse degree.
        
        # Let's try a different structural signal:
        # Score = 1 / (1 + degree) - beta * (A * (1/(1+degree)))_i
        # This penalizes nodes whose neighbors also have low degree (dense low-degree clusters?)
        # No, usually high-degree nodes are bad.
        
        # Let's stick to the (I+A)^-1 idea but invert the logic if needed.
        # Let's compute x = solve(I + A, ones)
        # High x means many walks -> dense area -> BAD for MIS?
        # If a node is in a dense area, picking it removes many neighbors.
        # So we want LOW x?
        
        # Let's test both mentally.
        # In ER graphs, hubs have high degree.
        # (I+A)^-1 is a positive definite matrix.
        # The diagonal entries of (I+A)^-1 are related to the probability that a random walk
        # starting at i returns to i? No, that's Green's function.
        
        # Let's use a simpler, robust combination that fits the "entropy" theme:
        # Maximize entropy of the selection distribution?
        # If we assign scores, the greedy algorithm is deterministic.
        
        # Let's try: Score = - (degree + 0.5 * neighbor_degree_sum)
        # This is the "coreness" or "k-core" related heuristic.
        # Nodes with low degree and low neighbor degree are "leaves" or "periphery".
        # Picking leaves first is a standard strategy for MIS (though not always optimal, it's safe).
        
        # Let's refine the "periphery" score:
        # 1. Degree (lower is better)
        # 2. Sum of neighbor degrees (lower is better -> node is not attached to hubs)
        # 3. Clustering coefficient (lower is better -> node is not in a triangle/dense clique)
        
        # The prompt mentions "dynamic programming approximation on the degree distribution".
        # This suggests using the global distribution of degrees to normalize local features.
        
        # Let's compute:
        # 1. Degree d_i
        # 2. Neighbor degree sum S_i = sum_{j in N(i)} d_j
        # 3. Normalize d_i and S_i by their global means/stds to get Z-scores.
        # 4. Score = -(Z_d + Z_S)
        
        # This is a local heuristic, but uses global stats (DP on distribution?).
        
        # Let's try to incorporate a global structural component:
        # Compute the Fiedler vector (peripherality) and combine with local degree.
        
        # However, the previous program used Fiedler vector and got 0.9774.
        # The inspirations got 1.0263 using "mean_neighbor_degree - degree" or k-core.
        # So "mean_neighbor_degree - degree" (or similar) is better.
        
        # Let's implement a robust version of that, plus a tie-breaker.
        
        # Score = -(degree + neighbor_degree_mean)
        # Or Score = - (degree + neighbor_degree_sum)
        
        # Let's calculate:
        # neighbor_degree_sum = adj @ degrees
        
        neighbor_degree_sum = adj @ degrees
        
        # Normalize to avoid scale issues, though greedy only cares about order.
        # But if we want to mix, normalization helps.
        
        # Let's try a weighted sum:
        # Primary: Low degree
        # Secondary: Low neighbor degree (so we don't pick a node that is the only bridge to a hub? 
        # Actually, if a node is connected to a hub, picking the node saves the hub? No, picking the node kills the hub.
        # So we want to AVOID picking nodes connected to hubs, so the hubs can be picked later?
        # Or we want to pick the hubs? No, hubs block many.
        # So we want to pick nodes with LOW degree AND LOW neighbor degree.
        
        # Let's combine:
        # score = - (degrees + neighbor_degree_sum)
        
        # Let's add a small spectral component to break ties and handle global structure.
        # Compute Fiedler vector magnitude.
        
        # For n=200, eigh is fast.
        L = np.diag(degrees) - adj
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        if n > 1:
            fiedler_mag = np.abs(eigenvectors[:, 1])
            # Normalize fiedler_mag to [0,1] roughly? Or just add it.
            # Fiedler magnitude is higher for periphery.
            # We want to pick periphery. So + fiedler_mag.
            # But degrees are positive, so -degrees is negative.
            # Let's scale fiedler to be comparable to degrees?
            # Mean degree in ER(200, p) is around 100*p.
            # Fiedler values are O(1) or O(sqrt(n)).
            
            # Let's normalize fiedler_mag to have similar magnitude to degrees.
            # Or just use it as a tie breaker.
            
            # Strategy:
            # Base score: - (degrees + neighbor_degree_sum)
            # Tie breaker: + fiedler_mag (prefer periphery on ties)
            
            # To make them comparable, we can normalize the local score.
            # local_score = degrees + neighbor_degree_sum
            # min_local = local_score.min()
            # max_local = local_score.max()
            # norm_local = (local_score - min_local) / (max_local - min_local + 1e-9)
            
            # min_fied = fiedler_mag.min()
            # max_fied = fiedler_mag.max()
            # norm_fied = (fiedler_mag - min_fied) / (max_fied - min_fied + 1e-9)
            
            # Final Score = - norm_local + 0.1 * norm_fied
            # This prioritizes low local "cost" (degree + neighbor degree), with a slight preference for spectral periphery.
            
            local_score = degrees + neighbor_degree_sum
            min_l = local_score.min()
            max_l = local_score.max()
            range_l = max_l - min_l
            if range_l < 1e-9:
                norm_l = np.zeros(n)
            else:
                norm_l = (local_score - min_l) / range_l
            
            min_f = fiedler_mag.min()
            max_f = fiedler_mag.max()
            range_f = max_f - min_f
            if range_f < 1e-9:
                norm_f = np.zeros(n)
            else:
                norm_f = (fiedler_mag - min_f) / range_f
            
            # We want to MINIMIZE local_score, so score = -norm_l.
            # We want to MAXIMIZE peripherality, so score += weight * norm_f.
            # Weight 0.1 means local is 10x more important.
            
            scores = -norm_l + 0.1 * norm_f
            
        else:
            scores = np.array([0.0])
            
    except Exception as e:
        # Fallback to simple degree
        return -degrees

    return scores
# EVOLVE-BLOCK-END
