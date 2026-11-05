import numpy as np
import time

def priority(adjacency):
    """
    Computes a priority score for each vertex to maximize the size of the 
    independent set produced by a greedy decoder.
    
    Strategy:
    1. Use a fast baseline greedy simulation (sorted by degree) to determine 
       an "expected" set of early picks.
    2. Vertices that are neighbors of these early picks are "endangered" and 
       should be deprioritized to avoid being skipped if the actual order shifts slightly.
    3. Vertices that are far from early picks (low degree, not neighbors of hubs) 
       are "safe" and should be prioritized.
    4. The final score combines raw degree (lower is better for MIS greedy) with 
       a "safety" penalty based on the simulated early picks.
    """
    start_time = time.time()
    
    # Ensure adjacency is a numpy array
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    # Deadline guard
    deadline = start_time + 110.0
    
    # 1. Compute degrees
    deg = np.sum(adjacency, axis=1).astype(int)
    
    # 2. Baseline Simulation: 
    # We simulate the greedy process using degree as the primary sort key.
    # This gives us a sense of which vertices are "dominant" (likely to be picked first).
    # In a standard MIS greedy, low degree is often preferred, but high degree hubs 
    # block many nodes. 
    # Let's simulate picking vertices in order of *lowest* degree first (common heuristic).
    # If we pick low degree first, high degree nodes might be left for later, 
    # but they are neighbors to many picked nodes, so they get skipped.
    
    # Actually, a better baseline for "what will be picked early" is often 
    # the nodes that are NOT neighbors of other low-degree nodes.
    
    # Let's use a simple proxy: 
    # "Risk" of a node v is the sum of degrees of its neighbors? No.
    # Let's simulate a specific greedy order: Sort by (degree, index).
    # We mark which nodes would be picked in this specific order.
    
    # However, the actual decoder uses *our* scores. 
    # If we want to maximize size, we want to pick nodes that don't block many others.
    # This is equivalent to minimizing the "weight" of the closed neighborhood.
    
    # Let's try a different angle: 
    # The greedy decoder picks v1, v2, ... 
    # v_i is picked if it is not a neighbor of any v_j for j < i.
    # To maximize count, we want to avoid picking "bad" vertices (high degree) early.
    # But if we give high degree vertices low priority, they might be picked late.
    # If they are picked late, they are likely already skipped (blocked by neighbors).
    # So, giving low priority to high degree nodes is good.
    
    # The current program does this with a weighted sum. 
    # My new strategy: 
    # I will compute a "blocking potential" for each node.
    # A node v blocks its neighbors. 
    # If v is picked, it removes deg(v) candidates from the pool.
    # We want to pick nodes that block few candidates.
    # So, score should be inversely related to degree.
    
    # But there is a second order effect: 
    # If node u has high degree, and we pick u, we lose many nodes.
    # If we skip u (by giving it low priority), we might pick its neighbors.
    # But if u is a hub, its neighbors are likely connected to each other? 
    # In ER graphs, no.
    
    # Let's use a "Simulated Greedy" to identify "Core" nodes that are hard to avoid.
    # If a node is in the high k-core, it's hard to remove.
    
    # Let's try a "Dynamic Priority" based on a single pass of greedy simulation 
    # using a simple score, then adjusting.
    
    # Simple Score: -degree (lower degree = higher priority)
    # Simulate:
    selected_sim = np.zeros(n, dtype=bool)
    order = np.argsort(deg)  # Low degree first
    
    for v in order:
        if not selected_sim[v]:
            # Check if any neighbor is already selected
            # In a true simulation, we'd check neighbors of previously selected.
            # For speed, we can just assume this order is the "intended" one 
            # and see who gets picked.
            # But we can't do a full O(N^2) check for every step in a tight loop if N is large, 
            # but N=200 is small.
            
            # Check if v is blocked by any previously selected node
            is_blocked = False
            # We can optimize this by maintaining a "blocked" set
            # But for N=200, even O(N^2) per step is 200*200*200 = 8M ops, which is fine.
            
            # Let's do a proper simulation of the greedy decoder 
            # using the score: -degree.
            pass
    
    # Actually, let's just compute the score directly based on "Efficiency".
    # Efficiency of picking v = 1 / (1 + deg(v)) ?
    
    # Let's look at the "Mean Neighbor Degree" (MND) again.
    # If a node has neighbors with high MND, it's in a dense region.
    # Dense regions have smaller MIS.
    # So we want to pick nodes in sparse regions.
    
    # New Strategy: 
    # Score = -(degree + alpha * mean_neighbor_degree)
    # This is similar to the current program but let's try a different combination.
    
    # What if we use the *second* order effect?
    # If we pick v, we lose N(v).
    # The value of losing N(v) depends on the degrees of N(v).
    # Loss(v) = sum(deg(u) for u in N(v))
    # We want to minimize Loss(v).
    # So score = -Loss(v) - deg(v)
    
    # Let's compute Loss(v) = sum of degrees of neighbors.
    # This is exactly `adjacency @ deg`.
    
    loss = adjacency @ deg.astype(float)
    
    # Combine degree and loss
    # Normalize both
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    deg_norm = normalize(deg.astype(float))
    loss_norm = normalize(loss)
    
    # We want low degree and low loss.
    # So score is negative of these.
    
    # Let's try a weighted sum where Loss is more important?
    # In ER graphs, high degree nodes have neighbors with high average degree?
    # Yes, assortativity is weak, but high degree nodes connect to many nodes.
    # The sum of degrees of neighbors (loss) is roughly deg(v) * avg_deg.
    # So loss is proportional to degree.
    # This might not add much new info.
    
    # Let's try "Clustering Coefficient" again but with a different weight.
    # High clustering means neighbors are connected, so picking v "wastes" less potential MIS?
    # No, if neighbors are connected, they can't all be in MIS anyway.
    # So picking a node in a clique is "efficient" because it blocks nodes that would block each other.
    # So HIGH clustering should be GOOD (higher priority)?
    # Let's test this hypothesis.
    # If v is in a triangle, picking v blocks 2 neighbors who are connected. 
    # If v is in a star, picking v blocks many leaves who are NOT connected.
    # The star case is bad for MIS size because we lose many independent nodes.
    # The triangle case is better because we lose fewer "potential" independent nodes.
    # So, we should PRIORITIZE high clustering?
    
    # The current program PUNISHES high clustering (weight 1.0 * cc_norm added to negative score).
    # Let's REVERSE the clustering weight.
    
    # Compute Clustering Coefficient
    cc = np.zeros(n, dtype=float)
    for v in range(n):
        d = deg[v]
        if d < 2:
            cc[v] = 0.0
            continue
        neighbors = np.where(adjacency[v])[0]
        sub_adj = adjacency[np.ix_(neighbors, neighbors)]
        edges_among = int(np.sum(sub_adj)) // 2
        cc[v] = 2.0 * edges_among / (d * (d - 1))
        
    cc_norm = normalize(cc)
    
    # New Score: 
    # -deg (low degree is good)
    # -loss (low loss is good)
    # +cc (high clustering is good, because it's "dense" locally, less waste)
    
    # Let's try: score = -(2.0 * deg_norm + 1.0 * loss_norm - 0.5 * cc_norm)
    # Note: the minus sign in front of the whole bracket means:
    # -2*deg - 1*loss + 0.5*cc
    
    score = -(2.0 * deg_norm + 1.0 * loss_norm - 0.5 * cc_norm)
    
    # Check time
    if time.time() > deadline:
         # Fallback
         return -deg.astype(float)
         
    return score
