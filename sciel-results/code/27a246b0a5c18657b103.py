import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy:
    1. Initialize scores using a "Hub-Avoidance" heuristic: vertices are scored by the sum of their neighbors' inverse degrees.
       This penalizes vertices adjacent to high-degree hubs (low inverse degree) and rewards those adjacent to low-degree vertices.
       The goal is to ensure hubs are not picked (low score) and that low-degree vertices in sparse regions are picked early.
    2. Perform a local search (swap-based optimization) to directly maximize the size of the independent set produced by the greedy decoder.
    3. The local search simulates the greedy selection and swaps vertex priorities if it leads to a larger independent set.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Precompute neighbor lists as numpy arrays for fast vectorized deactivation
    neighbors_list = []
    for i in range(n):
        neighbors_list.append(np.where(adj[i])[0])
    
    # 1. Initial Heuristic: Hub-Avoidance
    # We want to deprioritize vertices that are connected to hubs.
    # A hub has a high degree, so a low inverse degree.
    # If a vertex v is connected to a hub h, the term 1/deg(h) is small.
    # We want v to have a LOW score if it is connected to hubs? 
    # Wait, the prompt says: "explicitly penalizing vertices that are adjacent to high-degree hubs to ensure they are deprioritized".
    # So if v is adjacent to a hub, v's score should be low.
    # Score(v) = sum_{u in N(v)} (1 / deg(u))
    # If u is a hub, 1/deg(u) is small. So the sum is small.
    # This means vertices adjacent to hubs get a LOW score.
    # Vertices adjacent to low-degree nodes get a HIGH score (sum of larger 1/deg).
    # This encourages picking low-degree vertices in "open" areas, and avoids picking neighbors of hubs (which might be necessary to keep hubs available? No, hubs are low score so they are picked late. If we pick a neighbor of a hub, we block the hub. But if the hub is low score, it might be picked last anyway. By picking the neighbor, we remove the hub from contention, which is good if the hub would have blocked many others. But wait, if the hub is picked, it blocks many. If we pick the neighbor, the hub is blocked by the neighbor. The neighbor blocks the hub. The hub would have blocked the neighbor and others. 
    # Actually, standard MIS heuristics often prefer low degree. 
    # Let's stick to the prompt's suggestion: Score = sum of neighbors' inverse degrees.
    
    deg = adj.sum(axis=1).astype(float)
    # Avoid division by zero for isolated nodes (degree 0).
    # If deg is 0, we can treat 1/deg as 0 or some large number?
    # If a node is isolated, it has no neighbors. The sum is 0.
    # Isolated nodes should have high priority? Yes, they don't conflict.
    # But the formula gives 0.
    # Let's add a base score or handle isolated nodes.
    # If deg == 0, set inv_deg to a large value? Or just handle in sum.
    # Actually, if deg == 0, the vertex is independent of everyone. It should be picked.
    # The greedy decoder will pick it if its score is high enough.
    # If we give it 0, it might be picked last. That's fine, it will still be picked if no one else blocks it (which no one can).
    # But if there are other nodes with score 0, order matters? No, independent set size is what matters.
    
    inv_deg = np.zeros(n)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]
    
    # Score = sum of neighbors' inverse degrees
    # scores[i] = sum(inv_deg[j] for j in neighbors(i))
    scores = adj @ inv_deg
    
    # Add a small bonus for low degree?
    # The prompt says "penalizing vertices that are adjacent to high-degree hubs".
    # The sum of inverse degrees does this.
    # Let's also add a term that favors low degree vertices directly?
    # Maybe: Score = sum_inv_deg - 0.1 * deg?
    # Let's try the pure sum first, as it's the specific instruction.
    
    # 2. Fast Simulation Function
    def simulate(s):
        """
        Simulates the greedy decoder and returns the size of the independent set.
        """
        order = np.argsort(-s) # Descending score
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                # Deactivate neighbors
                if len(neighbors_list[v]) > 0:
                    active[neighbors_list[v]] = False
        return count

    # 3. Local Search
    best_size = simulate(scores)
    
    # Time budget: 12 seconds for search (limit is 150s, need safety margin)
    max_time = 12.0
    rng = np.random.default_rng(42)
    
    # We will perform random swaps.
    # Bias towards vertices that are close in rank? Or just random.
    # Random swaps over the whole set often work well for N=200.
    
    while time.time() - start_time < max_time:
        # Pick two random distinct vertices
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap scores
        scores[i], scores[j] = scores[j], scores[i]
        
        # Evaluate
        new_size = simulate(scores)
        
        if new_size > best_size:
            best_size = new_size
        else:
            # Revert
            scores[i], scores[j] = scores[j], scores[i]
            
    return scores
