import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Dynamic Iterative Re-scoring (Parallel Peeling Simulation)
    
    Instead of using static local features (like degree or average neighbor degree),
    this strategy simulates the greedy selection process in a parallel, iterative manner.
    
    1. Initialize scores based on low degree (LDF heuristic) but with a small random-like
       tie-breaker derived from vertex index to ensure determinism without true randomness
       (or just strict LDF if random tie-breaking is risky for reproducibility, but 
       deterministic index-based tie-breaking is safer). Actually, standard LDF is 
       deterministic. We will use a dynamic score update.
    
    2. We perform several iterations (passes). In each pass:
       a. We assume the top-k vertices (by current score) are "selected" or "removed" from the graph.
       b. We update the scores of the remaining vertices. 
       c. The update logic: A vertex v benefits from having neighbors that have been removed.
          Why? If a neighbor u is removed (either selected for IS or blocked), it no longer 
          competes with v. However, in MIS greedy, if u is selected, v is blocked. 
          
          Let's refine the "Dynamic" aspect. 
          Standard Greedy: Pick u, remove N(u) and u. 
          This priority function is called ONCE. The decoder then runs the greedy loop.
          We cannot change the decoder. We must return a static array of scores that 
          guides the decoder to make good choices.
          
          Therefore, "Dynamic" here means we compute scores that *anticipate* the dynamic 
          process. 
          
          Key Insight: 
          In the greedy decoder, a vertex with high score is picked first. 
          If we pick a vertex, we eliminate its neighbors. 
          Vertices with many high-scoring neighbors are at risk of being eliminated early.
          Vertices with few high-scoring neighbors are "safe" to pick later? 
          No, we want to pick them *now* if they are good candidates, or ensure they 
          aren't blocked by a suboptimal pick.
          
          Alternative Dynamic Strategy: "Soft" LDF via Iterative Pruning Simulation
          We can simulate the greedy process internally to estimate the "survival" or 
          "value" of each vertex.
          
          Let's try a different angle: **Weighted Degree with Exponential Decay**.
          The current best uses linear degree. 
          What if we use a non-linear transformation of the degree to emphasize 
          extreme low-degree vertices (leaves) more strongly?
          Score = -exp(alpha * deg) ?
          
          Or, let's look at the "Independent Set" constraint more closely.
          The greedy algorithm picks the highest score. 
          If two vertices are adjacent, only one can be picked. 
          We want the *pair* of adjacent vertices to have a high score difference, 
          so the better one is picked first. 
          
          How do we define "better"? 
          Usually, lower degree is better because it blocks fewer others. 
          So, for an edge (u, v), we want score(u) > score(v) if deg(u) < deg(v).
          
          The current program does this: score = -deg + ...
          
          Let's try a strategy based on **Core Number** (k-core decomposition).
          Vertices with low core number are in the "periphery" of the graph.
          Periphery vertices are less connected and often form larger independent sets 
          when picked early.
          
          Algorithm:
          1. Compute k-core numbers for all vertices.
          2. Primary score: -core_number.
          3. Secondary score: -degree (to break ties within same core).
          
          Why? 
          The k-core number measures the "density" of the neighborhood. 
          A vertex in the 1-core is a leaf or in a tree-like structure. 
          A vertex in the 10-core is in a very dense cluster. 
          Picking vertices from the low cores first allows us to "peel" the graph, 
          potentially isolating the dense cores for later, where the choices are more critical 
          but the set of candidates is smaller.
          
          However, simply picking low core numbers might ignore degree. 
          Let's combine: Score = -core_number * 10 - degree.
          
          Let's check the computational cost. 
          K-core decomposition is O(N+E). For N=200, it's trivial.
          
          Let's try this.
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # 1. Compute Degrees
    deg = adj.sum(axis=1)
    
    # 2. Compute K-Core Numbers
    # We can implement a simple peeling algorithm for k-cores.
    # k-core number of a vertex is the largest k such that the vertex belongs to the k-core.
    
    # Initialize core numbers to 0
    core = np.zeros(n, dtype=int)
    # Copy degrees for the peeling process
    current_deg = deg.copy()
    
    # We need to efficiently find vertices with degree < k
    # Since n is small (200), we can use a simple loop or bucket sort.
    # Bucket sort by degree for efficiency, though n is small.
    
    max_deg = int(np.max(current_deg))
    
    # Buckets: list of lists of vertices
    buckets = [[] for _ in range(max_deg + 1)]
    for i in range(n):
        d = int(current_deg[i])
        if d <= max_deg:
            buckets[d].append(i)
        else:
            buckets[max_deg].append(i)
            
    # active flag
    active = np.ones(n, dtype=bool)
    
    # Peeling process
    # We iterate k from 0 upwards? No, we peel from lowest degree.
    # The standard algorithm:
    # 1. Put vertices in buckets by degree.
    # 2. Let k = 0.
    # 3. While there are vertices in bucket k:
    #    - Pop a vertex u from bucket k.
    #    - If current_deg[u] < k, continue (it was already processed or degree changed).
    #    - Actually, the standard linear time algorithm is slightly different.
    
    # Simpler O(N^2) approach for N=200:
    # Repeat N times:
    #   Find vertex with min current degree.
    #   Assign core number = current degree.
    #   Remove vertex and its neighbors' degrees.
    
    # This is O(N^2) which is 40,000 ops. Very fast.
    
    remaining_deg = deg.copy()
    removed = np.zeros(n, dtype=bool)
    
    for _ in range(n):
        if time.time() - start_time > 140:
             # Fallback
             return -deg
             
        # Find unremoved vertex with min degree
        # Mask removed vertices with inf
        masked_deg = np.where(removed, np.inf, remaining_deg)
        u = int(np.argmin(masked_deg))
        
        if np.isinf(masked_deg[u]):
            break
            
        core[u] = int(remaining_deg[u])
        removed[u] = True
        
        # Update degrees of neighbors of u
        neighbors = np.nonzero(adj[u] > 0)[0]
        remaining_deg[neighbors] -= 1
        
    # 3. Construct Score
    # We want to prioritize low core numbers.
    # Within same core number, prioritize low degree.
    # 
    # Scale: Core numbers can be up to N. Degree can be up to N.
    # Let's weight core number more heavily? 
    # If core is 0, it's a leaf. Very good to pick.
    # If core is 10, it's dense.
    # 
    # Score = -10 * core - deg
    # 
    # Let's check if this differs significantly from -deg.
    # Leaves have deg=1, core=0. Score = -1.
    # Hub has deg=10, core=10. Score = -100 - 10 = -110.
    # Normal node deg=5, core=2. Score = -20 - 5 = -25.
    # 
    # Comparison with current best: -deg + 0.6 * avg_neighbor_deg
    # For leaf: -1 + 0.6 * (deg_of_parent / 2). If parent deg=10, avg=5. Score = -1 + 3 = 2.
    # For hub: -10 + 0.6 * (avg_neighbor_deg). If neighbors are deg 10, avg=10. Score = -10 + 6 = -4.
    # 
    # The core-based score gives leaves a huge advantage (-1 vs -25 vs -110).
    # The current score gives leaves a small advantage (2 vs -4).
    # 
    # Is picking leaves *always* better?
    # In a star graph, picking the center (hub) yields IS size 1. Picking a leaf yields IS size N-1 (all other leaves).
    # So yes, picking low-degree/low-core vertices is crucial.
    # 
    # The core number captures the "structural depth" better than just degree?
    # In a cycle, all have deg=2, core=2. Scores are equal. Tie-break by index? 
    # In a cycle, any vertex is equally good.
    # 
    # In a "barbell" graph (two cliques connected by a path), the path vertices have low core, 
    # clique vertices have high core. 
    # Picking path vertices first is good because they are "cheaper" (lower degree) 
    # and don't block the clique vertices? 
    # Actually, if we pick a path vertex, we block its path neighbors. 
    # The clique vertices are not blocked. 
    # Then we can pick clique vertices? 
    # If we pick a clique vertex, we block all other clique vertices. 
    # So for a clique of size K, we can only pick 1. 
    # It doesn't matter which one we pick, as long as we pick one. 
    # But if we pick a path vertex, we might block a path vertex that connects to the clique? 
    # 
    # The core number seems like a strong signal for "peripherality".
    
    score = -10.0 * core - deg
    
    # Add a tiny tie-breaker based on index to ensure deterministic but varied ordering if scores are identical
    # This helps if there are many vertices with same core and degree.
    # We don't want to break ties randomly, but deterministically.
    # However, the greedy decoder sorts by score. If scores are equal, it picks in index order (usually).
    # We can explicitly add a tiny negative index component to make the order explicit in the score.
    # But wait, if we add -index/100000, it might interfere with the main score if scales are close.
    # Core and Deg are integers. Score is integer-ish.
    # Let's just return the score. The decoder's sort stability will handle ties.
    
    if time.time() - start_time > 145:
        return -deg
        
    return score
