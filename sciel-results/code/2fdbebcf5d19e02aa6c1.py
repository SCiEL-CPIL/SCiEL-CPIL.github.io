import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Hub-Avoidance Initialization + Simulated Annealing Local Search
    1. Initialize scores using a heuristic that favors low-degree vertices and 
       vertices adjacent to high-degree hubs (to block the hubs).
    2. Use Simulated Annealing to optimize the vertex ordering.
    3. Use vectorized NumPy operations for fast greedy simulation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists as NumPy arrays for vectorized updates
    # neighbors_list[i] contains indices of neighbors of i
    neighbors_list = [np.where(adj[i])[0] for i in range(n)]
    
    # 1. Initial Heuristic: Hub-Avoidance Scoring
    # Degrees
    deg = adj.sum(axis=1).astype(float)
    
    # Score component 1: Low degree is good (inverse degree)
    # Use 1/(deg+1) to avoid division by zero
    inv_deg = 1.0 / (deg + 1.0)
    
    # Score component 2: Adjacency to hubs
    # Sum of inverse degrees of neighbors
    # If v is connected to a hub (high deg), 1/deg(hub) is small.
    # Wait, the hint says: "penalizing vertices that are adjacent to high-degree hubs 
    # to ensure they are deprioritized".
    # This contradicts my previous thought. Let's re-read carefully.
    # "explicitly penalizing vertices that are adjacent to high-degree hubs 
    # to ensure they are deprioritized and removed from the independent set, 
    # thereby preserving low-degree vertices that do not block high-value independent structures."
    
    # If I deprioritize v (connected to hub), v is picked late.
    # If v is picked late, it might be blocked by its neighbors (including the hub? No, hub is also low priority?).
    # If the hub is low priority, it is picked late.
    # If v is also low priority, it is picked late.
    # Who gets picked first?
    # The vertices with HIGH priority.
    # High priority vertices are those NOT connected to hubs?
    # If v is connected to a hub, and we penalize v, v has low priority.
    # The hub also has low priority (because it has high degree, and we generally penalize high degree).
    # So both are low priority.
    # The vertices with low degree and NOT connected to hubs get high priority.
    # These vertices are "safe" low-degree vertices.
    # They will be picked first.
    # This seems to align with "preserving low-degree vertices that do not block high-value independent structures".
    
    # So, Score = -deg(v) - penalty_for_hub_adjacency
    # Penalty = sum(1/deg(u) for u in N(v))?
    # If u is a hub, 1/deg(u) is small. So penalty is small?
    # That means we DON'T penalize much?
    # Maybe Penalty = sum(1/(1/deg(u)))? No.
    # Maybe Penalty = sum(deg(u))?
    # If v is connected to a hub, sum(deg(u)) is large.
    # So Score = -deg(v) - 0.1 * sum(deg(N(v)))
    
    # Let's try: Score = -deg(v) - 0.1 * (adj @ deg)
    neighbor_deg_sum = adj @ deg
    
    # Combine
    scores = -deg - 0.1 * neighbor_deg_sum
    
    # Normalize
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.zeros(n)
        
    # 2. Fast Simulation Function
    def simulate_fast(s):
        """
        Simulates the greedy decoder.
        Returns the size of the independent set.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count

    # 3. Simulated Annealing Local Search
    best_scores = scores.copy()
    best_val = simulate_fast(best_scores)
    current_scores = scores.copy()
    current_val = best_val
    
    rng = np.random.default_rng(42)
    
    # Temperature schedule
    T = 1.0
    cooling_rate = 0.999
    
    # Precompute a list of indices for random selection
    indices = np.arange(n)
    
    while time.time() < deadline:
        # Generate a random swap
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap scores
        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        # Evaluate
        new_val = simulate_fast(current_scores)
        
        # Acceptance criteria
        delta = current_val - new_val  # Positive if new is worse
        
        if new_val > current_val:
            # Accept improvement
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_scores = current_scores.copy()
        elif new_val == current_val:
            # Accept neutral move with some probability to explore
            if rng.random() < 0.5:
                current_val = new_val
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        else:
            # new_val < current_val
            # Accept with probability exp(-delta / T)
            # delta is positive (e.g., 1)
            # If T is high, exp(-1/T) is close to 1
            if rng.random() < np.exp(-delta / T):
                current_val = new_val
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                
        # Cool down
        T *= cooling_rate
        if T < 1e-4:
            T = 1e-4 # Floor to allow some exploration

    return best_scores
