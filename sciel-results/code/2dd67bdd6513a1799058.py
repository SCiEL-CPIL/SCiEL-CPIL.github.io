import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    Strategy:
    1. Start with a base score based on Lowest Degree (LDF).
    2. Simulate the greedy selection process using these scores.
    3. During simulation, track the "effective cost" of each pick: 
       the number of *currently available* neighbors that are blocked.
    4. Refine the scores: Vertices that had a low degree AND a low effective cost 
       in the simulation are given higher priority.
    5. This captures the dynamic interaction between vertices that static metrics miss.
    """
    start_time = time.time()
    
    # Convert to float array for fast vectorized operations
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # 1. Initial scores: Lowest Degree First (LDF)
    # Higher score = higher priority.
    # We want low degree to have high score.
    deg = adj.sum(axis=1)
    # Normalize degree to [0, 1] roughly for stable weighting
    max_deg = np.max(deg) if np.max(deg) > 0 else 1.0
    base_score = -deg / max_deg  # Range [-1, 0]
    
    # 2. Simulate greedy selection to calculate "effective blocking cost"
    # We will run one simulation pass to estimate the dynamic cost.
    
    # Current available mask
    available = np.ones(n, dtype=bool)
    
    # Sort vertices by base_score descending (i.e., lowest degree first)
    # We need to simulate the exact order the decoder will use.
    # The decoder picks highest score first.
    order = np.argsort(-base_score)
    
    # Track the number of available neighbors blocked by each selected vertex
    # cost[i] = number of available neighbors of i when i is selected
    effective_cost = np.zeros(n)
    
    # We don't need to simulate the *entire* set if we run out of time, 
    # but for n=200, this is very fast.
    for v in order:
        if not available[v]:
            continue
        
        # Count available neighbors of v
        # adj[v] is a vector of 1s for neighbors
        # available is a mask
        # Count of available neighbors = sum(adj[v] * available)
        # Since adj is float, we can use dot product or sum
        avail_nbrs = np.sum(adj[v] * available)
        
        # This is the "cost" of picking v: it removes v and its avail_nbrs from consideration
        effective_cost[v] = avail_nbrs
        
        # Mark v and its neighbors as unavailable for the simulation
        # We only need to update the 'available' mask for future iterations
        # To be efficient, we can just set available[v] = 0 and available[neighbors] = 0
        # But we need to be careful: we are simulating the *selection* order.
        # If we mark neighbors as unavailable, they won't be picked later in this sim.
        
        # Update available mask
        available[v] = False
        # Get indices of neighbors
        nbr_indices = np.where(adj[v] > 0)[0]
        available[nbr_indices] = False
        
        # Time check
        if time.time() - start_time > 100:
            break

    # 3. Refine scores
    # We want to prefer vertices with:
    # a) Low degree (static)
    # b) Low effective cost (dynamic: they didn't block many *available* neighbors)
    
    # Normalize effective cost
    max_cost = np.max(effective_cost) if np.max(effective_cost) > 0 else 1.0
    norm_cost = effective_cost / max_cost
    
    # Combine base score (negative degree) and negative cost
    # Weighting: Degree is the primary driver, but cost refines it.
    # If two vertices have similar degree, the one with lower blocking cost is better.
    # Let's use a weighted sum.
    # base_score is in [-1, 0]
    # norm_cost is in [0, 1]
    
    # Score = w1 * base_score - w2 * norm_cost
    # Since base_score is negative (lower degree is better), we add it.
    # Since norm_cost is positive (higher cost is worse), we subtract it.
    
    w1 = 1.0
    w2 = 0.5  # Secondary factor
    
    final_score = w1 * base_score - w2 * norm_cost
    
    # Safety deadline check
    if time.time() - start_time > 140:
        # Fallback to simple LDF
        return -deg
        
    return final_score
