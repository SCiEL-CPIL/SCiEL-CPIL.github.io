import numpy as np
import time

def _greedy_mis(adjacency, order):
    """Run greedy MIS with given vertex order (descending priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency[v]] = False
    return count

def _fast_greedy_size(adjacency, available_mask):
    """
    Estimate MIS size of the subgraph defined by available_mask 
    using a simple Min-Degree heuristic (vectorized-ish loop).
    This is used for the lookahead step.
    """
    n = adjacency.shape[0]
    # Get current degrees in the subgraph
    # deg[v] = sum(adjacency[v, available_mask])
    # We need to simulate the greedy process on the subgraph.
    # Since n=200, a simple loop is fast enough.
    
    avail = available_mask.copy()
    count = 0
    
    # Precompute degrees for the available nodes?
    # Degrees change as we pick nodes.
    
    # Optimization: Instead of full re-calculation, we can just run the standard greedy
    # on the available nodes, sorted by current degree.
    # But current degree changes.
    
    # Simple approach:
    # 1. Get indices of available nodes.
    # 2. Sort them by their degree in the *current* subgraph.
    # 3. Pick first, remove neighbors, repeat.
    
    # For lookahead, we just need a good estimate, not exact.
    # A simpler estimate: Greedy Min-Degree on the subgraph.
    
    # Let's just run the exact greedy on the subgraph using the same logic 
    # but restricted to available nodes.
    # To make it fast, we'll just iterate through nodes in order of 
    # their original degree (static) or just arbitrary order?
    # Arbitrary order (0..N) is bad.
    # Let's use the static degree order for the subgraph.
    
    # Get current degrees
    # This is expensive if done from scratch for every v in lookahead.
    # But N=200. 200 * 200 * 200 = 8M ops. OK.
    
    # Let's optimize the inner loop.
    # We will sort the available nodes by their degree in the subgraph.
    # deg_sub[v] = np.sum(adjacency[v] & avail)
    
    # This calculation is O(N^2) per lookahead step.
    # Total O(N^3) = 8M.
    
    # Actually, we can just use the static degree of the original graph 
    # restricted to available nodes as a proxy for the subgraph degree order.
    # It's an approximation, but fast.
    
    # Let's try to be slightly smarter:
    # 1. Compute degrees of available nodes in the subgraph.
    # 2. Sort available nodes by these degrees.
    # 3. Greedy pick.
    
    avail_indices = np.where(avail)[0]
    if len(avail_indices) == 0:
        return 0
        
    # Compute degrees in subgraph
    # sub_adj = adjacency[np.ix_(avail_indices, avail_indices)]
    # degs = sub_adj.sum(axis=1)
    
    # Vectorized degree calculation for available nodes
    # degs = np.sum(adjacency[:, avail] & avail, axis=1) # This gives deg for all nodes, mask later
    # Actually:
    degs = np.sum(adjacency[:, avail], axis=1)
    degs = degs[avail_indices]
    
    # Sort available indices by degree (ascending)
    sorted_indices = avail_indices[np.argsort(degs)]
    
    # Greedy simulation on this order
    avail_sim = np.zeros(n, dtype=bool)
    avail_sim[avail] = True
    
    cnt = 0
    for v in sorted_indices:
        if avail_sim[v]:
            cnt += 1
            avail_sim[v] = False
            avail_sim[adjacency[v]] = False
            
    return cnt

def priority(adjacency):
    n = adjacency.shape[0]
    deadline = time.time() + 110 # Safety margin
    
    degrees = adjacency.sum(axis=1)
    
    # Candidate 1: Min Degree (Standard)
    score_min_deg = -degrees.astype(float)
    
    # Candidate 2: Min Degree + Neighbor Sum
    # ND(v) = sum(degrees of neighbors)
    # This is (adjacency @ degrees)
    neighbor_deg_sum = adjacency.dot(degrees)
    
    # We want to pick v if it removes high-degree neighbors.
    # So we add neighbor_deg_sum to the score (since we want high score to be picked first).
    # Score = -deg(v) + lambda * ND(v)
    
    candidates = []
    
    # Add Min Degree
    candidates.append(score_min_deg)
    
    # Add variations of the Neighbor Sum heuristic
    for lam in [0.1, 0.5, 1.0, 2.0]:
        score = -degrees.astype(float) + lam * neighbor_deg_sum
        candidates.append(score)
        
    # Add variations that penalize neighbor sum (prefer isolated low-deg nodes)
    for lam in [-0.5, -1.0]:
        score = -degrees.astype(float) + lam * neighbor_deg_sum
        candidates.append(score)

    # Candidate: 1-Step Lookahead
    # Score(v) = 1 + Estimated_MIS_Size(G \ {v} \ N(v))
    # We use _fast_greedy_size for the estimate.
    
    if time.time() < deadline - 10:
        lookahead_scores = np.zeros(n)
        for v in range(n):
            # Create mask for available nodes after picking v
            mask = np.ones(n, dtype=bool)
            mask[v] = False
            mask[adjacency[v]] = False
            
            # Estimate size
            est_size = _fast_greedy_size(adjacency, mask)
            lookahead_scores[v] = 1 + est_size
            
        candidates.append(lookahead_scores)
        
        # Also try a mix of Lookahead and Min Degree
        # Normalize lookahead scores to be comparable? 
        # Lookahead scores are ~ N/2. Min Degree scores are ~ -N.
        # We can just evaluate them separately.
        
        # Try a hybrid: Rank based on Lookahead, break ties by Min Degree?
        # Or just add them? Scales are different.
        # Let's just keep them as separate candidates.

    # Evaluate all candidates
    best_size = -1
    best_score = candidates[0]
    
    for score in candidates:
        if time.time() > deadline - 5:
            break
            
        # Sort vertices by descending score
        # np.argsort returns indices that would sort the array ascending.
        # We want descending, so argsort(-score) or argsort(score)[::-1]
        order = np.argsort(-score)
        
        size = _greedy_mis(adjacency, order)
        
        if size > best_size:
            best_size = size
            best_score = score

    return best_score
