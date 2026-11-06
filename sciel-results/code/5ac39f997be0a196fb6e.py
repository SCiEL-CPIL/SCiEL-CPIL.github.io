import time
import random
import numpy as np

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 115.0
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute neighborhood masks (self + neighbors) for fast blocking updates
    # neighborhood_mask[v] is a Python int bitmask
    neighborhood_mask = [0] * MAX_V
    for v in range(MAX_V):
        mask = 1 << v
        for nb in neighbors[v]:
            mask |= (1 << nb)
        neighborhood_mask[v] = mask

    best_path = []
    best_len = 0
    
    # block_count[v] is the number of used vertices u such that dist(u, v) <= 1
    # This allows O(1) validity check: candidate w is valid if used_mask doesn't have w
    # and block_count[w] == 1 (only blocked by the current last vertex).
    # Wait, if w is a neighbor of current, current contributes 1.
    # If w is not used, and no other used vertex is neighbor of w, then block_count[w] is 1.
    
    def dfs(path, used_mask, block_count, time_limit):
        nonlocal best_path, best_len
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        if time.time() > time_limit:
            return
            
        current = path[-1]
        
        # Find candidates
        candidates = []
        for nb in neighbors[current]:
            # Check if nb is already used
            if (used_mask >> nb) & 1:
                continue
            # Check if nb is blocked by any other vertex
            # block_count[nb] includes 'current' (dist 1) and any other used vertex with dist <= 1.
            # If block_count[nb] == 1, it means only 'current' is blocking it (dist 1).
            # Any other used vertex would make it >= 2 (either dist 1 or dist 0 if nb was used, but we checked used).
            if block_count[nb] == 1:
                candidates.append(nb)
                
        if not candidates:
            return
            
        # Heuristic: Sort candidates
        # We want to pick the candidate that leaves the most options? 
        # Or the one that is most constrained?
        # Let's try: Score candidate nb by the number of its neighbors that are still "available" (block_count == 0).
        # If we pick nb, its neighbors will have block_count increased by 1.
        # A good move is one that doesn't kill too many future options.
        
        scored_candidates = []
        for nb in candidates:
            # Count how many neighbors of nb are currently "free" (block_count == 0)
            # Note: block_count is for the CURRENT state.
            # If we add nb, the block_count of nb's neighbors will increase.
            # We want to avoid nb if it blocks many "free" vertices?
            # Actually, we want to pick nb that has many *potential* next steps.
            # Potential next steps for nb are its neighbors x where (used_mask >> x) & 1 == 0 and block_count[x] == 0 (before adding nb).
            # After adding nb, block_count[x] becomes 1.
            # So the number of next steps from nb is the count of neighbors x of nb such that x is not used and block_count[x] == 0.
            
            next_steps = 0
            for nn in neighbors[nb]:
                if (used_mask >> nn) & 1:
                    continue
                if block_count[nn] == 0:
                    next_steps += 1
            scored_candidates.append((next_steps, nb))
            
        # Sort by next_steps descending (most flexible first) or ascending (most constrained first)?
        # Warnsdorff says most constrained (ascending).
        # But for snakes, sometimes you need to keep options open.
        # Let's try a mix: Randomly select from top K most flexible, or most constrained?
        # Let's try "Most Flexible" (descending) to avoid dead ends early.
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Randomize to escape local minima
        # Pick from the top 3 or 5
        k = min(3, len(scored_candidates))
        chosen_idx = random.randint(0, k - 1)
        chosen = scored_candidates[chosen_idx][1]
        
        # Apply move
        path.append(chosen)
        used_mask |= (1 << chosen)
        
        # Update block_count
        # Add chosen's neighborhood to block_count
        # We can't just OR the mask because we need counts.
        # We iterate through the neighbors of chosen and chosen itself.
        for u in [chosen] + neighbors[chosen]:
            block_count[u] += 1
            
        # Recurse
        dfs(path, used_mask, block_count, time_limit)
        
        # Backtrack
        path.pop()
        used_mask ^= (1 << chosen)
        for u in [chosen] + neighbors[chosen]:
            block_count[u] -= 1

    # Main loop: Multiple restarts
    random.seed(42)
    
    # We will run DFS with a time slice for each restart.
    # If one restart gets stuck, we abandon it and start a new one.
    # We keep the global best.
    
    while time.time() < deadline - 5:
        start = random.randint(0, MAX_V - 1)
        
        # Initialize state
        path = [start]
        used_mask = 1 << start
        block_count = [0] * MAX_V
        
        # Initialize block_count for start
        for u in [start] + neighbors[start]:
            block_count[u] += 1
            
        # Time slice for this DFS run
        # We give it a chunk of time, e.g., 5 seconds.
        # But if it finds a very long path, we might want to continue?
        # No, DFS backtracking is expensive. Better to restart with different randomness.
        slice_time = time.time() + 5.0
        
        # Add some randomness to the heuristic selection by shuffling candidates if ties?
        # The random.choice in the DFS handles this.
        
        dfs(path, used_mask, block_count, slice_time)
        
        # Update global best if this run was better
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        # Small sleep to allow other processes? No, single thread.
        
    if not best_path:
        return [0]
        
    return best_path
