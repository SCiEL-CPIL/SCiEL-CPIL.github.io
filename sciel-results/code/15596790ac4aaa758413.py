import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Identify valid vertices (no self-conflict)
    # Self conflict if 2*u is a perfect square
    valid_vertices = []
    for u in range(1, 257):
        s = 2 * u
        r = int(s**0.5)
        if r*r != s:
            valid_vertices.append(u)
            
    # Map valid vertices to 0..N-1 for bitmask indexing
    N = len(valid_vertices)
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    idx_to_v = [v for v in valid_vertices]
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask of neighbors of valid_vertices[i]
    adj = [0] * N
    
    # Precompute squares
    squares = set()
    k = 2
    while k*k <= 256 + 256:
        squares.add(k*k)
        k += 1
        
    for i, u in enumerate(valid_vertices):
        for sq in squares:
            v = sq - u
            if 1 <= v <= 256:
                if v in v_to_idx:
                    j = v_to_idx[v]
                    if i != j: # No self loops in this subgraph, but check just in case
                        adj[i] |= (1 << j)
                    
    # Helper: count bits
    def popcount(x):
        return x.bit_count()
        
    # Helper: greedy fill
    # Returns a mask of an independent set extending current_mask
    # It picks vertices that don't conflict with current_mask and each other
    def greedy_fill(current_mask, order):
        mask = current_mask
        # We need to find vertices that can be added.
        # A vertex v can be added if (adj[v] & mask) == 0.
        # We iterate through 'order' which is a list of indices.
        # To be fast, we can just iterate through all valid indices and check.
        # But 'order' allows us to prioritize low degree or random.
        for i in order:
            if (mask >> i) & 1: # Already in set
                continue
            if (adj[i] & mask) == 0:
                mask |= (1 << i)
        return mask

    # Initial greedy solution
    # Order by degree (ascending)
    degrees = [popcount(adj[i]) for i in range(N)]
    indices = list(range(N))
    indices.sort(key=lambda x: degrees[x])
    
    best_mask = greedy_fill(0, indices)
    best_size = popcount(best_mask)
    
    # Local Search with Kicks
    # We will run multiple restarts/kicks
    # Use a random order for kicks to explore different regions
    
    # Precompute a few static orders? Or generate random each time?
    # Random is better for diversity.
    
    # To speed up greedy_fill, we can pre-filter candidates.
    # But bitmask check is O(1) per vertex. N=245. 245 iterations per fill.
    # Very fast.
    
    num_iterations = 0
    while time.time() < deadline:
        num_iterations += 1
        
        # Kick: Remove some random vertices from best_mask
        # Number of kicks: 3 to 10
        kick_size = random.randint(3, 10)
        current_mask = best_mask
        
        # Remove kick_size vertices
        # Get list of set bits
        bits = []
        m = current_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bits.append(idx)
            m ^= lsb
            
        if len(bits) < kick_size:
            # If set is small, just remove all and restart
            kick_size = len(bits)
            
        if kick_size > 0:
            to_remove = random.sample(bits, kick_size)
            for idx in to_remove:
                current_mask &= ~(1 << idx)
        
        # Re-fill greedily with a random order
        order = list(range(N))
        random.shuffle(order)
        
        # Optional: Sort by degree in the remaining graph?
        # For now, random shuffle is a good mix.
        
        new_mask = greedy_fill(current_mask, order)
        
        new_size = popcount(new_mask)
        
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask
            
            # If we found a very large set, we can try to improve it further
            # by doing local swaps without kicking?
            # For now, just update best and continue kicking.
            
        # If we hit the theoretical max (unlikely to know exactly), break.
        # Theoretical max is N.
        
        # Check time frequently
        if time.time() > deadline:
            break
            
    # Convert best_mask back to list of values
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx_to_v[idx])
        m ^= lsb
        
    # Ensure non-empty
    if not result:
        return [1] # Fallback, though 1 is valid? 1+1=2 (not square). Yes.
        
    return result
# EVOLVE-BLOCK-END
