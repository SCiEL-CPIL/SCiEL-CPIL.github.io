import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 256
    # Squares up to 512
    squares = set()
    k = 2
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask where bit j is set if i and j conflict
    # We use 0-indexed internally, so vertex i is i-1
    adj = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if a + b in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        # Check diagonal (a+a)
        if a + a in squares:
            # This vertex cannot be in the set at all? 
            # The condition is "a+b is never a perfect square for a<=b in the set, INCLUDING a=b".
            # If a+a is a square, then a cannot be in the set because pair (a,a) violates the condition.
            # So we can mark such vertices as "forbidden".
            pass 

    # Identify forbidden vertices (where 2*a is a square)
    forbidden_mask = 0
    for i in range(N):
        a = i + 1
        if a + a in squares:
            forbidden_mask |= (1 << i)
    
    # Valid vertices are those not forbidden
    valid_mask = ((1 << N) - 1) ^ forbidden_mask
    
    # Greedy Initialization
    def greedy_init(order):
        current_set = 0
        # order is a list of indices to consider
        for i in order:
            if not (valid_mask & (1 << i)):
                continue
            # Check if i conflicts with any vertex in current_set
            # neighbors of i in current_set
            if (adj[i] & current_set) == 0:
                current_set |= (1 << i)
        return current_set
    
    best_set = 0
    best_size = 0
    
    # Try multiple random orderings
    for _ in range(20):
        order = list(range(N))
        random.shuffle(order)
        s = greedy_init(order)
        size = bin(s).count("1")
        if size > best_size:
            best_size = size
            best_set = s
    
    # Also try degree-based ordering
    degrees = [bin(adj[i]).count("1") for i in range(N)]
    order_deg = sorted(range(N), key=lambda x: degrees[x])
    s = greedy_init(order_deg)
    size = bin(s).count("1")
    if size > best_size:
        best_size = size
        best_set = s

    # Local Search: Hill Climbing
    # Move: Remove one vertex from set, then greedily add any possible vertices.
    # We repeat this until no improvement or time runs out.
    
    current_set = best_set
    current_size = best_size
    
    # Helper to get list of indices from bitmask
    def bits_to_list(mask):
        return [i for i in range(N) if (mask & (1 << i))]
    
    # Helper to greedily add vertices to a set given a starting mask
    def greedy_add(start_mask, available_mask):
        # available_mask contains vertices not in start_mask and not forbidden
        # We want to add vertices from available_mask that don't conflict with start_mask
        # and don't conflict with each other (greedy)
        result = start_mask
        # We can just iterate through available bits
        # To make it faster, we can pick the one with fewest conflicts or just random
        # Let's just iterate through all available bits and add if possible
        # This is O(N^2) in worst case but N=256 is small
        avail = available_mask
        while avail:
            # Pick a random bit from avail? Or lowest?
            # Lowest is deterministic. Let's use random for diversity in restarts?
            # For hill climbing step, deterministic might be better to ensure consistency
            lsb = avail & -avail
            i = lsb.bit_length() - 1
            avail ^= lsb
            
            if (adj[i] & result) == 0:
                result |= lsb
        return result

    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_list = bits_to_list(current_set)
        
        for v in current_list:
            # Remove v
            new_set = current_set & ~(1 << v)
            
            # Find vertices that can be added
            # A vertex u can be added if u is not in new_set, u is valid, and (adj[u] & new_set) == 0
            # We can compute this mask
            # Let's just iterate through all valid vertices not in new_set
            addable_mask = 0
            for u in range(N):
                if (new_set & (1 << u)) == 0 and (valid_mask & (1 << u)):
                    if (adj[u] & new_set) == 0:
                        addable_mask |= (1 << u)
                        
            # Greedily add from addable_mask
            final_set = greedy_add(new_set, addable_mask)
            new_size = bin(final_set).count("1")
            
            if new_size > current_size:
                current_set = final_set
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                improved = True
                break # Restart loop with new set
        
        # If no improvement found in this pass, try a random swap or just stop?
        # To escape local optima, we can try a random move that decreases size slightly
        # But for now, simple hill climbing is a good start.
        # Let's add a random restart if stuck
        if not improved:
             # Try a few random restarts of local search from current best
             # Or just break if time is tight
             if time.time() > deadline - 5:
                 break
             # Random perturbation
             if current_size > 0:
                 v = random.choice(bits_to_list(current_set))
                 new_set = current_set & ~(1 << v)
                 # Add a random valid vertex that doesn't conflict
                 candidates = []
                 for u in range(N):
                     if (new_set & (1 << u)) == 0 and (valid_mask & (1 << u)):
                         if (adj[u] & new_set) == 0:
                             candidates.append(u)
                 if candidates:
                     u = random.choice(candidates)
                     new_set |= (1 << u)
                     # Greedily fill
                     addable_mask = 0
                     for w in range(N):
                         if (new_set & (1 << w)) == 0 and (valid_mask & (1 << w)):
                             if (adj[w] & new_set) == 0:
                                 addable_mask |= (1 << w)
                     new_set = greedy_add(new_set, addable_mask)
                     new_size = bin(new_set).count("1")
                     if new_size > current_size:
                         current_set = new_set
                         current_size = new_size
                         if current_size > best_size:
                             best_size = current_size
                             best_set = current_set
                         improved = True

    # Convert best_set bitmask to list of integers (1-indexed)
    result = []
    for i in range(N):
        if best_set & (1 << i):
            result.append(i + 1)
            
    return result
