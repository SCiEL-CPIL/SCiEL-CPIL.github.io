import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    N = 256
    
    # Precompute the conflict graph
    # conflict[a][b] = True if (a*b) % (a+b) == 0, meaning a and b CANNOT both be in the set
    # We want a maximum independent set in this graph.
    
    # Bitmask representation for speed
    # 256 bits fits in 4 Python ints or we can use a single Python int (arbitrary precision)
    # Let's use Python's arbitrary precision integers as bitmasks.
    # Bit i represents number i+1 (so bit 0 is 1, bit 255 is 256)
    
    # conflict_mask[i] is a bitmask of all j such that (i+1, j+1) is a conflict
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                # a and b conflict
                # Set bit for b in a's mask, and bit for a in b's mask
                # Bit index for value v is v-1
                conflict_mask[a] |= (1 << (b - 1))
                conflict_mask[b] |= (1 << (a - 1))
    
    # Helper to check if adding x to set S (represented as bitmask) is valid
    # x is 1-indexed. S is a bitmask.
    # x is compatible with S if conflict_mask[x] & S == 0
    # Note: conflict_mask[x] has bits for other numbers. It doesn't include x itself.
    
    def can_add(x, S_mask):
        return (conflict_mask[x] & S_mask) == 0
    
    # Greedy heuristic: Minimum Degree
    # Start with empty set, repeatedly add the element with the fewest conflicts with remaining candidates?
    # Or start with all elements, repeatedly remove the one with the most conflicts?
    # Maximum Independent Set on a graph with 256 nodes.
    
    # Let's try a few strategies:
    # 1. Greedy Max Independent Set (remove highest degree)
    # 2. Greedy Add (add lowest degree compatible)
    # 3. Local Search (2-opt swaps)
    
    best_mask = 0
    best_size = 0
    
    def update_best(mask):
        nonlocal best_mask, best_size
        sz = mask.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Strategy 1: Greedy removal (Maximum Independent Set via Greedy Coloring/Removal)
    # Start with all nodes. While there are edges, remove the endpoint of an edge with max degree.
    # This gives an independent set (the complement of the removed set is NOT independent, the removed set is a vertex cover, so the rest is independent).
    # Wait, if we remove a vertex cover, the remaining graph has no edges, so it is an independent set.
    # Greedy vertex cover: while edge exists, pick vertex with max degree, remove it and its incident edges.
    # The set of removed vertices is a vertex cover. The set of remaining vertices is an independent set.
    # We want to maximize the independent set, so we want to MINIMIZE the vertex cover.
    # Greedy vertex cover usually gives a decent approximation.
    
    def greedy_vertex_cover():
        # Returns the independent set (complement of vertex cover)
        # Actually, let's just build the independent set directly.
        # Greedy Independent Set:
        # Candidates = all nodes
        # While candidates not empty:
        #   Pick node v with min degree in the induced subgraph on candidates?
        #   Add v to independent set
        #   Remove v and its neighbors from candidates
        
        candidates_mask = (1 << N) - 1  # All nodes
        independent_mask = 0
        
        # To pick min degree, we need to count bits in (conflict_mask[v] & candidates_mask)
        # This is O(N^2) per step if done naively, but N=256 is small.
        # However, we can optimize by just iterating.
        
        while candidates_mask:
            if time.time() > deadline - 5:
                break
            
            # Find node with minimum degree in candidates
            min_deg = N + 1
            best_v = -1
            
            # Iterate through set bits in candidates_mask
            c = candidates_mask
            while c:
                # Get lowest set bit
                lsb = c & (-c)
                v = lsb.bit_length()  # 1-indexed index (bit_length of 1<<k is k+1, so index is k, value is k+1)
                # Wait. 1 << 0 is 1. bit_length is 1. Index 0 is value 1.
                # So value = v.
                
                # Degree of v in candidates:
                deg = (conflict_mask[v] & candidates_mask).bit_count()
                
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                
                c &= c - 1  # Remove lowest set bit
            
            if best_v == -1:
                break
                
            # Add best_v to independent set
            independent_mask |= (1 << (best_v - 1))
            
            # Remove best_v and its neighbors from candidates
            # Neighbors are in conflict_mask[best_v]
            # We need to remove best_v and all neighbors from candidates_mask
            # candidates_mask &= ~(conflict_mask[best_v] | (1 << (best_v - 1)))
            
            remove_mask = conflict_mask[best_v] | (1 << (best_v - 1))
            candidates_mask &= ~remove_mask
            
        return independent_mask
    
    # Run greedy vertex cover a few times? It's deterministic.
    # Let's try randomizing the tie-breaking.
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        # Randomized Greedy Independent Set
        candidates_mask = (1 << N) - 1
        independent_mask = 0
        
        while candidates_mask:
            if time.time() > deadline - 5:
                break
            
            # Pick a random node? Or min degree with random tie-breaking?
            # Min degree is usually better for Max Independent Set.
            min_deg = N + 1
            candidates_for_best = []
            
            c = candidates_mask
            while c:
                lsb = c & (-c)
                v = lsb.bit_length()
                
                deg = (conflict_mask[v] & candidates_mask).bit_count()
                
                if deg < min_deg:
                    min_deg = deg
                    candidates_for_best = [v]
                elif deg == min_deg:
                    candidates_for_best.append(v)
                
                c &= c - 1
            
            if not candidates_for_best:
                break
                
            best_v = random.choice(candidates_for_best)
            
            independent_mask |= (1 << (best_v - 1))
            
            remove_mask = conflict_mask[best_v] | (1 << (best_v - 1))
            candidates_mask &= ~remove_mask
        
        update_best(independent_mask)
        
        # Local Search: 2-opt swaps
        # Try to swap a node in the set with a node outside to increase size?
        # Actually, just try to add nodes. If we can't add, try removing one and adding two?
        # This is complex. Let's stick to a simpler local search:
        # Try to add any compatible node. (Greedy already did this effectively).
        # Try to remove a node and see if we can add more than 1.
        
        # Let's try a "kick" phase:
        # Remove k random nodes, then re-greedy.
        if best_size < 150:  # Only do expensive kicks if we're far from potential optimum
             # This is a heuristic threshold.
             pass
        
        # Simpler: Try to improve current independent_mask
        # If we have time, do a local search on the current best.
        
    # Let's try a more systematic local search on the best found so far
    # Start from best_mask
    current_mask = best_mask
    current_size = best_size
    
    # Try to add any node
    # If no node can be added, try removing one and adding two.
    # This is a 2-exchange local search.
    
    # Precompute list of nodes not in current_mask
    # This is expensive to recompute.
    
    # Let's do a few rounds of local search
    for ls_round in range(50):
        if time.time() > deadline - 2:
            break
            
        improved = False
        
        # 1. Try to add a single node
        # Find all nodes not in current_mask
        # candidates = ((1 << N) - 1) ^ current_mask
        # for v in candidates:
        #   if can_add(v, current_mask):
        #       current_mask |= (1 << (v-1))
        #       improved = True
        
        # This is O(N^2) per round if we check all.
        # But we can just iterate.
        
        c = ((1 << N) - 1) ^ current_mask
        while c:
            lsb = c & (-c)
            v = lsb.bit_length()
            if can_add(v, current_mask):
                current_mask |= lsb
                improved = True
            c &= c - 1
            
        if improved:
            update_best(current_mask)
            continue # Restart local search with new state
        
        # 2. Try 1-removal, 2-addition (or just 1-addition after removal)
        # For each v in current_mask:
        #   temp_mask = current_mask ^ (1 << (v-1))
        #   Try to add nodes to temp_mask greedily
        #   If size(temp_mask) > size(current_mask), accept
        
        # This is expensive.
        # Let's try a limited version:
        # Pick a random v from current_mask
        # Remove it
        # Try to add ALL possible nodes
        # If size increased, keep
        
        if current_size == 0:
            break
            
        # Pick random v
        # Get list of bits in current_mask
        # Or just pick a random index 1..N and check if it's in mask
        # Better: iterate through bits
        
        c = current_mask
        # Shuffle bits? Python doesn't have easy shuffle for bits.
        # Let's just pick a random node from 1..N and if it's in mask, use it.
        # Or iterate through a random subset of nodes in mask.
        
        # Let's just iterate through all nodes in mask, but randomly pick one to try.
        # To avoid O(N^2) per step, we can just try a few.
        
        tried = 0
        while tried < 10:
            v = random.randint(1, N)
            if not (current_mask & (1 << (v-1))):
                continue
                
            tried += 1
            
            temp_mask = current_mask ^ (1 << (v-1))
            
            # Greedy add to temp_mask
            c2 = ((1 << N) - 1) ^ temp_mask
            added = 0
            while c2:
                lsb = c2 & (-c2)
                w = lsb.bit_length()
                if can_add(w, temp_mask):
                    temp_mask |= lsb
                    added += 1
                c2 &= c2 - 1
                
            new_size = temp_mask.bit_count()
            if new_size > current_size:
                current_mask = temp_mask
                current_size = new_size
                update_best(current_mask)
                improved = True
                break
                
        if improved:
            continue
            
        # If no improvement after trying single adds and some 1-removals, break
        # To avoid getting stuck in a local minimum, we might need to accept neutral moves or do random kicks.
        # For now, let's just break and rely on the restart loop.
        break

    # Final Polish:
    # Try to add any remaining nodes
    c = ((1 << N) - 1) ^ best_mask
    while c:
        lsb = c & (-c)
        v = lsb.bit_length()
        if can_add(v, best_mask):
            best_mask |= lsb
        c &= c - 1
        
    update_best(best_mask)
    
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
            
    return result
