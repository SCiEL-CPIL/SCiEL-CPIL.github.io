import time
import sys

def construct():
    deadline = time.time() + 85
    N = 256
    
    # 1. Build Graph
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Initial Solution (Greedy)
    # Start with odds
    S = set(range(1, N + 1, 2))
    # Add evens if possible
    for e in range(2, N + 1, 2):
        ok = True
        for y in S:
            if (adj[e] >> y) & 1:
                ok = False
                break
        if ok:
            S.add(e)
            
    best_size = len(S)
    best_set = S

    # 3. Branch and Bound
    # We will search for a set larger than best_size
    
    # Helper: Greedy Coloring Upper Bound
    def color_bound(candidates):
        # Returns the number of colors needed for the subgraph induced by candidates
        # This is an upper bound on the size of the independent set in candidates
        colors = 0
        uncolored = candidates
        while uncolored:
            # Pick vertex with max degree in uncolored
            # To speed up, we can just pick the lowest index or use a heuristic
            # Let's use max degree for better bound
            v = -1
            max_deg = -1
            temp = uncolored
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                deg = (adj[idx] & uncolored).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                temp ^= lsb
            
            # Assign this color to v and all non-neighbors of v in uncolored
            # Greedy: pick vertices one by one that don't conflict with each other
            color_class = 0
            available = uncolored
            while available:
                # Pick vertex with max degree in available
                v2 = -1
                max_deg2 = -1
                temp2 = available
                while temp2:
                    lsb = temp2 & -temp2
                    idx2 = lsb.bit_length() - 1
                    deg2 = (adj[idx2] & available).bit_count()
                    if deg2 > max_deg2:
                        max_deg2 = deg2
                        v2 = idx2
                    temp2 ^= lsb
                
                color_class |= (1 << v2)
                # Remove v2 and its neighbors from available
                available &= ~adj[v2]
                available &= ~(1 << v2)
                
            uncolored &= ~color_class
            colors += 1
            if time.time() > deadline:
                return N # Return a large number to abort search
        return colors

    # Recursive DFS
    sys.setrecursionlimit(10000)
    
    def dfs(candidates, size, current_set_bits):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        
        # Prune
        # Upper bound: size + color_bound(candidates)
        # If size + popcount(candidates) <= best_size, prune immediately (cheap check)
        if size + candidates.bit_count() <= best_size:
            return
            
        # Check tighter bound
        ub = color_bound(candidates)
        if size + ub <= best_size:
            return
            
        if candidates == 0:
            if size > best_size:
                best_size = size
                # Reconstruct set from current_set_bits
                # This is expensive, so we only do it if we found a new best
                s = set()
                temp = current_set_bits
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    s.add(idx)
                    temp ^= lsb
                best_set = s
            return

        # Select a vertex to branch on
        # Heuristic: Pick vertex with max degree in candidates
        v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = (adj[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp ^= lsb
            
        # Branch 1: Include v
        # New candidates: remove v and its neighbors
        new_cands = candidates & ~adj[v] & ~(1 << v)
        dfs(new_cands, size + 1, current_set_bits | (1 << v))
        
        # Branch 2: Exclude v
        # New candidates: remove v
        new_cands = candidates & ~(1 << v)
        dfs(new_cands, size, current_set_bits)

    # Start DFS with all vertices
    # Note: We already have a good best_size from greedy.
    # We can start with all vertices as candidates.
    all_bits = (1 << (N + 1)) - 2 # Bits 1..N
    dfs(all_bits, 0, 0)
    
    return list(best_set)

construct()
