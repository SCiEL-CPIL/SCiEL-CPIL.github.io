import random
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n = 16
    core_n = 11
    new_vertices = list(range(core_n, n))
    num_new = len(new_vertices)

    # 1. Build Grötzsch Graph (Vertices 0..10)
    # C5: 0-1-2-3-4-0
    # Mycielski:
    # Copies: 5..9 (5 is copy of 0, ..., 9 is copy of 4)
    # Top: 10
    # Edges:
    # C5 edges
    # For each edge (u,v) in C5, add (u, copy(v)) and (v, copy(u))
    # Top connected to all copies
    
    core_edges = []
    c5_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in c5_edges:
        core_edges.append([u, v])
    
    for u, v in c5_edges:
        core_edges.append([u, 5+v])
        core_edges.append([v, 5+u])
        
    for i in range(5):
        core_edges.append([10, 5+i])
        
    # Verify core is triangle-free and not 3-colorable (known to be true)
    # But let's double check triangle-free for safety
    adj_core = [0] * core_n
    for u, v in core_edges:
        adj_core[u] |= (1 << v)
        adj_core[v] |= (1 << u)
        
    for u in range(core_n):
        for v in range(u+1, core_n):
            if adj_core[u] & (1 << v):
                if adj_core[u] & adj_core[v]:
                    # Triangle found, error in construction
                    pass 

    # 2. Find all independent sets in the core
    # There are 2^11 = 2048 subsets.
    independent_sets = []
    for mask in range(1 << core_n):
        is_ind = True
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            if adj_core[u] & mask:
                is_ind = False
                break
            m ^= lsb
        if is_ind:
            # Store as bitmask for fast intersection checks
            independent_sets.append(mask)
            
    # Sort by size descending to prioritize larger sets
    independent_sets.sort(key=lambda x: -bin(x).count("1"))
    
    # We only need a good subset of independent sets for the search
    # Top 50 might be sufficient
    top_sets = independent_sets[:50]
    
    # 3. Search for optimal configuration of new vertices
    # State: list of 5 bitmasks (independent sets from core)
    # And a bitmask of edges between new vertices (10 possible edges)
    
    best_score = -1
    best_state = None
    
    rng = random.Random(42)
    
    def evaluate_state(sets, new_edges_mask):
        # sets: list of 5 bitmasks
        # new_edges_mask: 10-bit integer representing edges between new vertices
        # Edges between new vertices i, j correspond to bit i*5 - i*(i+1)/2 + j? 
        # Let's use a simple mapping: index k in 0..9 corresponds to pair (i,j)
        # Pairs: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        
        # Check validity of new_edges_mask given sets
        # If edge (i,j) exists, sets[i] & sets[j] must be 0
        # And the graph of new vertices must be triangle-free
        
        # Check triangle-free for new vertices
        # Build adjacency list for new vertices
        adj_new = [0] * num_new
        for k in range(10):
            if new_edges_mask & (1 << k):
                # Get i, j
                # Mapping logic
                # Let's precompute pairs
                pass
        
        # Precomputed pairs for 5 vertices
        pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        
        for k, (i, j) in enumerate(pairs):
            if new_edges_mask & (1 << k):
                # Check disjointness with core sets
                if sets[i] & sets[j]:
                    return -1 # Invalid: triangle via core
                adj_new[i] |= (1 << j)
                adj_new[j] |= (1 << i)
                
        # Check triangle-free in new vertices
        for u in range(num_new):
            for v in range(u+1, num_new):
                if adj_new[u] & (1 << v):
                    if adj_new[u] & adj_new[v]:
                        return -1 # Invalid: triangle in new vertices
                        
        # Calculate score
        score = len(core_edges) # 20
        for s in sets:
            score += bin(s).count("1")
        score += bin(new_edges_mask).count("1")
        
        return score

    # Local Search
    # Start with random assignment of top sets
    # And random valid edges between new vertices?
    # Or start with no edges between new vertices and add greedily?
    
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        # Random initialization
        current_sets = [rng.choice(top_sets) for _ in range(num_new)]
        
        # Initialize new_edges_mask to 0
        current_new_edges = 0
        
        # Try to add edges between new vertices greedily
        # We can just evaluate all possible new_edges masks? 2^10 = 1024.
        # For each candidate new_edges_mask, check validity and score.
        
        best_local_score = -1
        best_local_new_edges = 0
        
        for mask in range(1 << 10):
            s = evaluate_state(current_sets, mask)
            if s > best_local_score:
                best_local_score = s
                best_local_new_edges = mask
                
        if best_local_score > best_score:
            best_score = best_local_score
            best_state = (list(current_sets), best_local_new_edges)
            
        # Local search step: change one set
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(num_new):
                for new_set in top_sets:
                    if new_set == current_sets[i]:
                        continue
                    old_set = current_sets[i]
                    current_sets[i] = new_set
                    
                    # Re-optimize new_edges for this fixed set configuration
                    local_best_score = -1
                    local_best_mask = 0
                    for mask in range(1 << 10):
                        s = evaluate_state(current_sets, mask)
                        if s > local_best_score:
                            local_best_score = s
                            local_best_mask = mask
                            
                    if local_best_score > best_local_score:
                        best_local_score = local_best_score
                        best_local_new_edges = local_best_mask
                        improved = True
                        if best_local_score > best_score:
                            best_score = best_local_score
                            best_state = (list(current_sets), best_local_new_edges)
                    else:
                        current_sets[i] = old_set
                        
    if best_state is None:
        # Fallback: Return core edges only (valid, but low score)
        return core_edges
        
    sets, new_edges_mask = best_state
    
    # Construct final edges
    final_edges = list(core_edges)
    
    # Add edges from new vertices to core
    for i, v_new in enumerate(new_vertices):
        s = sets[i]
        m = s
        while m:
            lsb = m & -m
            v_core = lsb.bit_length() - 1
            final_edges.append([v_new, v_core])
            m ^= lsb
            
    # Add edges between new vertices
    pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    for k, (i, j) in enumerate(pairs):
        if new_edges_mask & (1 << k):
            u = new_vertices[i]
            v = new_vertices[j]
            final_edges.append([u, v])
            
    # Final Verification
    # 1. Triangle Free
    adj = [0] * n
    for u, v in final_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    is_tf = True
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                if adj[u] & adj[v]:
                    is_tf = False
                    break
        if not is_tf:
            break
            
    if not is_tf:
        # Should not happen if logic is correct, but fallback to core
        return core_edges
        
    # 2. Not 3-colorable
    # Since it contains Grötzsch, it should be. But let's verify with a quick check?
    # If the check is too slow, we trust the construction.
    # Given the time budget, we can run a simple backtracking 3-color check.
    
    def is_3_colorable():
        # Backtracking
        # Order vertices by degree
        degrees = [bin(adj[i]).count("1") for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = [False] * 3
            # Check neighbors
            neighbors = adj[v]
            m = neighbors
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if color[u] != -1:
                    used[color[u]] = True
                m ^= lsb
                
            for c in range(3):
                if not used[c]:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
            
        return backtrack(0)
        
    if is_3_colorable():
        # This is a critical failure. The graph is 3-colorable.
        # This shouldn't happen if Grötzsch is a subgraph.
        # If it does, we need to return a valid graph.
        # Fallback to Grötzsch + isolated vertices?
        return core_edges

    return final_edges
