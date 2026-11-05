import random
import time

def construct():
    d = 7
    n = 2 ** d
    total_edges = n * d // 2
    
    # Precompute all edges and their associated square conflicts
    # Edge representation: (u, v) with u < v
    # For each edge, store the list of tuples of the other 3 edges in the squares it belongs to.
    # Actually, we just need to check if the other 3 are present.
    
    edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_index[(u, v)] = len(edges) - 1
    
    # For each edge, precompute the "conflict conditions".
    # An edge e=(u,v) differing in bit i.
    # For each j != i, the square is formed by bits i and j.
    # The other edges are:
    # e1: (u, u^ (1<<j))
    # e2: (v, v^ (1<<j))
    # e3: (u^ (1<<j), v^ (1<<j))
    # We need to ensure NOT (e1 in S AND e2 in S AND e3 in S).
    
    # Store for each edge index, a list of lists of indices of the other 3 edges.
    # conflicts[edge_idx] = [ [idx1, idx2, idx3], ... for each j ]
    conflicts = [[] for _ in range(len(edges))]
    
    for idx, (u, v) in enumerate(edges):
        # Determine which bit is flipped
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            # Generate the other 3 edges
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_index[e1]
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            
            conflicts[idx].append([idx1, idx2, idx3])

    best_count = 0
    best_edges = []
    
    # Time limit 120s, let's use 110s to be safe
    start_time = time.time()
    time_limit = 110
    
    # Randomized Greedy with Restarts
    # We can also try to bias the random shuffle or use a specific order.
    # Standard shuffle is a good baseline.
    
    while time.time() - start_time < time_limit:
        random.shuffle(edges)
        
        selected = [False] * len(edges)
        count = 0
        
        # Map from edge tuple to index for quick lookup? 
        # We are iterating over shuffled edge indices.
        # We need to check if adding edge 'idx' creates a C4.
        # This requires checking if for ANY square associated with 'idx',
        # the other 3 edges are already selected.
        
        for idx in range(len(edges)):
            # Check if this edge can be added
            can_add = True
            for square_conflict in conflicts[idx]:
                if selected[square_conflict[0]] and selected[square_conflict[1]] and selected[square_conflict[2]]:
                    can_add = False
                    break
            
            if can_add:
                selected[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            # Reconstruct edge list
            best_edges = [list(edges[i]) for i in range(len(edges)) if selected[i]]
            
            # If we hit the theoretical upper bound or a very high number, we might stop early?
            # 308 is upper bound. 304 is known. If we get 305+, it's great.
            if best_count >= 305:
                # We can keep searching for a bit longer to see if we can go higher, 
                # but 304 is the target. Let's not stop immediately to try for 305+.
                pass

    return best_edges
