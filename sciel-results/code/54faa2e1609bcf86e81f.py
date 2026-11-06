import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145 # 5s margin
    
    n = 16
    
    # Grötzsch graph edges (0-10)
    groetzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (5,0), (5,2),
        (6,1), (6,3),
        (7,2), (7,4),
        (8,3), (8,0),
        (9,4), (9,1),
        (10,5), (10,6), (10,7), (10,8), (10,9)
    ]
    
    # Initialize base masks
    base_masks = [0] * n
    for u, v in groetzsch_edges:
        base_masks[u] |= (1 << v)
        base_masks[v] |= (1 << u)
        
    # Potential edges: all pairs not in base
    potential_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (base_masks[i] & (1 << j)) == 0:
                potential_edges.append((i, j))
                
    # Count base edges
    base_count = len(groetzsch_edges)
    
    best_count = base_count
    best_masks = base_masks[:]
    
    # Search
    while time.time() < deadline:
        cur_masks = best_masks[:]
        cur_count = best_count
        
        # Random order
        shuffled = potential_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            # Check if edge exists
            if (cur_masks[u] & (1 << v)) != 0:
                continue
            # Check triangle: common neighbors must be 0
            if (cur_masks[u] & cur_masks[v]) == 0:
                cur_masks[u] |= (1 << v)
                cur_masks[v] |= (1 << u)
                cur_count += 1
                
        if cur_count > best_count:
            best_count = cur_count
            best_masks = cur_masks[:]
            
        # Early exit if max possible reached
        if best_count == 64:
            break

    # Construct edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_masks[i] & (1 << j)):
                edges.append([i, j])
                
    return edges
