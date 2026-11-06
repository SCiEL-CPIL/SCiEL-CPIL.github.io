import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Generate the 37-point hexagon of radius 3
    # Points (a, b) such that hex_dist(a, b) <= 3
    # hex_dist = (|a| + |b| + |a+b|) // 2
    hex_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if (abs(a) + abs(b) + abs(a + b)) // 2 <= 3:
                hex_points.append((a, b))
    
    # There should be 37 points
    if len(hex_points) != 37:
        # Fallback to greedy if something goes wrong
        return _greedy_construct()

    # 2. Pre-compute adjacency and degrees
    point_to_idx = {p: i for i, p in enumerate(hex_points)}
    
    # Adjacency matrix or list
    adj = [[] for _ in range(37)]
    edges = []
    
    for i in range(37):
        a1, b1 = hex_points[i]
        for j in range(i + 1, 37):
            a2, b2 = hex_points[j]
            da = a1 - a2
            db = b1 - b2
            # d^2 = da^2 + db^2 + da*db
            if da*da + db*db + da*db == 1:
                adj[i].append(j)
                adj[j].append(i)
                edges.append((i, j))
                
    total_edges = len(edges)
    
    # 3. Search for the best 5 points to remove
    # We want to minimize the loss of edges.
    # Loss = sum(deg(v) for v in removed) - edges_within_removed
    # Max edges = total_edges - min_loss
    
    best_removed = None
    min_loss = float('inf')
    
    # To speed up, we can pre-compute edges between any pair
    # But 5-combinations of 37 is ~4.3M.
    # Each check involves summing 5 degrees and checking 10 pairs for edges.
    # This is fast in Python if optimized.
    
    # Pre-compute a set of edges for fast lookup
    edge_set = set()
    for i, j in edges:
        if i < j:
            edge_set.add((i, j))
        else:
            edge_set.add((j, i))
            
    # Pre-compute degrees
    degrees = [len(adj[i]) for i in range(37)]
    
    # Iterate combinations
    # 4.3M iterations.
    # In Python, this might take a few seconds.
    # 150s is plenty.
    
    for combo in itertools.combinations(range(37), 5):
        # Calculate loss
        loss = sum(degrees[i] for i in combo)
        
        # Subtract edges within combo
        # There are 10 pairs in 5 items
        for k in range(5):
            for l in range(k+1, 5):
                i = combo[k]
                j = combo[l]
                if i < j:
                    if (i, j) in edge_set:
                        loss -= 1
                else:
                    if (j, i) in edge_set:
                        loss -= 1
                        
        if loss < min_loss:
            min_loss = loss
            best_removed = combo
            
        # Time check
        if time.time() > deadline:
            break
            
    if best_removed is None:
        # Fallback
        return _greedy_construct()
        
    # Construct the result
    removed_set = set(best_removed)
    result_indices = [i for i in range(37) if i not in removed_set]
    
    # Verify count
    if len(result_indices) != 32:
        return _greedy_construct()
        
    result_points = [hex_points[i] for i in result_indices]
    
    # Convert to list of lists
    return [[a, b] for a, b in result_points]

def _greedy_construct():
    # Fallback greedy method from the original program
    n = 32
    pool = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            hex_dist = (abs(a) + abs(b) + abs(a + b)) // 2
            if hex_dist <= 3:
                pool.append((a, b))
                
    point_to_idx = {p: i for i, p in enumerate(pool)}
    
    def is_adjacent(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        da = a1 - a2
        db = b1 - b2
        return da*da + db*db + da*db == 1
        
    adj = {p: [] for p in pool}
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if is_adjacent(pool[i], pool[j]):
                adj[pool[i]].append(pool[j])
                adj[pool[j]].append(pool[i])
                
    selected = [(0, 0)]
    selected_set = {(0, 0)}
    
    while len(selected) < n:
        best_point = None
        best_edges = -1
        
        for p in pool:
            if p in selected_set:
                continue
            
            new_edges = 0
            for neighbor in adj[p]:
                if neighbor in selected_set:
                    new_edges += 1
            
            if new_edges > best_edges:
                best_edges = new_edges
                best_point = p
                
        if best_point is None:
            break
            
        selected.append(best_point)
        selected_set.add(best_point)
        
    return [[a, b] for a, b in selected]
