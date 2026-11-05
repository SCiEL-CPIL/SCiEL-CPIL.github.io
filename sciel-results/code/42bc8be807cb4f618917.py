import time
import random
import math

def is_triangle_free(adj, u, v):
    # Check if adding edge (u, v) creates a triangle
    # This is true if there is a common neighbor w
    # adj is a list of sets
    if u in adj[v] or v in adj[u]:
        return False # Already connected or self-loop, though we assume simple graph
    # Check common neighbors
    # Optimization: iterate over smaller set
    if len(adj[u]) < len(adj[v]):
        for w in adj[u]:
            if w in adj[v]:
                return False
    else:
        for w in adj[v]:
            if w in adj[u]:
                return False
    return True

def is_3_colorable(adj, n):
    # Backtracking 3-coloring
    colors = [-1] * n
    # Order vertices by degree (descending) to prune faster
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Check available colors
        used = set()
        for w in adj[v]:
            if colors[w] != -1:
                used.add(colors[w])
        for c in range(3):
            if c not in used:
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False
        
    return backtrack(0)

def construct():
    n = 16
    # Start with Grötzsch graph (11 vertices)
    # Vertices 0-4: C5
    # Vertices 5-9: C5
    # Vertex 10: connected to 5-9
    # Edges:
    edges = set()
    # C5 on 0-4
    for i in range(5):
        edges.add((i, (i+1)%5))
    # C5 on 5-9
    for i in range(5):
        edges.add((5+i, 5+(i+1)%5))
    # Connections between the two C5s in Grötzsch
    # Standard Grötzsch construction:
    # Let v_i be vertices of first C5. u_i be vertices of second C5.
    # Edges (v_i, u_{i+1}) and (v_i, u_{i-1})? 
    # Actually, Mycielski of C5:
    # Original C5: 0-1-2-3-4-0
    # New vertices 5-9 corresponding to 0-4
    # Edges (i, 5+i) for i in 0..4? No.
    # Mycielski:
    # V' = V u {v'_i for v_i in V} u {w}
    # E' = E u {(v_i, v'_j) | (v_i, v_j) in E} u {(v'_i, w) for all i}
    # Let's just use a known edge list for Grötzsch (11 vertices, 20 edges)
    # From literature:
    # 0-1, 1-2, 2-3, 3-4, 4-0 (C5)
    # 5-6, 6-7, 7-8, 8-9, 9-5 (C5)
    # 0-6, 1-7, 2-8, 3-9, 4-5 (cross)
    # 10 connected to 5,6,7,8,9
    gruetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 5),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    initial_edges = set()
    for u, v in gruetzsch_edges:
        initial_edges.add((min(u, v), max(u, v)))
        
    # Add vertices 11-15 as isolated initially
    # We will add edges to them during search
    
    adj = [set() for _ in range(n)]
    for u, v in initial_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    current_edges = initial_edges
    # Check if current is 3-colorable (Grötzsch is not)
    # It should be False
    if is_3_colorable(adj, n):
        # This shouldn't happen for Grötzsch, but just in case
        pass
        
    best_edges = set(current_edges)
    best_score = len(current_edges)
    
    start_time = time.time()
    max_time = 20.0 # Run for 20 seconds to allow deep search
    T = 10.0
    cooling = 0.999
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start_time < max_time:
        # Choose a move
        # 1. Add edge
        # 2. Remove edge
        # 3. Swap
        
        move_type = random.random()
        new_edges = set(current_edges)
        new_adj = [set(s) for s in adj]
        
        if move_type < 0.5:
            # Add edge
            # Pick random non-edge
            # Optimization: pick from list of non-edges
            # For simplicity, pick random pair, check if in set
            while True:
                u, v = random.randint(0, n-1), random.randint(0, n-1)
                if u == v: continue
                u, v = min(u, v), max(u, v)
                if (u, v) not in current_edges:
                    break
            if is_triangle_free(new_adj, u, v):
                new_edges.add((u, v))
                new_adj[u].add(v)
                new_adj[v].add(u)
            else:
                # Invalid move, reject
                continue
        elif move_type < 0.8:
            # Remove edge
            if not current_edges:
                continue
            u, v = random.choice(list(current_edges))
            new_edges.remove((u, v))
            new_adj[u].remove(v)
            new_adj[v].remove(u)
        else:
            # Swap: remove one, add one
            if not current_edges:
                continue
            u, v = random.choice(list(current_edges))
            new_edges.remove((u, v))
            new_adj[u].remove(v)
            new_adj[v].remove(u)
            
            # Try to add a different edge
            added = False
            for _ in range(10): # Try a few times to find a valid add
                x, y = random.randint(0, n-1), random.randint(0, n-1)
                if x == y: continue
                x, y = min(x, y), max(x, y)
                if (x, y) not in new_edges:
                    if is_triangle_free(new_adj, x, y):
                        new_edges.add((x, y))
                        new_adj[x].add(y)
                        new_adj[y].add(x)
                        added = True
                        break
            if not added:
                continue
                
        # Evaluate new state
        # Check 3-colorability
        if is_3_colorable(new_adj, n):
            new_score = len(new_edges) - 1000 # Penalty
        else:
            new_score = len(new_edges)
            
        current_score = len(current_edges) - (1000 if is_3_colorable(adj, n) else 0)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_edges = new_edges
            adj = new_adj
            if len(current_edges) > best_score and not is_3_colorable(adj, n):
                best_score = len(current_edges)
                best_edges = set(current_edges)
                
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return [[u, v] for u, v in best_edges]
