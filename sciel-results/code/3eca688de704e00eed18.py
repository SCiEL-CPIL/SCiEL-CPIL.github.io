import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100 seconds out of 120s limit
    
    n = 16
    
    # Helper: Check if graph is triangle-free
    def is_triangle_free(adj):
        for u in range(n):
            neighbors = adj[u]
            # For each pair of neighbors, check if they are connected
            # This is O(n^3) in worst case if done naively, but n=16 is small.
            # Better: for each edge (u, v), check if adj[u] & adj[v] has any bit.
            # We iterate over edges.
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    if adj[u] & adj[v]:
                        return False
        return True

    # Helper: Check if graph is 3-colorable
    def is_3_colorable(adj):
        colors = [-1] * n
        
        def backtrack(u):
            if u == n:
                return True
            # Try colors 0, 1, 2
            for c in range(3):
                # Check if neighbor has color c
                conflict = False
                neighbors = adj[u]
                # Iterate neighbors
                # Optimization: if any neighbor has color c, conflict
                # We can just check: 
                # Actually, simpler: 
                # If (colors[neighbor] == c) for any neighbor, fail.
                # To speed up, we can precompute or just check.
                for v in range(n):
                    if neighbors & (1 << v):
                        if colors[v] == c:
                            conflict = True
                            break
                if not conflict:
                    colors[u] = c
                    if backtrack(u + 1):
                        return True
                    colors[u] = -1
            return False
        
        return backtrack(0)

    def get_edge_count(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Initialize with Groetzsch graph (11 vertices)
    # Vertices 0-4: Cycle
    # Vertices 5-9: Connected to cycle
    # Vertex 10: Connected to 5-9
    adj = [0] * n
    # Cycle 0-1-2-3-4-0
    cycle = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for u, v in cycle:
        add_edge(adj, u, v)
        
    # Mycielski step 1: vertices 5-9
    # 5 connected to 0, 1? No.
    # Standard Groetzsch:
    # V = {0,1,2,3,4, 5,6,7,8,9, 10}
    # Edges:
    # (i, i+1) for i in 0..4 (mod 5)
    # (i, 5+(i+1)%5) and (i+1, 5+i) ?
    # Let's stick to the seed's construction which is known to be correct.
    # Seed:
    # cycle = [(i, (i + 1) % 5) for i in range(5)]
    # edges = [list(e) for e in cycle]
    # for u, w in cycle:
    #     edges += [[u, 5 + w], [w, 5 + u]]
    # return edges + [[5 + u, 10] for u in range(5)]
    
    # Let's just use the seed edges to initialize adj
    edges = []
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, w in cycle:
        edges.append((u, w))
    for u, w in cycle:
        edges.append((u, 5 + w))
        edges.append((w, 5 + u))
    for u in range(5):
        edges.append((5 + u, 10))
        
    for u, v in edges:
        add_edge(adj, u, v)
        
    best_adj = adj[:]
    best_count = get_edge_count(adj)
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Local Search
    current_adj = adj[:]
    current_count = best_count
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add an edge
            # Pick random edge not in graph
            # Optimization: pick random u, v
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            if u > v: u, v = v, u
            
            if current_adj[u] & (1 << v):
                continue # Already exists
            
            # Check triangle free
            if current_adj[u] & current_adj[v]:
                continue # Would create triangle
            
            # Try adding
            add_edge(current_adj, u, v)
            new_count = current_count + 1
            
            if not is_3_colorable(current_adj):
                # Valid 4-chromatic
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_adj = current_adj[:]
            else:
                # Revert
                remove_edge(current_adj, u, v)
                # In SA, we might accept if T is high, but let's keep it simple hill climbing first
                # If we want to escape, we can accept with probability exp(-1/T)
                # But adding an edge that makes it 3-colorable is a "bad" move for our constraint
                # So we revert.
                
        elif move_type < 0.8:
            # Try to remove an edge (to allow adding others later)
            # Pick random existing edge
            u = random.randint(0, n - 1)
            # Find a neighbor
            neighbors = bin(current_adj[u]).count('1')
            if neighbors == 0: continue
            # Pick random bit
            # Simple way: pick random v and check if connected
            v = random.randint(0, n - 1)
            if v == u: continue
            if not (current_adj[u] & (1 << v)):
                continue
            
            remove_edge(current_adj, u, v)
            current_count -= 1
            
            # Removing an edge cannot make it 3-colorable if it wasn't before?
            # Actually, removing edges makes it *easier* to color.
            # So if it was 4-chromatic, it might become 3-colorable.
            # We want to maintain 4-chromatic.
            # So if it becomes 3-colorable, we should revert or accept with penalty.
            if is_3_colorable(current_adj):
                # It became 3-colorable. Bad.
                # Revert
                add_edge(current_adj, u, v)
                current_count += 1
            else:
                # Still 4-chromatic.
                # We have fewer edges. Accept?
                # Yes, to allow future additions.
                pass
                
        else:
            # Swap: Remove one, add one
            # Pick edge to remove
            u1 = random.randint(0, n - 1)
            v1 = random.randint(0, n - 1)
            if u1 == v1: continue
            if u1 > v1: u1, v1 = v1, u1
            if not (current_adj[u1] & (1 << v1)):
                continue
            
            # Pick edge to add
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2: continue
            if u2 > v2: u2, v2 = v2, u2
            if current_adj[u2] & (1 << v2):
                continue
            
            # Check triangle free for new edge in context of graph without e1
            # Temporarily remove e1
            remove_edge(current_adj, u1, v1)
            
            # Check if e2 can be added without triangle
            if current_adj[u2] & current_adj[v2]:
                # Conflict
                add_edge(current_adj, u1, v1)
                continue
            
            # Add e2
            add_edge(current_adj, u2, v2)
            
            # Count is same
            new_count = current_count
            
            if not is_3_colorable(current_adj):
                # Valid
                # Accept
                pass
            else:
                # Revert
                remove_edge(current_adj, u2, v2)
                add_edge(current_adj, u1, v1)
                
        # Cooling
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Convert best_adj back to list of edges
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                final_edges.append([u, v])
                
    return final_edges
