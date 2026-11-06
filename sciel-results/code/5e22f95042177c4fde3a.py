import random
import time

def _compute_max_book(edges, n=40):
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v) if u < v else (v, u))
    
    max_book = 0
    for u, v in edge_set:
        # Number of common neighbors
        common = len(adj[u].intersection(adj[v]))
        if common > max_book:
            max_book = common
    return max_book

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial construction: K_{10,30} + random G_B
    # This gives a good starting point with lower theoretical max book than K_{9,31}
    # if we can control G_B degrees, but intra-B book size is 10.
    # We will use local search to reduce it.
    
    a_size = 10
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = []
    # Add all cross edges
    for u in A:
        for v in B:
            edges.append([u, v])
            
    num_cross = a_size * (n - a_size)
    edges_needed = 401 - num_cross
    
    b_indices = list(range(len(B)))
    b_edges = []
    
    # Generate G_B with low degree
    random.seed(123)
    possible_edges = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            possible_edges.append((i, j))
    random.shuffle(possible_edges)
    
    deg = [0] * len(B)
    max_deg_target = 8 # Try to keep degrees low
    
    for i, j in possible_edges:
        if len(b_edges) == edges_needed:
            break
        if deg[i] < max_deg_target and deg[j] < max_deg_target:
            b_edges.append((i, j))
            deg[i] += 1
            deg[j] += 1
            
    if len(b_edges) < edges_needed:
        for i, j in possible_edges:
            if len(b_edges) == edges_needed:
                break
            if (i, j) not in b_edges:
                b_edges.append((i, j))
                deg[i] += 1
                deg[j] += 1
                
    for i, j in b_edges:
        edges.append([B[i], B[j]])
        
    # Local Search to minimize max book
    # Representation: set of edges for fast lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v) if u < v else (v, u))
        
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
        
    def get_max_book():
        max_b = 0
        for u, v in edge_set:
            common = len(adj[u].intersection(adj[v]))
            if common > max_b:
                max_b = common
        return max_b
        
    current_max_book = get_max_book()
    best_edges = list(edge_set)
    
    # Simple local search: try to flip edges
    # We must maintain exactly 401 edges.
    # Flip: remove (u,v), add (x,y)
    
    # Candidate edges to add: non-edges
    # Candidate edges to remove: existing edges
    
    # To speed up, we only consider flips that might reduce the max book.
    
    while time.time() < deadline:
        # Pick a random edge to remove
        # Pick a random non-edge to add
        
        # Heuristic: Remove an edge that is in many triangles (high book)
        # Add an edge that creates few triangles
        
        # For simplicity, random restart or random moves
        # Let's try random moves first
        
        u_rem, v_rem = random.choice(list(edge_set))
        
        # Find a non-edge to add
        # Sample random pairs
        added = False
        for _ in range(100):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) not in edge_set:
                # Check if adding this edge increases max book too much?
                # Just try it.
                # Remove (u_rem, v_rem)
                edge_set.remove((u_rem, v_rem))
                adj[u_rem].discard(v_rem)
                adj[v_rem].discard(u_rem)
                
                # Add (x, y)
                edge_set.add((x, y))
                adj[x].add(y)
                adj[y].add(x)
                
                new_max_book = get_max_book()
                
                if new_max_book < current_max_book:
                    current_max_book = new_max_book
                    best_edges = list(edge_set)
                else:
                    # Revert
                    edge_set.remove((x, y))
                    adj[x].discard(y)
                    adj[y].discard(x)
                    
                    edge_set.add((u_rem, v_rem))
                    adj[u_rem].add(v_rem)
                    adj[v_rem].add(u_rem)
                
                added = True
                break
                
        if not added:
            continue
            
    # Convert best_edges back to list of lists
    final_edges = [[u, v] for u, v in best_edges]
    
    # Ensure exactly 401 edges
    if len(final_edges) != 401:
        # Fallback to initial construction if something went wrong
        return [[u, v] for u, v in edge_set]
        
    return final_edges
