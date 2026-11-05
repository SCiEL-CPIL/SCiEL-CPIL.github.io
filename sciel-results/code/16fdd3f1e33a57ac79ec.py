import random
import itertools

def is_k4_free(adj):
    """
    Check if the graph defined by adjacency matrix is K4-free.
    adj is a list of lists or numpy array.
    """
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                for w in range(v + 1, n):
                    if adj[u][w] and adj[v][w]:
                        # Triangle u-v-w
                        for x in range(w + 1, n):
                            if adj[u][x] and adj[v][x] and adj[w][x]:
                                return False
    return True

def max_triangle_free_size(adj):
    """
    Find the size of the largest induced subgraph that is triangle-free.
    """
    n = len(adj)
    # We want to find the max k such that there exists a subset S of size k
    # where the induced subgraph on S has no triangles.
    
    # Heuristic: Check subsets of size k from n down to 1
    # For n=18, checking all subsets of size 12 is ~18k.
    # Checking all subsets of size 11 is ~31k.
    # Total is manageable.
    
    for k in range(n, 0, -1):
        for subset in itertools.combinations(range(n), k):
            # Check if induced subgraph on subset is triangle-free
            is_tf = True
            for i in range(len(subset)):
                u = subset[i]
                for j in range(i + 1, len(subset)):
                    v = subset[j]
                    if adj[u][v]:
                        for l in range(j + 1, len(subset)):
                            w = subset[l]
                            if adj[u][w] and adj[v][w]:
                                is_tf = False
                                break
                        if not is_tf:
                            break
                    if not is_tf:
                        break
                if not is_tf:
                    break
            if is_tf:
                return k
    return 0

def construct():
    n = 18
    # Initialize T_3(18)
    part_size = 6
    parts = [
        list(range(0, part_size)),
        list(range(part_size, 2 * part_size)),
        list(range(2 * part_size, 3 * part_size))
    ]
    
    adj = [[False] * n for _ in range(n)]
    
    # Add edges between parts
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u][v] = True
                    adj[v][u] = True
                    
    current_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                current_edges.append([u, v])
                
    current_score = max_triangle_free_size(adj)
    best_score = current_score
    best_edges = current_edges[:]
    
    # Local Search
    # Time limit approx 100 seconds
    import time
    start_time = time.time()
    limit = 100
    
    while time.time() - start_time < limit:
        # Generate a random move
        move_type = random.choice(['add_internal', 'remove_cross'])
        
        if move_type == 'add_internal':
            part_idx = random.randint(0, 2)
            part = parts[part_idx]
            u, v = random.sample(part, 2)
            if not adj[u][v]:
                # Try move
                adj[u][v] = True
                adj[v][u] = True
                if is_k4_free(adj):
                    score = max_triangle_free_size(adj)
                    if score < best_score:
                        best_score = score
                        best_edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i][j]:
                                    best_edges.append([i, j])
                else:
                    # Revert
                    adj[u][v] = False
                    adj[v][u] = False
        else:
            # Remove cross edge
            i, j = random.sample(range(3), 2)
            u = random.choice(parts[i])
            v = random.choice(parts[j])
            if adj[u][v]:
                # Try move
                adj[u][v] = False
                adj[v][u] = False
                if is_k4_free(adj):
                    score = max_triangle_free_size(adj)
                    if score < best_score:
                        best_score = score
                        best_edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i][j]:
                                    best_edges.append([i, j])
                else:
                    # Revert
                    adj[u][v] = True
                    adj[v][u] = True
                    
    return best_edges
