import time
import random

def _has_independent_set_10(adj_matrix, n):
    """
    Check if there is an independent set of size 10 in the graph.
    Returns True if found.
    Uses a simple backtracking search.
    """
    # adj_matrix is a 2D list or array where adj_matrix[i][j] is 1 if edge exists.
    # Independent set: no edges between vertices.
    # So we want a set S such that for all u, v in S, adj_matrix[u][v] == 0.
    
    # Optimization: Pre-compute non-neighbors?
    # For each vertex u, non_neighbors[u] is a list of vertices v != u such that adj_matrix[u][v] == 0.
    non_neighbors = []
    for i in range(n):
        nn = []
        for j in range(n):
            if i != j and adj_matrix[i][j] == 0:
                nn.append(j)
        non_neighbors.append(nn)
    
    # Sort vertices by degree (or non-degree) to prune?
    # We want to find an independent set of size 10.
    # Start with a candidate set.
    
    # Simple DFS
    def dfs(current_set, available_vertices):
        if len(current_set) == 10:
            return True
        if len(available_vertices) < 10 - len(current_set):
            return False
        
        # Pick a vertex from available
        # Heuristic: pick one with fewest connections to other available?
        # Or just random/first.
        # Let's try picking the vertex that minimizes the size of the next available set.
        best_v = -1
        best_next_avail = None
        min_size = len(available_vertices) + 1
        
        # Limit search depth/width for speed
        # If available is large, this is expensive.
        # We only need to check a few candidates.
        
        for v in available_vertices:
            # Next available: vertices in available that are non-neighbors of v
            next_avail = [u for u in available_vertices if u != v and adj_matrix[v][u] == 0]
            if len(next_avail) < min_size:
                min_size = len(next_avail)
                best_v = v
                best_next_avail = next_avail
            
            # Early exit if we found a good candidate
            if min_size == 0:
                break
                
        if best_v == -1:
            return False
            
        if dfs(current_set + [best_v], best_next_avail):
            return True
            
        # Backtrack: try other vertices?
        # For speed, we might not backtrack fully.
        # But for correctness, we should.
        # Given n=40, and we just need to find ONE, a greedy approach might fail.
        # Let's use a randomized greedy with restarts.
        
        return False

    # Randomized Greedy Search for Independent Set of size 10
    # We try to build an independent set of size 10.
    # If we fail, it doesn't mean it doesn't exist, but it's a heuristic.
    # To be exact, we need backtracking.
    # However, for n=40, exact check might be slow if alpha is close to 10.
    # But if alpha < 10, we want to prove it.
    # Proving alpha < 10 is hard.
    
    # Alternative: Check if the graph contains an independent set of size 10.
    # If it does, return True.
    # If it doesn't, return False.
    
    # Let's use a branch and bound.
    # State: (current_set, candidates)
    # candidates are vertices that can be added to current_set.
    
    def exact_dfs(current_size, candidates):
        if current_size == 10:
            return True
        if len(candidates) < 10 - current_size:
            return False
        
        # Sort candidates by degree in the subgraph induced by candidates?
        # Or just pick one.
        # Branch on including or excluding the first candidate.
        
        # To speed up, pick a vertex with high degree in the complement?
        # i.e., low degree in G?
        
        # Let's just pick the first one.
        v = candidates[0]
        rest = candidates[1:]
        
        # Option 1: Include v
        # New candidates: vertices in rest that are non-neighbors of v
        new_cands = [u for u in rest if adj_matrix[v][u] == 0]
        if exact_dfs(current_size + 1, new_cands):
            return True
            
        # Option 2: Exclude v
        if exact_dfs(current_size, rest):
            return True
            
        return False

    # Initial candidates: all vertices
    # To make it faster, order vertices by non-degree (descending)?
    # Vertices with many non-neighbors are good candidates for independent set.
    order = sorted(range(n), key=lambda i: -len(non_neighbors[i]))
    return exact_dfs(0, order)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Try n from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this n
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Generate a random maximal triangle-free graph
            # Start with empty graph
            adj = [[0]*n for _ in range(n)]
            
            # Randomly add edges
            # To make it "maximal", we should try to add edges where possible.
            # Iterate through all pairs
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        # Check if adding (i,j) creates a triangle
                        is_triangle = False
                        for k in range(n):
                            if k != i and k != j:
                                if adj[i][k] == 1 and adj[j][k] == 1:
                                    is_triangle = True
                                    break
                        if not is_triangle:
                            adj[i][j] = 1
                            adj[j][i] = 1
            
            # Check if alpha < 10
            # If _has_independent_set_10 returns False, then alpha < 10
            if not _has_independent_set_10(adj, n):
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j] == 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
    # Fallback to seed solution
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
