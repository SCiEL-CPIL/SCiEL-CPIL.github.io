"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 12
    
    def get_adjacency(edges):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
        return True
    
    def min_degree_ok(adj):
        return all(len(adj[u]) >= 3 for u in range(n))
    
    def has_cycle(adj):
        visited = [False] * n
        parent = [-1] * n
        has_cyc = [False]
        
        def dfs(u):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    parent[v] = u
                    dfs(v)
                elif v != parent[u]:
                    has_cyc[0] = True
                    return
        
        for i in range(n):
            if not visited[i]:
                dfs(i)
                if has_cyc[0]:
                    return True
        return has_cyc[0]
    
    def find_distinct_cycle_lengths(adj):
        """Find all distinct simple cycle lengths in the graph."""
        cycle_lengths = set()
        
        # Use DFS to enumerate all simple cycles
        # For each pair (u, v) with u < v, find all simple paths from u to v
        # and check if they form a cycle (they always do since u-v is an edge)
        
        # Better approach: for each vertex as root, do DFS to find cycles
        # A cycle is found when we return to the start vertex
        
        # Since n=12 is small, we can enumerate cycles more directly
        # For each subset of vertices of size k (4 <= k <= 12), check if they form a cycle
        # This is too expensive for k up to 12
        
        # Alternative: BFS/DFS from each vertex, tracking path
        # Limit path length to avoid exponential blowup
        
        # For each starting vertex s, do DFS with path tracking
        # A cycle is formed when we reach s again with path length >= 4
        
        for s in range(n):
            # DFS from s, only allowing vertices > s in the path (except s at end)
            # This avoids counting the same cycle multiple times
            # Actually, let's use a different approach:
            # For each edge (u, v), find all simple paths from u to v not using that edge
            # Each such path + the edge forms a cycle
            
            # Simpler: enumerate all simple cycles by DFS
            # Start from each vertex, only visit vertices >= start to avoid duplicates
            # But this doesn't work well for cycles
            
            # Let's just do a full enumeration with canonical ordering
            # A cycle is represented by its vertices in order, with the smallest vertex first
            # and the second vertex being smaller than the last vertex (to break symmetry)
            
            pass
        
        # More efficient: for each vertex s, find all simple cycles where s is the smallest vertex
        # and the cycle goes s -> ... -> s
        # We do DFS from s, only visiting vertices >= s (except returning to s)
        
        for s in range(n):
            # DFS to find all simple cycles starting and ending at s
            # where all intermediate vertices are > s
            stack = [(s, [s])]
            # Use iterative DFS to avoid recursion limits
            # But we need to track the path
            
            # Actually, let's use a recursive approach with memoization of visited
            # in the current path
            
            visited = [False] * n
            visited[s] = True
            
            def dfs(u, path_len):
                for v in sorted(adj[u]):
                    if v == s and path_len >= 3:  # cycle of length path_len+1 >= 4
                        cycle_lengths.add(path_len + 1)
                        return
                    if v > s and not visited[v]:
                        visited[v] = True
                        dfs(v, path_len + 1)
                        visited[v] = False
            
            dfs(s, 0)
        
        return cycle_lengths
    
    def generate_two_k33():
        """Generate two disjoint K_{3,3} components."""
        edges = []
        # Component 1: Vertices 0,1,2 (L) and 3,4,5 (R)
        for i in range(3):
            for j in range(3):
                edges.append([i, j + 3])
        # Component 2: Vertices 6,7,8 (L) and 9,10,11 (R)
        for i in range(3):
            for j in range(3):
                edges.append([i + 6, j + 9])
        return edges
    
    def validate(edges):
        """Check all constraints."""
        if len(edges) != len(set(tuple(sorted(e)) for e in edges)):
            return False  # duplicates
        for u, v in edges:
            if u == v:
                return False  # loop
            if not (0 <= u < n and 0 <= v < n):
                return False
        adj = get_adjacency(edges)
        if not min_degree_ok(adj):
            return False
        if not is_triangle_free(adj):
            return False
        if not has_cycle(adj):
            return False
        return True
    
    def score_edges(edges):
        """Return the number of distinct cycle lengths."""
        adj = get_adjacency(edges)
        lengths = find_distinct_cycle_lengths(adj)
        return len(lengths)
    
    # Start with the known good solution: two K_{3,3}
    best_edges = generate_two_k33()
    best_score = score_edges(best_edges)
    
    # Try to find a better solution with fewer distinct cycle lengths
    # The theoretical minimum is 1 (all cycles same length)
    # For triangle-free graphs with min degree 3 on 12 vertices,
    # can we have all cycles the same length?
    
    # A graph where all cycles have length 4 would be a union of C4s,
    # but C4 has degree 2, so we need to connect them while maintaining
    # min degree 3 and girth 4.
    
    # Let's try random search with local improvement
    random.seed(42)
    
    def random_graph():
        """Generate a random graph with min degree at least 3."""
        # Start with a random 3-regular graph (if it exists)
        # or just generate random edges and check
        edges = set()
        for _ in range(200):  # try up to 200 times
            edges = set()
            # Generate random edges
            all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_possible)
            
            # Greedily add edges ensuring min degree 3
            deg = [0] * n
            for u, v in all_possible:
                if deg[u] < 3 and deg[v] < 3:
                    # Check if adding this edge creates a triangle
                    adj_temp = get_adjacency(list(edges))
                    adj_temp[u].add(v)
                    adj_temp[v].add(u)
                    if is_triangle_free(adj_temp):
                        edges.add((u, v))
                        deg[u] += 1
                        deg[v] += 1
            
            if all(d >= 3 for d in deg):
                adj = get_adjacency(list(edges))
                if is_triangle_free(adj) and has_cycle(adj):
                    return [list(e) for e in edges]
        
        return None
    
    def local_improve(edges):
        """Try to improve the graph by edge swaps."""
        edges_set = set(tuple(sorted(e)) for e in edges)
        current_score = score_edges(list(edges_set))
        
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            # Try to remove an edge and add another
            edge_list = list(edges_set)
            if not edge_list:
                break
            
            rem_edge = random.choice(edge_list)
            edges_set.remove(rem_edge)
            
            # Try to add a random new edge
            added = False
            for _ in range(50):
                u, v = random.sample(range(n), 2)
                new_edge = tuple(sorted((u, v)))
                if new_edge not in edges_set and new_edge != rem_edge:
                    test_set = edges_set | {new_edge}
                    test_edges = [list(e) for e in test_set]
                    if validate(test_edges):
                        new_score = score_edges(test_edges)
                        if new_score < current_score:
                            edges_set = test_set
                            current_score = new_score
                            added = True
                            break
            
            if not added:
                edges_set.add(rem_edge)
        
        return [list(e) for e in edges_set], current_score
    
    # Try random restarts with local improvement
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        candidate = random_graph()
        if candidate is None:
            continue
        
        improved, score = local_improve(candidate)
        if score < best_score:
            best_edges = improved
            best_score = score
    
    # Also try to improve the initial K_{3,3} solution
    improved, score = local_improve(best_edges)
    if score < best_score:
        best_edges = improved
        best_score = score
    
    # Final validation
    if not validate(best_edges):
        # Fallback to known good solution
        best_edges = generate_two_k33()
    
    # Ensure return type is correct
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
