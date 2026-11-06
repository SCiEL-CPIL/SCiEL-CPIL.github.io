import random
import time
import itertools

def _has_q3_fast(adj, n=16):
    """
    Check if graph with adjacency matrix adj (list of lists of bools) on n vertices
    contains Q3 as a subgraph.
    Returns True if found, False otherwise.
    Uses enumeration of 8-vertex subsets and backtracking.
    """
    # Q3 edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    q3_adj = [[0]*8 for _ in range(8)]
    for u, v in q3_edges:
        q3_adj[u][v] = 1
        q3_adj[v][u] = 1
    
    # Precompute neighbors in Q3 for faster checking
    q3_neighbors = [set() for _ in range(8)]
    for u, v in q3_edges:
        q3_neighbors[u].add(v)
        q3_neighbors[v].add(u)

    def backtrack(subset, assign):
        # assign: dict q3_vertex -> graph_vertex
        # subset: list of 8 graph vertices
        
        # Heuristic: pick the Q3 vertex with the most assigned neighbors
        # Or just iterate 1..7
        # Since we fix 0, we can iterate 1..7
        # To be faster, we can pick the unassigned Q3 vertex with the most constraints
        # For simplicity and speed in Python, let's just try 1..7
        
        for qv in range(1, 8):
            if qv in assign:
                continue
            
            # Try all unassigned graph vertices in subset
            used = set(assign.values())
            for gv in subset:
                if gv in used:
                    continue
                
                # Check if gv is connected to all assigned neighbors of qv
                ok = True
                for neighbor_qv in q3_neighbors[qv]:
                    if neighbor_qv in assign:
                        if not adj[gv][assign[neighbor_qv]]:
                            ok = False
                            break
                if not ok:
                    continue
                
                assign[qv] = gv
                if backtrack(subset, assign):
                    return True
                del assign[qv]
        
        return False

    # Enumerate all 8-vertex subsets
    # To speed up, we can break early if found
    for subset in itertools.combinations(range(n), 8):
        # Quick check: Does the induced subgraph have at least 12 edges?
        # If not, it can't contain Q3.
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[subset[i]][subset[j]]:
                    edge_count += 1
        if edge_count < 12:
            continue
            
        # Try to find Q3 in this subset
        # Fix Q3 vertex 0 to subset[0]
        # To handle symmetries, we could try all 8 positions, but fixing to 0 and relying on the subset enumeration to cover all cases is sufficient?
        # No, fixing Q3 vertex 0 to a specific graph vertex in the subset might miss cases where Q3 vertex 0 maps to a different vertex in the subset.
        # However, since we iterate over ALL subsets, and for each subset we try to map Q3 onto it, we must try all mappings.
        # Mapping Q3 vertex 0 to any of the 8 vertices in the subset.
        
        # Optimization: Only try mapping Q3 vertex 0 to vertices in the subset that have degree >= 3 in the induced subgraph?
        # All Q3 vertices have degree 3.
        
        for start_gv in subset:
            assign = {0: start_gv}
            if backtrack(subset, assign):
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    n = 16
    
    best_edges = None
    best_count = 0
    
    # Strategy: Start with K16, remove edges to destroy Q3s.
    # Then try to add edges back.
    
    # Initialize K16
    adj = [[1]*n for _ in range(n)]
    for i in range(n):
        adj[i][i] = 0
    edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            edge_set.add((i, j))
    
    # Phase 1: Remove edges to make it Q3-free
    # We need a function to find a Q3 and return one of its edges
    def find_q3_edge(adj, n=16):
        # Reuse logic from _has_q3_fast but return an edge
        q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        q3_neighbors = [set() for _ in range(8)]
        for u, v in q3_edges:
            q3_neighbors[u].add(v)
            q3_neighbors[v].add(u)

        def backtrack(subset, assign):
            for qv in range(1, 8):
                if qv in assign:
                    continue
                used = set(assign.values())
                for gv in subset:
                    if gv in used:
                        continue
                    ok = True
                    for neighbor_qv in q3_neighbors[qv]:
                        if neighbor_qv in assign:
                            if not adj[gv][assign[neighbor_qv]]:
                                ok = False
                                break
                    if not ok:
                        continue
                    assign[qv] = gv
                    if backtrack(subset, assign):
                        return True
                    del assign[qv]
            return False

        for subset in itertools.combinations(range(n), 8):
            edge_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[subset[i]][subset[j]]:
                        edge_count += 1
            if edge_count < 12:
                continue
            
            for start_gv in subset:
                assign = {0: start_gv}
                if backtrack(subset, assign):
                    # Found a Q3, return a random edge from it
                    # Map Q3 edges to graph edges
                    g_edges = []
                    for u, v in q3_edges:
                        gu = assign[u]
                        gv = assign[v]
                        g_edges.append((gu, gv))
                    return random.choice(g_edges)
        return None

    iters = 0
    while time.time() < deadline - 20:
        e = find_q3_edge(adj, n)
        if e is None:
            break
        # Remove edge e
        u, v = e
        if u > v: u, v = v, u
        edge_set.remove((u, v))
        adj[u][v] = 0
        adj[v][u] = 0
        iters += 1
        if iters > 1000: break # Safety

    # Phase 2: Try to add edges back
    # Greedy: try to add any missing edge that doesn't create a Q3
    added = True
    while added and time.time() < deadline - 10:
        added = False
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 10:
                break
            # Try adding
            edge_set.add((u, v))
            adj[u][v] = 1
            adj[v][u] = 1
            if not _has_q3_fast(adj, n):
                added = True
            else:
                edge_set.remove((u, v))
                adj[u][v] = 0
                adj[v][u] = 0
    
    best_count = len(edge_set)
    best_edges = list(edge_set)
    
    # If this result is bad, try a random restart?
    # The current approach starts from K16 which is very dense, so it should find a good solution.
    
    return [[u, v] for (u, v) in best_edges]
