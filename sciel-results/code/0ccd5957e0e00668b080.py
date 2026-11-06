import random
import time

def _has_q3(adj, n):
    """Check if graph with adjacency matrix adj on n vertices contains Q3 as subgraph.
    Returns True if found, False otherwise."""
    # Q3: 8 vertices, edges between vertices differing in exactly 1 bit
    # Vertices: 0,1,2,3,4,5,6,7 (binary 000 to 111)
    # Edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    q3_vertices = list(range(8))
    
    # Backtracking: assign Q3 vertices to graph vertices
    # Use symmetry breaking: fix Q3 vertex 0 to graph vertex 0
    # Actually, let's be smarter. Fix Q3 vertex 0 to each possible graph vertex.
    
    def backtrack(assign, remaining):
        if not remaining:
            return True
        # Pick the next Q3 vertex to assign (heuristic: most constrained)
        # For simplicity, just take the first remaining
        qv = remaining[0]
        rest = remaining[1:]
        
        # Try all unassigned graph vertices
        for gv in range(n):
            if gv in assign.values():
                continue
            # Check edges from qv to already-assigned Q3 vertices
            ok = True
            for (a, b) in q3_edges:
                if a == qv and b in assign:
                    if not adj[gv][assign[b]]:
                        ok = False
                        break
                elif b == qv and a in assign:
                    if not adj[gv][assign[a]]:
                        ok = False
                        break
            if not ok:
                continue
            assign[qv] = gv
            if backtrack(assign, rest):
                return True
            del assign[qv]
        return False
    
    # Try fixing Q3 vertex 0 to each graph vertex
    for start in range(n):
        assign = {0: start}
        if backtrack(assign, list(range(1, 8))):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    n = 16
    
    # Use numpy for faster adjacency operations
    import numpy as np
    
    # Start with a random graph with a reasonable density
    # Let's start around 55-65 edges
    random.seed(42)
    
    best_edges = None
    best_count = 0
    
    # Try multiple random starts
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Start with random graph
        edge_set = set()
        # Randomly add edges with probability ~0.5
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.55:
                    edge_set.add((i, j))
        
        # Build adjacency
        adj = [[0]*n for _ in range(n)]
        for (u, v) in edge_set:
            adj[u][v] = 1
            adj[v][u] = 1
        
        # Phase 1: Remove edges to eliminate Q3
        iters = 0
        while time.time() < deadline - 10 and iters < 500:
            if not _has_q3(adj, n):
                break
            # Find a Q3 and remove a random edge from it
            # For now, just randomly remove an edge
            # Better: find which edge to remove
            # Simple approach: remove random edge, recheck
            if edge_set:
                e = random.choice(list(edge_set))
                edge_set.remove(e)
                adj[e[0]][e[1]] = 0
                adj[e[1]][e[0]] = 0
            iters += 1
        
        if time.time() > deadline - 5:
            break
        
        # Phase 2: Try to add edges
        if not _has_q3(adj, n):
            added = True
            while added and time.time() < deadline - 5:
                added = False
                # Try adding random missing edges
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
                random.shuffle(missing)
                for (u, v) in missing:
                    if time.time() > deadline - 5:
                        break
                    edge_set.add((u, v))
                    adj[u][v] = 1
                    adj[v][u] = 1
                    if not _has_q3(adj, n):
                        added = True
                    else:
                        edge_set.remove((u, v))
                        adj[u][v] = 0
                        adj[v][u] = 0
        
        # Update best
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = list(edge_set)
    
    # If we didn't find anything, fall back to K3,13
    if best_edges is None or best_count <= 39:
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return [[u, v] for (u, v) in best_edges]
