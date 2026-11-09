import random
import time

def construct():
    n_vertices = 40
    best_edges_count = 0
    best_result = []
    
    # Helper to check if adding (u,v) creates a C6
    def creates_c6(u, v, adj, edges):
        n2_u = []
        for a in adj[u]:
            for b in adj[a]:
                if b != u:
                    n2_u.append((b, a))
                    
        n2_v = []
        for d in adj[v]:
            for c in adj[d]:
                if c != v:
                    n2_v.append((c, d))
                    
        for b, a in n2_u:
            if b == v or a == v:
                continue
            for c, d in n2_v:
                if c == u or d == u:
                    continue
                if b == c:
                    continue
                if b == d or a == c or a == d:
                    continue
                if (b, c) in edges:
                    return True
        return False

    def find_c6(adj, edges):
        # Returns a list of 6 vertices forming a C6, or None
        # We look for a cycle of length 6
        # u - a - b - c - d - v - u
        # Fix u, a, b. Then we need a path of length 3 from b to u, avoiding a, b, u, and not using edges that close a smaller cycle?
        # Actually, just finding any 6-cycle.
        
        # Optimization: Iterate over edges (u, a).
        # Extend to (u, a, b).
        # Then look for path b -> c -> d -> u.
        # c in N(b) \ {a}
        # d in N(c) \ {b, u} (if d==u, length is 3 from b to u, total cycle 5? u-a-b-c-u is C4. u-a-b-c-d-u is C5. We want C6.)
        # So path b-c-d-e-u? No, u-a-b-c-d-e-u is C6?
        # Vertices: u, a, b, c, d, e.
        # Edges: (u,a), (a,b), (b,c), (c,d), (d,e), (e,u).
        
        for u in range(n_vertices):
            for a in adj[u]:
                for b in adj[a]:
                    if b == u: continue
                    # Path so far: u-a-b
                    # Need path b-c-d-e-u of length 3 from b to u? No, length 3 edges: b-c, c-d, d-e, e-u?
                    # Total edges: (u,a), (a,b), (b,c), (c,d), (d,e), (e,u).
                    # Vertices: u, a, b, c, d, e.
                    # Check distinctness: u,a,b,c,d,e all distinct.
                    for c in adj[b]:
                        if c == u or c == a: continue
                        for d in adj[c]:
                            if d == u or d == a or d == b: continue
                            for e in adj[d]:
                                if e == u or e == a or e == b or e == c: continue
                                if (e, u) in edges:
                                    # Check if this is a valid C6
                                    # Vertices: u, a, b, c, d, e
                                    # Edges: (u,a), (a,b), (b,c), (c,d), (d,e), (e,u)
                                    # We need to ensure no chords? No, C6 is a subgraph.
                                    # The definition of "containing no six-cycle" usually means no induced C6?
                                    # "containing no six-cycle" means the graph does not have a C6 as a subgraph.
                                    # So if there is a C6 subgraph, it's invalid.
                                    # Chords don't matter, the cycle exists.
                                    return [u, a, b, c, d, e]
        return None

    def repair_graph(edges):
        adj = [[] for _ in range(n_vertices)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Remove edges to break C6s
        # We want to maximize edges, so remove as few as possible.
        # Random removal.
        while True:
            cycle = find_c6(adj, edges)
            if cycle is None:
                break
            # Pick a random edge from the cycle to remove
            # Cycle is u-a-b-c-d-e-u
            # Edges: (u,a), (a,b), (b,c), (c,d), (d,e), (e,u)
            cycle_edges = [
                (cycle[0], cycle[1]),
                (cycle[1], cycle[2]),
                (cycle[2], cycle[3]),
                (cycle[3], cycle[4]),
                (cycle[4], cycle[5]),
                (cycle[5], cycle[0])
            ]
            # Normalize edges
            normalized = []
            for e in cycle_edges:
                if e[0] > e[1]: e = (e[1], e[0])
                normalized.append(e)
            
            rem_edge = random.choice(normalized)
            # Remove from edges set
            edges.discard(rem_edge)
            edges.discard((rem_edge[1], rem_edge[0]))
            
            # Update adj
            u, v = rem_edge
            adj[u].remove(v)
            adj[v].remove(u)
            
        return edges, adj

    def greedy_add(edges, adj):
        # Try to add all missing edges
        candidates = []
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if (i, j) not in edges:
                    candidates.append((i, j))
        
        random.shuffle(candidates)
        
        for u, v in candidates:
            if not creates_c6(u, v, adj, edges):
                edges.add((u, v))
                edges.add((v, u))
                adj[u].append(v)
                adj[v].append(u)
                
        return edges, adj

    start_time = time.time()
    
    # Strategy: Random Graph Repair
    # Try various densities
    for attempt in range(1000):
        if time.time() - start_time > 110:
            break
            
        random.seed(attempt)
        
        # Start with a random graph
        # Target ~110-120 edges
        num_edges = random.randint(110, 130)
        edges = set()
        while len(edges) < num_edges:
            u = random.randint(0, n_vertices - 1)
            v = random.randint(0, n_vertices - 1)
            if u != v:
                if u > v: u, v = v, u
                edges.add((u, v))
                edges.add((v, u))
                
        # Repair
        edges, adj = repair_graph(edges)
        
        # Greedy Add
        edges, adj = greedy_add(edges, adj)
        
        count = len([e for e in edges if e[0] < e[1]])
        
        if count > best_edges_count:
            best_edges_count = count
            best_result = [[u, v] for u, v in edges if u < v]
            
        # If we hit 105, we might try to improve further with local search?
        # For now, just collect best.
        
    return best_result
