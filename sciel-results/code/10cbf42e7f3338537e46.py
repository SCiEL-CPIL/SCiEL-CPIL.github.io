import time

def construct():
    n = 15
    deadline = time.time() + 80
    
    def check_valid(edges_list):
        """Check if graph satisfies constraint. Returns True if valid."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges_list:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        for u, v in edges_list:
            degs_u = set(deg[s] for s in adj[u] if s != v)
            degs_v = set(deg[t] for t in adj[v] if t != u)
            if degs_u & degs_v:
                return False
        return True
    
    # Try construction 1: Complete graph minus edges to create degree variation
    # Let's try: start with K15 and remove edges strategically
    
    # Construction: Make vertex 0 connected to all, vertex 1 to all except 0, etc.
    # This creates a "nested" structure
    
    # Actually, let me try: for i < j, include edge (i,j) if some condition
    # Let's try including edge (i,j) if i+j < some threshold, or based on parity
    
    # Let me try a specific known-good structure:
    # Complete graph on 15 vertices minus a perfect matching-like structure
    
    # Try: all edges except those where both endpoints are in a specific set
    
    # Let me just try several random graphs and a few structured ones
    
    candidates = []
    
    # Candidate 1: Complete graph
    complete = [(i, j) for i in range(n) for j in range(i+1, n)]
    candidates.append(complete)
    
    # Candidate 2: Complete minus edges within first 8 vertices
    c2 = [(i, j) for i in range(n) for j in range(i+1, n) if not (i < 8 and j < 8)]
    candidates.append(c2)
    
    # Candidate 3: Complete bipartite K(7,8)
    c3 = [(i, j) for i in range(7) for j in range(7, 15)]
    candidates.append(c3)
    
    # Candidate 4: Complete 2-partite with parts 7 and 8, plus some internal edges
    # Let's try: all edges between {0-6} and {7-14}, plus all edges within {7-14}
    c4 = [(i, j) for i in range(7) for j in range(7, 15)]
    c4 += [(i, j) for i in range(7, 15) for j in range(i+1, 15)]
    candidates.append(c4)
    
    # Candidate 5: Turan graph T(15,3) - complete 3-partite with parts as equal as possible
    # Parts: {0,3,6,9,12}, {1,4,7,10,13}, {2,5,8,11,14}
    parts = [[0,3,6,9,12], [1,4,7,10,13], [2,5,8,11,14]]
    c5 = []
    for a in range(3):
        for b in range(a+1, 3):
            for i in parts[a]:
                for j in parts[b]:
                    if i < j:
                        c5.append((i, j))
                    else:
                        c5.append((j, i))
    candidates.append(c5)
    
    # Candidate 6: Complete graph minus a star (remove all edges from vertex 0)
    c6 = [(i, j) for i in range(1, n) for j in range(i+1, n)]
    candidates.append(c6)
    
    # Candidate 7: Try a graph where we remove edges to make degrees distinct-ish
    # Start with K15, remove edges (0,1), (0,2), ..., to reduce vertex 0's degree
    # This is getting complicated. Let me just evaluate the candidates above.
    
    best = []
    best_len = 0
    
    for cand in candidates:
        if check_valid(cand):
            if len(cand) > best_len:
                best_len = len(cand)
                best = cand
    
    # If none worked, try a more systematic approach
    if best_len == 0:
        # Try removing edges from K15 one by one
        # Start with K15 and greedily remove edges to fix violations
        edges = set(complete)
        
        def get_violations(edge_set):
            """Return list of violating edges."""
            deg = [0] * n
            adj = [set() for _ in range(n)]
            for u, v in edge_set:
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)
            
            violations = []
            for u, v in edge_set:
                degs_u = set(deg[s] for s in adj[u] if s != v)
                degs_v = set(deg[t] for t in adj[v] if t != u)
                if degs_u & degs_v:
                    violations.append((u, v))
            return violations
        
        # Greedy: while there are violations, remove one edge that fixes the most
        while True:
            if time.time() > deadline:
                break
            viols = get_violations(edges)
            if not viols:
                break
            # Try removing each violating edge and see which fixes most violations
            # This is expensive, so just remove the first one
            u, v = viols[0]
            edges.discard((u, v))
            edges.discard((v, u))
        
        best = list(edges)
        best_len = len(best)
    
    # Convert to proper format
    result = []
    for u, v in best:
        if u < v:
            result.append([int(u), int(v)])
        else:
            result.append([int(v), int(u)])
    
    return result
