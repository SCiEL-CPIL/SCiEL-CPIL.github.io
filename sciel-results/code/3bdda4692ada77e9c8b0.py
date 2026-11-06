"""Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph."""
import random
import itertools

def _is_k4_free(n, adj):
    """Check if graph with adjacency matrix adj on n vertices is K4-free."""
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _max_triangle_free_induced(n, adj):
    """Exact branch and bound for largest triangle-free induced subgraph."""
    # A set S is triangle-free iff for every edge (u,v) in S, there is no w in S adjacent to both.
    # Equivalently, the subgraph induced by S has no triangles.
    # We can use: a set is triangle-free iff it is an independent set in the triangle-hypergraph? No.
    # Better: use the fact that a graph is triangle-free iff each vertex's neighbors within the set form an independent set.
    # Branch and bound: build set incrementally, check triangle-free property.
    
    best = [0]
    
    def dfs(start, current_set, current_size):
        if current_size + (n - start) <= best[0]:
            return
        if current_size > best[0]:
            best[0] = current_size
        
        # Try adding each vertex from start to n-1
        for v in range(start, n):
            # Check if adding v keeps the set triangle-free
            # v must not form a triangle with any pair in current_set
            ok = True
            for i in range(len(current_set)):
                u1 = current_set[i]
                if not adj[u1][v]:
                    continue
                for j in range(i+1, len(current_set)):
                    u2 = current_set[j]
                    if adj[u1][u2] and adj[u2][v]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            current_set.append(v)
            dfs(v + 1, current_set, current_size + 1)
            current_set.pop()
    
    dfs(0, [], 0)
    return best[0]

def _score_graph(n, adj):
    """Return the largest triangle-free induced subgraph size."""
    return _max_triangle_free_induced(n, adj)

def _random_k4_free_graph(n, trials=5000):
    """Generate random K4-free graphs and return the best one found."""
    best_score = n + 1
    best_adj = None
    
    for trial in range(trials):
        # Randomly assign edges, then remove edges to ensure K4-free
        # Strategy: start with random graph, remove one edge from each K4
        adj = [[False] * n for _ in range(n)]
        
        # Random edge assignment
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = adj[j][i] = True
        
        # Remove edges to make K4-free
        # Find all K4s and remove an edge from each
        changed = True
        while changed:
            changed = False
            for a in range(n):
                for b in range(a+1, n):
                    if not adj[a][b]:
                        continue
                    for c in range(b+1, n):
                        if not adj[a][c] or not adj[b][c]:
                            continue
                        for d in range(c+1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                # Found K4, remove a random edge from it
                                edges_in_k4 = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                e = random.choice(edges_in_k4)
                                adj[e[0]][e[1]] = adj[e[1]][e[0]] = False
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        
        if not _is_k4_free(n, adj):
            continue
        
        score = _score_graph(n, adj)
        if score < best_score:
            best_score = score
            best_adj = [row[:] for row in adj]
            if best_score <= 7:
                break
    
    if best_adj is None:
        # Fallback: complete tripartite
        adj = [[False] * n for _ in range(n)]
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a][b] = adj[b][a] = True
        return adj
    
    return best_adj

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph."""
    n = 18
    
    # Try several structured constructions and random search
    candidates = []
    
    # 1. Complete tripartite K6,6,6 (baseline: score 12)
    adj1 = [[False] * n for _ in range(n)]
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a][b] = adj1[b][a] = True
    candidates.append(adj1)
    
    # 2. Complete 4-partite with sizes 5,4,4,5
    adj2 = [[False] * n for _ in range(n)]
    sizes = [5, 4, 4, 5]
    start = 0
    parts2 = []
    for s in sizes:
        parts2.append(list(range(start, start+s)))
        start += s
    for i in range(4):
        for j in range(i+1, 4):
            for a in parts2[i]:
                for b in parts2[j]:
                    adj2[a][b] = adj2[b][a] = True
    candidates.append(adj2)
    
    # 3. Complete 5-partite with sizes 4,4,4,3,3
    adj3 = [[False] * n for _ in range(n)]
    sizes3 = [4, 4, 4, 3, 3]
    start = 0
    parts3 = []
    for s in sizes3:
        parts3.append(list(range(start, start+s)))
        start += s
    for i in range(5):
        for j in range(i+1, 5):
            for a in parts3[i]:
                for b in parts3[j]:
                    adj3[a][b] = adj3[b][a] = True
    candidates.append(adj3)
    
    # 4. Complete 6-partite with sizes 3,3,3,3,3,3
    adj4 = [[False] * n for _ in range(n)]
    sizes4 = [3, 3, 3, 3, 3, 3]
    start = 0
    parts4 = []
    for s in sizes4:
        parts4.append(list(range(start, start+s)))
        start += s
    for i in range(6):
        for j in range(i+1, 6):
            for a in parts4[i]:
                for b in parts4[j]:
                    adj4[a][b] = adj4[b][a] = True
    candidates.append(adj4)
    
    # 5. Turan-like with some internal edges removed to break K4s
    # Use a random K4-free search
    random.seed(42)
    adj5 = _random_k4_free_graph(n, trials=3000)
    candidates.append(adj5)
    
    # Evaluate all candidates
    best_adj = None
    best_score = n + 1
    for adj in candidates:
        if not _is_k4_free(n, adj):
            continue
        score = _score_graph(n, adj)
        if score < best_score:
            best_score = score
            best_adj = adj
        if best_score <= 7:
            break
    
    if best_adj is None:
        best_adj = adj1
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
