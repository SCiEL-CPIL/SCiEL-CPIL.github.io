import time
import random

def _has_c5_after_adding_edge(adj, u, v):
    """Check if adding edge (u,v) creates a C5. adj is list of sets."""
    nu = adj[u]
    nv = adj[v]
    for a in nu:
        if a == v:
            continue
        for c in nv:
            if a == c or c == u:
                continue
            for b in adj[a]:
                if b in adj[c] and b != u and b != v and b != a and b != c:
                    return True
    return False

def _check_alpha_leq_8(adj, n):
    """Check if independence number is <= 8."""
    vertices = list(range(n))
    vertices.sort(key=lambda x: len(adj[x]))
    
    def dfs(current_size, candidates):
        if current_size + len(candidates) < 9:
            return False
        if current_size == 9:
            return True
        if not candidates:
            return False
        
        v = candidates[0]
        rest = candidates[1:]
        
        new_cands = [u for u in rest if u not in adj[v]]
        if dfs(current_size + 1, new_cands):
            return True
        
        if dfs(current_size, rest):
            return True
        
        return False
    
    return not dfs(0, vertices)

def _check_c5_free(adj, n):
    """Verify graph is C5-free."""
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            edge_set.add((min(u,v), max(u,v)))
    
    for u, v in edge_set:
        for a in adj[u]:
            if a == v:
                continue
            for b in adj[a]:
                if b == u or b == v:
                    continue
                for c in adj[b]:
                    if c == a or c == u or c == v:
                        continue
                    if c in adj[v]:
                        return False
    return True

def _make_k4_disjoint(k):
    """Create k disjoint K4s. Returns (n, adj)."""
    n = 4 * k
    adj = [set() for _ in range(n)]
    for b in range(k):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i].add(base + j)
                adj[base + j].add(base + i)
    return n, adj

def _try_extend(adj, n, deadline, max_attempts=100):
    """
    Try to add one vertex to the graph. Returns (success, new_n, new_adj).
    """
    new_v = n
    candidates = list(range(n))
    random.shuffle(candidates)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return False, n, adj
        
        # Try different numbers of edges for the new vertex
        num_edges = attempt % 5  # 0,1,2,3,4
        if num_edges > len(candidates):
            num_edges = len(candidates)
        
        if num_edges == 0:
            neighbors = []
        else:
            neighbors = random.sample(candidates, num_edges)
        
        # Check C5 for each potential edge
        valid = True
        for w in neighbors:
            if _has_c5_after_adding_edge(adj, new_v, w):
                valid = False
                break
        if not valid:
            continue
        
        # Tentatively add
        new_adj = [set(s) for s in adj]
        new_adj.append(set(neighbors))
        for w in neighbors:
            new_adj[w].add(new_v)
        
        if _check_c5_free(new_adj, n + 1) and _check_alpha_leq_8(new_adj, n + 1):
            return True, n + 1, new_adj
    
    return False, n, adj

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Start with 8 disjoint K4s
    n, adj = _make_k4_disjoint(8)
    
    # Try to extend up to 35 vertices
    for _ in range(3):  # Try to add 3 more vertices (33, 34, 35)
        if time.time() > deadline:
            break
        success, n, adj = _try_extend(adj, n, deadline, max_attempts=50)
        if not success:
            break
    
    # Build edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
