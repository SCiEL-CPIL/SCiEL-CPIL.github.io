import time
import random
from itertools import combinations

def _is_3_colorable(adj, n):
    """Check if graph is 3-colorable using backtracking with bitmasks for speed."""
    color = [-1] * n
    deg = [sum(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    # Precompute neighbor lists for fast color checking
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                neighbors[i].add(j)
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        for c in range(3):
            ok = True
            for u in neighbors[v]:
                if color[u] == c:
                    ok = False
                    break
            if ok:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def _is_not_3_colorable(adj, n):
    return not _is_3_colorable(adj, n)

def _count_edges(adj, n):
    return sum(sum(adj[i]) for i in range(n)) // 2

def _get_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def _groetzsch_adj(n=16):
    """Create adjacency matrix for Grötzsch graph on vertices 0-10."""
    adj = [[0]*n for _ in range(n)]
    # Cycle on 0-4
    for i in range(5):
        u, v = i, (i+1) % 5
        adj[u][v] = adj[v][u] = 1
    # Vertices 5-9
    for i in range(5):
        u, w = i, (i+1) % 5
        adj[u][5+w] = adj[5+w][u] = 1
        adj[w][5+u] = adj[5+u][w] = 1
    # Vertex 10
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    return adj

def _find_independent_sets(adj, n, size, max_count=20):
    """Find independent sets of given size in the graph."""
    sets = []
    vertices = list(range(n))
    
    def is_independent(S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if adj[S[i]][S[j]]:
                    return False
        return True

    for comb in combinations(vertices, size):
        if is_independent(comb):
            sets.append(list(comb))
            if len(sets) >= max_count:
                break
    return sets

def _can_add_edge(adj, i, j, n):
    """Check if adding edge (i, j) creates a triangle."""
    for k in range(n):
        if k != i and k != j and adj[i][k] and adj[j][k]:
            return False
    return True

def _local_search(adj, n, deadline, random_state=None):
    """Local search: try to add edges while maintaining triangle-freeness and non-3-colorability."""
    if random_state is not None:
        random.seed(random_state)
    
    current_adj = [row[:] for row in adj]
    current_edges = _count_edges(current_adj, n)
    
    # Greedy addition phase
    for restart in range(50):
        if time.time() > deadline:
            break
        
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not current_adj[i][j]:
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        for (i, j) in non_edges:
            if _can_add_edge(current_adj, i, j, n):
                current_adj[i][j] = 1
                current_adj[j][i] = 1
        
        new_edges = _count_edges(current_adj, n)
        if new_edges > current_edges:
            current_edges = new_edges
    
    return current_adj, current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    best_adj = None
    best_edges = 0
    
    # Get all independent sets of size 5 in the Grötzsch part (vertices 0-10)
    base_adj = _groetzsch_adj(n)
    iso_sets = _find_independent_sets(base_adj, 11, 5, max_count=50)
    
    if len(iso_sets) < 5:
        # Fallback: use smaller independent sets
        iso_sets_4 = _find_independent_sets(base_adj, 11, 4, max_count=50)
        iso_sets = iso_sets + iso_sets_4
        if len(iso_sets) < 5:
            while len(iso_sets) < 5:
                iso_sets.append([0, 1, 2, 3, 4])
    
    # Try multiple combinations of independent sets for the 5 new vertices
    # Use combinations with replacement to allow reusing independent sets
    import itertools
    
    # First, try the original approach: 5 distinct independent sets
    for combo_idx in range(min(20, len(iso_sets))):
        if time.time() > deadline:
            break
        
        # Pick 5 independent sets (with possible repetition)
        selected_sets = []
        for i in range(5):
            idx = (combo_idx * 5 + i) % len(iso_sets)
            selected_sets.append(iso_sets[idx])
        
        adj = _groetzsch_adj(n)
        for i in range(5):
            v_new = 11 + i
            for u in selected_sets[i]:
                adj[v_new][u] = 1
                adj[u][v_new] = 1
        
        # Verify not 3-colorable
        if not _is_not_3_colorable(adj, n):
            continue
        
        # Local search
        search_adj, search_edges = _local_search(adj, n, deadline)
        
        if search_edges > best_edges:
            best_edges = search_edges
            best_adj = search_adj
    
    # If we haven't found anything, use the original approach
    if best_adj is None:
        adj = _groetzsch_adj(n)
        for i in range(5):
            v_new = 11 + i
            for u in iso_sets[i]:
                adj[v_new][u] = 1
                adj[u][v_new] = 1
        
        if _is_not_3_colorable(adj, n):
            search_adj, search_edges = _local_search(adj, n, deadline)
            best_adj = search_adj
            best_edges = search_edges
    
    if best_adj is None:
        # Emergency fallback: return empty graph (should not happen)
        return []
    
    return _get_edges(best_adj, n)
