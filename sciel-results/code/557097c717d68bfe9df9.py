import time
import random
from itertools import combinations

def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix adj on n vertices is triangle-free."""
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True

def _is_not_3_colorable(adj, n):
    """Check if graph is NOT 3-colorable using backtracking."""
    color = [-1] * n
    deg = [sum(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        for c in range(3):
            ok = True
            for u in range(n):
                if adj[v][u] and color[u] == c:
                    ok = False
                    break
            if ok:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return not backtrack(0)

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

def _find_max_independent_sets(adj, n, count, size):
    """Find 'count' independent sets of size 'size' in the graph."""
    sets = []
    vertices = list(range(n))
    
    def is_independent(S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if adj[S[i]][S[j]]:
                    return False
        return True

    # Simple search: generate combinations
    # For n=11, size=5, C(11,5) = 462. Very fast.
    for comb in combinations(vertices, size):
        if is_independent(comb):
            sets.append(list(comb))
            if len(sets) >= count:
                break
    return sets

def _fast_is_not_3_colorable(adj_masks, n):
    """
    Fast 3-colorability check using bitmasks.
    Returns True if NOT 3-colorable.
    """
    # Order vertices by degree (descending) for better pruning
    degs = [adj_masks[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda x: -degs[x])
    
    # colors[v] is the color assigned to vertex v, or -1 if unassigned
    colors = [-1] * n
    # For each vertex, we need to know which colors are forbidden by its colored neighbors.
    # We can maintain a bitmask of forbidden colors for each vertex?
    # Or just check neighbors each time. With bitmasks, checking neighbors is fast.
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        
        # Determine available colors
        # A color c is available if no neighbor has color c.
        # We can iterate through neighbors of v and mark colors.
        forbidden = 0
        # Get neighbors of v
        neighbors = adj_masks[v]
        # Iterate through set bits in neighbors
        while neighbors:
            lsb = neighbors & -neighbors
            u = lsb.bit_length() - 1
            c = colors[u]
            if c != -1:
                forbidden |= (1 << c)
            neighbors ^= lsb
            
        # Try each available color
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False
    
    return not backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n = 16
    adj = _groetzsch_adj(n)
    
    # Find 5 independent sets of size 5 in the Grötzsch part (0-10)
    iso_sets = _find_max_independent_sets(adj, 11, 5, 5)
    
    if len(iso_sets) < 5:
        while len(iso_sets) < 5:
            iso_sets.append(iso_sets[0] if iso_sets else [])

    # Connect new vertices 11-15 to the independent sets
    for i in range(5):
        v_new = 11 + i
        for u in iso_sets[i]:
            adj[v_new][u] = 1
            adj[u][v_new] = 1
    
    # Convert to bitmask representation for fast checks
    # adj_masks[i] is an integer where bit j is 1 if (i, j) is an edge
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
                
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
    # Local Search
    # We will try to add edges.
    # To be efficient, we maintain a list of non-edges and shuffle it.
    # For each non-edge (u, v), check if adding it creates a triangle.
    # Triangle check: (adj_masks[u] & adj_masks[v]) == 0
    
    # Pre-compute list of all potential edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Current state
    current_masks = adj_masks[:]
    current_edge_count = best_edges
    
    # We want to maximize edges.
    # Strategy: Greedily add edges if they don't create triangles and graph remains not 3-colorable.
    # If we add an edge and it becomes 3-colorable, we must remove it.
    # This is a constraint satisfaction problem.
    
    # A better strategy for local search:
    # 1. Start with the initial graph.
    # 2. Iterate through non-edges.
    # 3. If adding edge (u,v) is triangle-free, tentatively add it.
    # 4. Check if still not 3-colorable.
    # 5. If yes, keep it. If no, remove it.
    # 6. Repeat until no more improvements or time limit.
    
    # To avoid getting stuck in a local optimum where we can't add any single edge,
    # we might need to consider removing an edge to allow adding two others.
    # However, the task asks to maximize edges. The seed only adds.
    # Let's stick to adding edges first, as it's simpler and likely effective given the sparse start.
    
    # Optimization: Only check 3-colorability if the triangle check passes.
    # Triangle check is very fast with bitmasks.
    # 3-colorability check is the expensive part.
    
    # We can also prioritize edges that connect high-degree vertices or are "central".
    # But random order with many iterations is robust.
    
    # Let's use a "best improvement" or "first improvement" strategy.
    # First improvement is faster per iteration.
    
    # We will loop until deadline.
    # In each pass, we shuffle non-edges and try to add them.
    
    # To speed up 3-colorability, we can cache results?
    # The state space is huge, so caching might not help much unless we are revisiting states.
    # Local search usually visits unique states.
    
    # Let's refine the 3-colorability check.
    # The recursive `_fast_is_not_3_colorable` is good.
    
    # One optimization: If the graph is 3-colorable, we might want to know which edge caused it?
    # No, we just revert.
    
    # Let's run the local search.
    
    # Track current edge set to know which are non-edges
    # We can just generate non-edges on the fly or maintain a set.
    # For n=16, 120 pairs total. Generating list is cheap.
    
    while time.time() < deadline:
        improved = False
        # Get current non-edges
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not (current_masks[i] & (1 << j)):
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        for (u, v) in non_edges:
            if time.time() > deadline:
                break
            
            # Check triangle
            # If (current_masks[u] & current_masks[v]) != 0, adding (u,v) creates a triangle
            if current_masks[u] & current_masks[v]:
                continue
            
            # Tentatively add edge
            current_masks[u] |= (1 << v)
            current_masks[v] |= (1 << u)
            current_edge_count += 1
            
            # Check 3-colorability
            if _fast_is_not_3_colorable(current_masks, n):
                # Keep edge
                improved = True
                # Update best if better
                if current_edge_count > best_edges:
                    best_edges = current_edge_count
                    # Reconstruct adj matrix for best
                    # We can do this at the end or now.
                    # Doing it now is O(N^2) which is fine.
                    # But we only need to return edges, so we can convert masks to edges at the end.
                    # However, we need to keep track of the best masks.
                    best_masks = current_masks[:]
            else:
                # Revert
                current_masks[u] ^= (1 << v)
                current_masks[v] ^= (1 << u)
                current_edge_count -= 1
        
        if not improved:
            # No single edge addition worked. 
            # We could try 2-edge swaps (remove one, add two) but that's complex.
            # Let's try a few random restarts or perturbations?
            # Or just break if we've done several passes with no improvement.
            # For now, break to avoid infinite loop of useless shuffles.
            # If we have time, we could try removing an edge and then adding others.
            # Let's implement a simple "kick": remove a random edge, then try to add edges again.
            # This allows escaping local optima.
            
            # If we have very few edges, we might not have any to remove.
            # But we start with ~25 edges.
            
            # Let's try a few kicks.
            kicked = False
            for _ in range(10):
                if time.time() > deadline:
                    break
                # Pick a random edge to remove
                # Find all edges
                edges_list = []
                for i in range(n):
                    for j in range(i+1, n):
                        if current_masks[i] & (1 << j):
                            edges_list.append((i, j))
                
                if not edges_list:
                    break
                
                u, v = random.choice(edges_list)
                # Remove edge
                current_masks[u] ^= (1 << v)
                current_masks[v] ^= (1 << u)
                current_edge_count -= 1
                kicked = True
                
                # Try to add edges again (one pass)
                non_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if not (current_masks[i] & (1 << j)):
                            non_edges.append((i, j))
                random.shuffle(non_edges)
                
                for (a, b) in non_edges:
                    if time.time() > deadline:
                        break
                    if current_masks[a] & current_masks[b]:
                        continue
                    
                    current_masks[a] |= (1 << b)
                    current_masks[b] |= (1 << a)
                    current_edge_count += 1
                    
                    if _fast_is_not_3_colorable(current_masks, n):
                        if current_edge_count > best_edges:
                            best_edges = current_edge_count
                            best_masks = current_masks[:]
                    else:
                        current_masks[a] ^= (1 << b)
                        current_masks[b] ^= (1 << a)
                        current_edge_count -= 1
                        
                # If we improved, stop kicking
                if current_edge_count >= best_edges:
                    break
            
            if not kicked:
                break
            
    # Convert best_masks to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_masks[i] & (1 << j):
                edges.append([i, j])
    
    return edges
