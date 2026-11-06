"""Triangle-free 4-chromatic graphs on 16 vertices via core extension and local search."""
# EVOLVE-BLOCK-START
import random
import time

def _build_grotzsch_edges():
    """
    Return the edge list of the Grötzsch graph (Mycielski of C5) on vertices 0..10.
    Vertices 11..15 are isolated in the initial state.
    """
    # C5: 0-1-2-3-4-0
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    # Mycielski construction:
    # Add vertex 5 connected to neighbors of 0,1,2,3,4 in the cycle?
    # Standard Mycielski:
    # V = {v0...v4, w0...w4, u}
    # Edges:
    # 1. Original cycle edges: (vi, vi+1)
    # 2. wi connected to vi+1 and vi-1 (neighbors of vi in C5)
    # 3. u connected to all wi
    
    # Let's map:
    # v0=0, v1=1, v2=2, v3=3, v4=4
    # w0=5, w1=6, w2=7, w3=8, w4=9
    # u=10
    
    # wi is connected to neighbors of vi in the cycle.
    # Neighbors of v0(0) are v4(4) and v1(1). So w0(5) connects to 4 and 1.
    # Neighbors of v1(1) are v0(0) and v2(2). So w1(6) connects to 0 and 2.
    # Neighbors of v2(2) are v1(1) and v3(3). So w2(7) connects to 1 and 3.
    # Neighbors of v3(3) are v2(2) and v4(4). So w3(8) connects to 2 and 4.
    # Neighbors of v4(4) are v3(3) and v0(0). So w4(9) connects to 3 and 0.
    
    for i in range(5):
        w_idx = 5 + i
        # Neighbors of i in C5
        prev = (i - 1) % 5
        next = (i + 1) % 5
        edges.append([w_idx, prev])
        edges.append([w_idx, next])
        
    # u (10) connects to all w (5..9)
    for i in range(5):
        edges.append([10, 5 + i])
        
    return edges

def _make_bitmask_adj(edge_list, n=16):
    """Convert edge list to list of integer bitmasks representing adjacency."""
    adj = [0] * n
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_valid_triangle_free(adj):
    """Verify that adjacency bitmasks define a simple triangle-free graph."""
    n = len(adj)
    for u in range(n):
        if adj[u] & (1 << u):
            return False
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            # If u and v are connected, they must not share any common neighbor
            if adj[u] & adj[v]:
                return False
    return True

def _is_3_colorable(adj):
    """
    Check if the graph defined by adj bitmasks is 3-colorable.
    Returns True if 3-colorable, False otherwise.
    """
    n = len(adj)
    degrees = [0] * n
    for i in range(n):
        degrees[i] = bin(adj[i]).count('1')
    
    order = sorted(range(n), key=lambda x: -degrees[x])
    colors = [-1] * n
    
    def backtrack(v_idx):
        if v_idx == n:
            return True
        
        v = order[v_idx]
        mask = 0
        neighbors = adj[v]
        while neighbors:
            lb = neighbors & (-neighbors)
            u = lb.bit_length() - 1
            neighbors ^= lb
            if colors[u] != -1:
                mask |= (1 << colors[u])
        
        available = ~mask & 0b111
        if available == 0:
            return False
        
        for c in range(3):
            if available & (1 << c):
                colors[v] = c
                if backtrack(v_idx + 1):
                    return True
                colors[v] = -1
        return False
    
    return backtrack(0)

def _greedy_add_edges(adj, edge_set, non_edges, rng, n=16):
    """
    Greedily add edges from non_edges that don't create triangles.
    Returns updated adj, edge_set, non_edges.
    """
    changed = True
    while changed:
        changed = False
        # Shuffle non_edges to randomize order
        rng.shuffle(non_edges)
        for i in range(len(non_edges) - 1, -1, -1):
            u, v = non_edges[i]
            # Check if adding (u, v) creates a triangle
            # Triangle exists if u and v share a common neighbor
            if adj[u] & adj[v] == 0:
                # Safe to add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                non_edges.pop(i)
                changed = True
                # Continue checking other edges? 
                # Since we added an edge, some other non_edges might now be invalid.
                # But we just check adj[u] & adj[v] for the specific edge.
                # Other edges are not affected by this addition in terms of *their* triangle check with *each other*, 
                # only their check with the *current* graph.
                # So we can continue the loop.
    return adj, edge_set, non_edges

def _local_search(seed, deadline):
    n = 16
    rng = random.Random(seed)
    
    # Initialize with Grotzsch
    grotzsch = _build_grotzsch_edges()
    adj = _make_bitmask_adj(grotzsch, n)
    edge_set = set((min(u, v), max(u, v)) for u, v in grotzsch)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    non_edges = [e for e in all_edges if e not in edge_set]
    
    # Initial greedy fill
    adj, edge_set, non_edges = _greedy_add_edges(adj, edge_set, non_edges, rng, n)
    
    best_count = len(edge_set)
    best_adj = adj[:]
    
    # Local Search: Remove one edge, greedily refill, keep if better
    while time.time() < deadline:
        if not edge_set:
            break
            
        # Pick a random edge to remove
        # Heuristic: Remove edges that are part of many "potential triangles" if we were to add other edges?
        # For now, random is a good baseline.
        e_out = rng.choice(list(edge_set))
        u_out, v_out = e_out
        
        # Save state
        saved_adj = adj[:]
        saved_edge_set = set(edge_set)
        saved_non_edges = non_edges[:]
        
        # Remove edge
        adj[u_out] &= ~(1 << v_out)
        adj[v_out] &= ~(1 << u_out)
        edge_set.remove(e_out)
        non_edges.append(e_out)
        
        # Greedily refill
        adj, edge_set, non_edges = _greedy_add_edges(adj, edge_set, non_edges, rng, n)
        
        # Check if improvement
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_adj = adj[:]
        elif len(edge_set) == best_count:
            # Tie: keep current or revert? 
            # Usually keep current to explore different local optima.
            pass
        else:
            # Revert
            adj = saved_adj
            edge_set = saved_edge_set
            non_edges = saved_non_edges
            
    # Convert best_adj to edge list
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result_edges.append([u, v])
    
    return result_edges, len(result_edges)

def construct():
    """
    Construct a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Run multiple instances of local search with different seeds
    seed = 0
    while time.time() < deadline - 5.0:
        edges, count = _local_search(seed, deadline)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        # If we reach a very high count, we might stop early? 
        # Max possible edges in triangle-free graph on 16 vertices is floor(16^2/4) = 64 (Mantel's Theorem).
        # But it must contain a 4-chromatic core.
        # If we hit 60+, it's likely optimal or close.
        if best_count >= 60:
            break

    # Final validation
    if best_edges:
        adj = _make_bitmask_adj(best_edges)
        if not _is_valid_triangle_free(adj):
            # Fallback to Grotzsch
            grotzsch = _build_grotzsch_edges()
            best_edges = list(grotzsch)
        elif _is_3_colorable(adj):
            # This should not happen if we started with Grotzsch and only added edges,
            # but just in case of a bug in logic.
            grotzsch = _build_grotzsch_edges()
            best_edges = list(grotzsch)
    else:
        best_edges = _build_grotzsch_edges()
    
    return best_edges
# EVOLVE-BLOCK-END
