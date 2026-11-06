import time
import random
import sys

def _groetzsch_edges():
    """
    Returns edges of the Grötzsch graph (Mycielski of C5).
    Vertices 0..4 form C5.
    Vertices 5..9 are clones.
    Vertex 10 is the root.
    """
    edges = []
    # C5 on 0..4
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Clones 5..9
    # u in C5 connected to clone of u+1 and u-1?
    # Mycielski: v_i connected to v_{i-1}, v_{i+1} (in C5)
    # w_i connected to v_{i-1}, v_{i+1}
    # root connected to all w_i
    # Let's use the standard construction from the seed which worked.
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[10, 5 + u] for u in range(5)]
    return edges

def _chvatal_edges():
    """
    Returns edges of the Chvátal graph on 12 vertices.
    """
    # Chvátal graph is a specific 12-vertex triangle-free 4-chromatic graph.
    # We can construct it or hardcode it.
    # Hardcoding is safer.
    # Reference: https://en.wikipedia.org/wiki/Chv%C3%A1tal_graph
    edges = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 6], [1, 7], [1, 8], [1, 9],
        [2, 6], [2, 7], [2, 10], [2, 11],
        [3, 8], [3, 9], [3, 10], [3, 11],
        [4, 6], [4, 9], [4, 10],
        [5, 7], [5, 8], [5, 11],
        [6, 10], [6, 11],
        [7, 9], [7, 11],
        [8, 10], [8, 11]
    ]
    # Verify triangle-free and non-3-colorable in code? 
    # Let's assume this is correct. 
    # Actually, let's use a known valid list.
    # The seed used Grötzsch. Grötzsch is 11 vertices.
    # Let's stick to Grötzsch for reliability, but maybe expand it better.
    return edges

def _make_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_triangle_free_add(adj, u, v):
    return (adj[u] & adj[v]) == 0

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True

# Pre-compute bitmasks for Grötzsch and Chvátal induced subgraph checks
# This is complex to implement generically for arbitrary subsets.
# Instead, we will just use the fast backtracking 3-colorer.
# For n=16, it's fast enough.

def _is_3_colorable(adj, deadline=None):
    n = len(adj)
    color = [0] * n
    
    # Order vertices by degree descending for better pruning?
    # Or use MRV dynamically.
    
    def backtrack():
        if deadline and time.time() > deadline:
            return None
        
        # MRV
        best_v = -1
        min_avail = 4
        best_avail_colors = 0
        
        for v in range(n):
            if color[v] == 0:
                unavail = 0
                mask = adj[v]
                while mask:
                    lb = mask & (-mask)
                    u = lb.bit_length() - 1
                    mask ^= lb
                    if color[u] != 0:
                        unavail |= (1 << (color[u]-1))
                avail_colors = 7 ^ unavail
                count = bin(avail_colors).count('1')
                if count == 0:
                    return False
                if count < min_avail:
                    min_avail = count
                    best_v = v
                    best_avail_colors = avail_colors
                    if count == 1:
                        break
        
        if best_v == -1:
            return True
            
        c = 1
        while c <= 3:
            if best_avail_colors & (1 << (c-1)):
                color[best_v] = c
                result = backtrack()
                if result is not None:
                    return result
                color[best_v] = 0
            c += 1
        return False

    result = backtrack()
    return result if result is not None else True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    # Start with Grötzsch graph on vertices 0..10
    grotzsch_edges = _groetzsch_edges()
    
    best_edges = list(grotzsch_edges)
    best_adj = _make_adj(best_edges, n)
    best_count = len(best_edges)
    
    # Function to validate and update best
    def try_update(edges, adj, count):
        nonlocal best_edges, best_adj, best_count
        if count > best_count:
            # Verify
            if _is_triangle_free(adj) and not _is_3_colorable(adj, deadline=start_time + 115.0):
                best_edges = edges
                best_adj = adj
                best_count = count
                return True
        return False

    # Initial Greedy Fill
    def greedy_fill(adj, edges, seed):
        rng = random.Random(seed)
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        remaining = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set]
        rng.shuffle(remaining)
        
        for u, v in remaining:
            if time.time() > deadline:
                break
            if _is_triangle_free_add(adj, u, v):
                # Check 3-colorability
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _is_3_colorable(adj, deadline=start_time + 115.0):
                    edge_set.add((min(u,v), max(u,v)))
                    edges.append([u, v])
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        return edges, adj, len(edges)

    # Run several greedy fills with different seeds
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        edges = list(grotzsch_edges)
        adj = _make_adj(edges, n)
        edges, adj, count = greedy_fill(adj, edges, seed)
        try_update(edges, adj, count)
        
    # Local Search on the best found so far
    def local_search(edges, adj, count, seed):
        rng = random.Random(seed)
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding edges
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            remaining = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set]
            rng.shuffle(remaining)
            
            for u, v in remaining:
                if time.time() > deadline:
                    break
                if _is_triangle_free_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not _is_3_colorable(adj, deadline=start_time + 115.0):
                        edge_set.add((min(u,v), max(u,v)))
                        edges.append([u, v])
                        count += 1
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        
            if not improved and time.time() < deadline - 5:
                # Try edge swap: remove one, add many
                if edges:
                    # Remove a random edge
                    idx = rng.randrange(len(edges))
                    u, v = edges.pop(idx)
                    edge_set.discard((min(u,v), max(u,v)))
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    count -= 1
                    
                    # Try to add edges again
                    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
                    remaining = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set]
                    rng.shuffle(remaining)
                    
                    for eu, ev in remaining:
                        if time.time() > deadline:
                            break
                        if _is_triangle_free_add(adj, eu, ev):
                            adj[eu] |= (1 << ev)
                            adj[ev] |= (1 << eu)
                            if not _is_3_colorable(adj, deadline=start_time + 115.0):
                                edge_set.add((min(eu,ev), max(eu,ev)))
                                edges.append([eu, ev])
                                count += 1
                                
                    if count > best_count:
                        try_update(edges, adj, count)
                        improved = True
                    else:
                        # Revert swap if it didn't help
                        # This is complex to revert exactly. 
                        # Better to just restart from best if stuck.
                        pass

        return edges, adj, count

    # Run local search on best
    if time.time() < deadline - 10:
        edges, adj, count = local_search(list(best_edges), _make_adj(best_edges, n), best_count, 123)
        try_update(edges, adj, count)
        
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
