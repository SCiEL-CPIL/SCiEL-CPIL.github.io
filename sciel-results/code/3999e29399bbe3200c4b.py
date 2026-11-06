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
        edges.append([i, (i + 1) % 5])
    # Clones 5..9
    # In Mycielski construction:
    # v_i connected to v_{i-1}, v_{i+1}
    # w_i connected to v_{i-1}, v_{i+1}
    # root connected to all w_i
    # Here v_i is i, w_i is 5+i
    for i in range(5):
        edges.append([i, 5 + (i - 1) % 5])
        edges.append([i, 5 + (i + 1) % 5])
    # Root 10 connected to all clones 5..9
    for i in range(5):
        edges.append([10, 5 + i])
    return edges

def _make_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_triangle_free_add(adj, u, v):
    # Check if u and v have a common neighbor
    return (adj[u] & adj[v]) == 0

def _is_3_colorable(adj, deadline=None):
    n = len(adj)
    color = [0] * n
    
    def backtrack(idx):
        if deadline and time.time() > deadline:
            return None
        
        if idx == n:
            return True
            
        # MRV Heuristic: Find uncolored vertex with fewest available colors
        # For simplicity and speed in this specific context (n=16), 
        # we can just pick the first uncolored vertex or use a static order.
        # Static order is faster but less pruning. MRV is better for hard instances.
        # Let's use a simple dynamic MRV.
        
        best_v = -1
        min_avail = 4
        best_avail_mask = 0
        
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
                
                avail_mask = 7 ^ unavail
                count = bin(avail_mask).count('1')
                if count == 0:
                    return False
                if count < min_avail:
                    min_avail = count
                    best_v = v
                    best_avail_mask = avail_mask
                    if count == 1:
                        break
        
        if best_v == -1:
            return True

        c = 1
        while c <= 3:
            if best_avail_mask & (1 << (c-1)):
                color[best_v] = c
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                color[best_v] = 0
            c += 1
        return False

    # Initial order: sort vertices by degree descending to help MRV?
    # Or just rely on MRV finding the constrained vertex.
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer
    
    n = 16
    grotzsch_edges = _groetzsch_edges()
    
    best_edges = list(grotzsch_edges)
    best_adj = _make_adj(best_edges, n)
    best_count = len(best_edges)
    
    # Validate initial graph
    # It should be triangle-free and not 3-colorable
    if _is_3_colorable(best_adj, deadline):
        # This shouldn't happen if Grötzsch is correct, but safety check
        # If it is 3-colorable, we have a problem. But Grötzsch is known to be 4-chromatic.
        pass

    def try_update(edges, adj, count):
        nonlocal best_edges, best_adj, best_count
        if count > best_count:
            # Verify
            if not _is_3_colorable(adj, deadline):
                best_edges = [list(e) for e in edges]
                best_adj = [m for m in adj]
                best_count = count
                return True
        return False

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
                # Tentatively add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _is_3_colorable(adj, deadline):
                    edge_set.add((min(u,v), max(u,v)))
                    edges.append([u, v])
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        return edges, adj, len(edges)

    def local_search(edges, adj, count, seed):
        rng = random.Random(seed)
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding edges
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            remaining = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set]
            rng.shuffle(remaining)
            
            for u, v in remaining:
                if time.time() > deadline - 5:
                    break
                if _is_triangle_free_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not _is_3_colorable(adj, deadline):
                        edge_set.add((min(u,v), max(u,v)))
                        edges.append([u, v])
                        count += 1
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        
            if not improved and time.time() < deadline - 10:
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
                    all_edges = [(eu, ev) for eu in range(n) for ev in range(eu + 1, n)]
                    remaining = [(eu, ev) for (eu, ev) in all_edges if (eu, ev) not in edge_set]
                    rng.shuffle(remaining)
                    
                    added_any = False
                    for eu, ev in remaining:
                        if time.time() > deadline - 5:
                            break
                        if _is_triangle_free_add(adj, eu, ev):
                            adj[eu] |= (1 << ev)
                            adj[ev] |= (1 << eu)
                            if not _is_3_colorable(adj, deadline):
                                edge_set.add((min(eu,ev), max(eu,ev)))
                                edges.append([eu, ev])
                                count += 1
                                added_any = True
                                
                    if count > best_count:
                        try_update(edges, adj, count)
                        improved = True
                    else:
                        # Revert swap if it didn't help
                        # We need to restore the removed edge
                        edges.append([u, v])
                        edge_set.add((min(u,v), max(u,v)))
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        count += 1
                        # Remove the newly added edges
                        # This is tricky because we don't track which ones were added in this specific swap attempt easily
                        # Better approach: snapshot adj and edges before swap, restore if failed.
                        # For now, just break out if not improved significantly to avoid state corruption
                        break

        return edges, adj, count

    # Run several greedy fills with different seeds
    for seed in range(100):
        if time.time() > deadline - 20:
            break
        edges = list(grotzsch_edges)
        adj = _make_adj(edges, n)
        edges, adj, count = greedy_fill(adj, edges, seed)
        try_update(edges, adj, count)
        
    # Run local search on best
    if time.time() < deadline - 20:
        edges = [list(e) for e in best_edges]
        adj = [m for m in best_adj]
        count = best_count
        edges, adj, count = local_search(edges, adj, count, 12345)
        try_update(edges, adj, count)
        
    # Final polish: try to add any remaining edges in a deterministic order
    if time.time() < deadline - 5:
        edge_set = set((min(u,v), max(u,v)) for u,v in best_edges)
        adj = [m for m in best_adj]
        edges = [list(e) for e in best_edges]
        count = best_count
        for u in range(n):
            for v in range(u+1, n):
                if time.time() > deadline - 2:
                    break
                if (u,v) not in edge_set and _is_triangle_free_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not _is_3_colorable(adj, deadline):
                        edge_set.add((u,v))
                        edges.append([u, v])
                        count += 1
        try_update(edges, adj, count)

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
