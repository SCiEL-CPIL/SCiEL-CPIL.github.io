import time
import random
import sys

def _make_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_triangle_free_add(adj, u, v):
    return (adj[u] & adj[v]) == 0

def _is_triangle_free(adj, n=16):
    for u in range(n):
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            if v > u:
                if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                    return False
    return True

def _is_3_colorable(adj, n=16, deadline=None):
    # Fast backtracking with MRV
    color = [0] * n
    adj_masks = adj[:]
    
    # Static ordering by degree might be faster than dynamic MRV for small n?
    # Let's stick to dynamic MRV as it prunes better.
    
    def backtrack():
        if deadline and time.time() > deadline:
            return None
        
        best_v = -1
        min_avail = 4
        best_avail_colors = 0
        
        for v in range(n):
            if color[v] == 0:
                unavail = 0
                mask = adj_masks[v]
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

def _build_grotzsch_edges():
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[10, 5 + u] for u in range(5)]
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 16
    
    best_edges = []
    best_count = 0
    
    def try_update(edges, adj, edge_set, count):
        nonlocal best_edges, best_count
        if count > best_count:
            if _is_triangle_free(adj, n) and not _is_3_colorable(adj, n):
                best_edges = edges
                best_count = count
                return True
        return False

    for seed in range(1000):
        if time.time() > deadline - 10:
            break
            
        rng = random.Random(seed)
        
        # Start with Grötzsch graph
        base_edges = _build_grotzsch_edges()
        edges = list(base_edges)
        adj = _make_adj(edges, n)
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        
        # Greedy completion
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
        rng.shuffle(all_possible)
        
        for u, v in all_possible:
            if time.time() > deadline - 10:
                break
            if _is_triangle_free_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _is_3_colorable(adj, n):
                    edge_set.add((u, v))
                    edges.append([u, v])
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # 1. Try adding edges
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
            rng.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline - 10:
                    break
                if _is_triangle_free_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not _is_3_colorable(adj, n):
                        edge_set.add((u, v))
                        edges.append([u, v])
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            
            # 2. Swap
            if not improved and edges:
                # Try a few random swaps
                for _ in range(10):
                    if time.time() > deadline - 10:
                        break
                    idx = rng.randrange(len(edges))
                    u, v = edges[idx]
                    
                    # Remove
                    edges.pop(idx)
                    edge_set.discard((u, v))
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Add back greedily
                    missing = [(eu, ev) for eu in range(n) for ev in range(eu + 1, n) if (eu, ev) not in edge_set]
                    rng.shuffle(missing)
                    
                    added_any = False
                    for eu, ev in missing:
                        if time.time() > deadline - 10:
                            break
                        if _is_triangle_free_add(adj, eu, ev):
                            adj[eu] |= (1 << ev)
                            adj[ev] |= (1 << eu)
                            if not _is_3_colorable(adj, n):
                                edge_set.add((eu, ev))
                                edges.append([eu, ev])
                                added_any = True
                            else:
                                adj[eu] &= ~(1 << ev)
                                adj[ev] &= ~(1 << eu)
                                
                    if added_any and len(edges) > best_count: # Or just if added_any
                        improved = True
                        break
                    else:
                        # Revert
                        edges.append([u, v])
                        edge_set.add((u, v))
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

        try_update(edges, adj, edge_set, len(edges))
        
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
