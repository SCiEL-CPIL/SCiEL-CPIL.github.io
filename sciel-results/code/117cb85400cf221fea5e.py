import time
import random

_START = time.time()
_DEADLINE = _START + 108

def _has_path_5(u, v, adj_masks):
    # BFS/DFS for simple path of length 5 from u to v
    # We need exactly 5 edges, so 6 vertices: u = v0, v1, v2, v3, v4, v5 = v
    # Use bitmask for visited nodes (n=40 fits in int)
    # State: (node, depth, visited_mask)
    # We can prune: if remaining depth > shortest possible distance to v, skip
    # But computing shortest path each time is expensive, so just use DFS with good pruning
    
    # Quick check: if u and v are adjacent, we need path of length 5 between them
    # which means a cycle of length 6 through that edge
    
    stack = [(u, 0, 1 << u)]
    while stack:
        node, depth, mask = stack.pop()
        if depth == 5:
            if node == v:
                return True
            continue
        if depth == 4:
            # Next node must be v
            if adj_masks[node] & (1 << v) and not (mask & (1 << v)):
                return True
            continue
        neighbors = adj_masks[node]
        while neighbors:
            bit = neighbors & -neighbors
            nb = bit.bit_length() - 1
            neighbors ^= bit
            if mask & (1 << nb):
                continue
            if nb == v and depth != 4:
                continue
            stack.append((nb, depth + 1, mask | (1 << nb)))
    return False

def _greedy_with_local_search(n=40, seed=0, deadline=None):
    if deadline is None:
        deadline = _DEADLINE
    rng = random.Random(seed)
    adj_masks = [0] * n
    edges = set()
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    # Greedy Phase
    for u, v in all_edges:
        if time.time() > deadline - 2:
            break
        if not _has_path_5(u, v, adj_masks):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.add((u, v))
            
    # Local Search Phase - try to add non-edges
    non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
    rng.shuffle(non_edges)
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for u, v in non_edges:
            if (u, v) in edges:
                continue
            if not _has_path_5(u, v, adj_masks):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
                improved = True
                
    # Local Search Phase - try edge swaps (remove one edge, add up to two non-edges)
    if time.time() < deadline - 3:
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            # Pick a random edge to remove
            edge_list = list(edges)
            if not edge_list:
                break
            ru, rv = rng.choice(edge_list)
            adj_masks[ru] &= ~(1 << rv)
            adj_masks[rv] &= ~(1 << ru)
            edges.discard((ru, rv))
            
            # Try to add up to 2 non-edges
            added = 0
            for u, v in non_edges:
                if (u, v) in edges:
                    continue
                if not _has_path_5(u, v, adj_masks):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges.add((u, v))
                    added += 1
                    if added >= 2:
                        break
            # If we couldn't add 2, try more
            if added < 2:
                for u, v in non_edges:
                    if (u, v) in edges:
                        continue
                    if not _has_path_5(u, v, adj_masks):
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        edges.add((u, v))
                        added += 1
                        if added >= 2:
                            break
    
    return list(edges)

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 108
    
    best_edges = []
    best_count = 0
    
    # Use multiple seeds with different strategies
    for seed in range(10000):
        if time.time() > _DEADLINE - 2:
            break
        
        try:
            edges = _greedy_with_local_search(40, seed)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
                if count >= 105:
                    break
        except Exception:
            continue
            
    return [[u, v] for u, v in best_edges]
