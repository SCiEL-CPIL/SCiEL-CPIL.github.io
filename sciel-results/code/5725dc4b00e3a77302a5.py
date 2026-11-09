"""Construct a 4-regular graph with girth >= 10 on as few vertices as possible."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _find_short_cycle(adj, n, max_len):
    """Find a specific short cycle (length <= max_len) using non-backtracking BFS.
    Returns a list of vertices forming the cycle, or None."""
    for start in range(n):
        # Non-backtracking BFS
        # State: (current_vertex, previous_vertex) -> (depth, path)
        # To save memory and time, we only store depth and parent pointer for path reconstruction if needed
        # But for small depths (<= 5 for radius), we can just store the path.
        visited = {}
        queue = deque()
        for v in adj[start]:
            queue.append((v, start))
            visited[(v, start)] = (1, [start, v])
        
        while queue:
            u, prev = queue.popleft()
            depth, path = visited[(u, prev)]
            if depth >= max_len:
                continue
            for w in adj[u]:
                if w == prev:
                    continue
                if w == start and depth + 1 >= 3:
                    # Found a cycle
                    # path is [start, ..., u]
                    # Cycle is path + [start]
                    # Length is depth + 1
                    if depth + 1 <= max_len:
                        return path + [start]
                elif (w, u) not in visited:
                    visited[(w, u)] = (depth + 1, path + [w])
                    queue.append((w, u))
    return None

def _has_cycle_of_length_less_than(adj, n, limit):
    """Check if there is any cycle of length < limit.
    Uses non-backtracking BFS from each vertex up to depth limit//2 - 1.
    If a walk of length L < limit closes a cycle, we return True.
    """
    max_depth = (limit - 1) // 2
    # We need to detect cycles of length 3, 4, ..., limit-1.
    # BFS from start, depth d. If we hit a vertex at depth d' such that d + d' + 1 < limit, we have a short cycle.
    # Simpler: Just run _find_short_cycle with max_len = limit - 1.
    # But _find_short_cycle is a bit slow due to path storage.
    # Let's optimize _find_short_cycle to not store full paths if we just need existence?
    # No, we need the cycle to break it.
    return _find_short_cycle(adj, n, limit - 1) is not None

def _random_k_regular(n, k, rng):
    """Generate a random k-regular graph using configuration model."""
    if n * k % 2 != 0:
        raise ValueError("n*k must be even")
    
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    rng.shuffle(half_edges)
    
    adj = [set() for _ in range(n)]
    for i in range(0, n * k, 2):
        u, v = half_edges[i], half_edges[i + 1]
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    
    # Fix multi-edges/self-loops
    # This simple loop might not fix everything, but for k=4 it's usually okay or we restart.
    # A more robust fix:
    for _ in range(100):
        bad = False
        for u in range(n):
            if len(adj[u]) != k:
                bad = True
                break
        if not bad:
            break
        
        # Find a vertex with wrong degree and fix
        for u in range(n):
            if len(adj[u]) < k:
                # Need to add edge
                candidates = [v for v in range(n) if v != u and v not in adj[u] and len(adj[v]) < k]
                if candidates:
                    v = rng.choice(candidates)
                    adj[u].add(v)
                    adj[v].add(u)
            elif len(adj[u]) > k:
                # Remove edge
                v = rng.choice(list(adj[u]))
                adj[u].discard(v)
                adj[v].discard(u)
    
    adj_list = [sorted(adj[i]) for i in range(n)]
    for i in range(n):
        if len(adj_list[i]) != k:
            raise ValueError(f"Vertex {i} has degree {len(adj_list[i])} instead of {k}")
    return adj_list

def _apply_two_switch(adj, n, k, edge_to_swap, rng):
    """Apply a 2-switch involving a specific edge (u, v) from adj.
    Returns new_adj if successful, else None.
    """
    u, v = edge_to_swap
    if v not in adj[u]:
        return None
        
    # Find another edge (x, y) to swap with
    # x, y must be distinct from u, v and not adjacent to u, v in a way that creates loops/multiedges
    # Swap (u, v) and (x, y) -> (u, x) and (v, y)
    # Conditions:
    # 1. x != y
    # 2. {x, y} disjoint from {u, v}
    # 3. x not in adj[u] (unless x==v, but x!=v)
    # 4. y not in adj[v] (unless y==u, but y!=u)
    # 5. u not in adj[y] ? No, we are removing (u, v) and (x, y).
    #    We add (u, x) and (v, y).
    #    So we need x not in adj[u] and y not in adj[v].
    #    Also we need u not in adj[y] is not required for validity of the swap itself, 
    #    but we must ensure we don't create multi-edges.
    #    Since (u, v) is removed, u is no longer connected to v.
    #    Since (x, y) is removed, x is no longer connected to y.
    #    New edges: (u, x) and (v, y).
    #    We need x != u, x != v, y != u, y != v.
    #    We need x not in adj[u] (to avoid multi-edge with existing edge, but (u,v) is removed so if x==v it's ok? No x!=v).
    #    We need y not in adj[v].
    
    # Collect all edges
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
                
    # Filter valid edges
    valid_edges = []
    for x, y in edges:
        if x in (u, v) or y in (u, v):
            continue
        # Check if swap is valid
        # Remove (u, v) and (x, y). Add (u, x) and (v, y).
        # u-x valid if x not in adj[u] (after removing v).
        # v-y valid if y not in adj[v] (after removing u).
        # Since x, y are not u, v, and we are removing (u, v), 
        # x is in adj[u] only if there was another edge (u, x).
        if x in adj[u] or y in adj[v]:
            continue
        # Also check if (u, y) or (v, x) exist? No, we are adding (u, x) and (v, y).
        # We must ensure we don't create a multi-edge.
        # If (u, x) already exists, we can't add it.
        # So x must NOT be in adj[u].
        # And y must NOT be in adj[v].
        valid_edges.append((x, y))
        
    if not valid_edges:
        return None
        
    x, y = rng.choice(valid_edges)
    
    # Apply swap
    new_adj = [list(s) for s in adj]
    
    # Remove (u, v)
    new_adj[u].remove(v)
    new_adj[v].remove(u)
    
    # Remove (x, y)
    new_adj[x].remove(y)
    new_adj[y].remove(x)
    
    # Add (u, x)
    new_adj[u].append(x)
    new_adj[x].append(u)
    
    # Add (v, y)
    new_adj[v].append(y)
    new_adj[y].append(v)
    
    # Sort and verify
    for i in range(n):
        new_adj[i] = sorted(new_adj[i])
        if len(new_adj[i]) != k:
            return None
            
    return new_adj

def _improve_girth(adj, n, k, target_girth, rng, deadline):
    """Try to increase girth to target_girth using targeted 2-switches."""
    current_adj = [list(s) for s in adj]
    
    while time.time() < deadline:
        # Find a short cycle
        cycle = _find_short_cycle(current_adj, n, target_girth - 1)
        if not cycle:
            return current_adj # Success
        
        # Pick a random edge from the cycle
        # Cycle is [v0, v1, ..., v_{L-1}, v0]
        # Edges are (v0, v1), (v1, v2), ...
        L = len(cycle) - 1
        edge_idx = rng.randint(0, L - 1)
        u = cycle[edge_idx]
        v = cycle[(edge_idx + 1) % L]
        
        # Try to apply a 2-switch on this edge
        new_adj = _apply_two_switch(current_adj, n, k, (u, v), rng)
        
        if new_adj:
            # Check if the new graph is valid (no short cycles)
            # To be efficient, we just check if it has a cycle < target_girth
            # If it does, we might still accept it if it's "better" (e.g. shorter min cycle), 
            # but for simplicity and speed, let's require it to be free of short cycles 
            # OR just accept if it doesn't introduce a *new* short cycle of length < min(current_min, target).
            # Actually, just checking _find_short_cycle is expensive.
            # Let's accept the switch if it breaks the specific cycle we found?
            # No, it might create another one.
            
            # Heuristic: Accept if the new graph has NO cycle of length < target_girth.
            # This is a strong condition. If we can't satisfy it, we are stuck.
            # Let's try a few random switches on this edge.
            accepted = False
            for _ in range(10):
                new_adj = _apply_two_switch(current_adj, n, k, (u, v), rng)
                if not new_adj:
                    continue
                # Check girth
                if _find_short_cycle(new_adj, n, target_girth - 1) is None:
                    current_adj = new_adj
                    accepted = True
                    break
            if accepted:
                continue
                
            # If we failed to break the cycle with 10 tries, try a different edge in the cycle
            # or restart?
            # Let's just continue the loop, it will pick the same cycle again probably.
            # To avoid infinite loop on same cycle, we can shuffle the edge selection.
            # The rng.choice in _apply_two_switch provides randomness.
            
            # If we are stuck, we might need to restart the whole graph generation.
            # Let's add a counter for consecutive failures.
            # For now, just keep trying.
            
    return current_adj

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 110
    
    # Moore bound
    if g % 2 == 0:
        moore = 1 + k * (k - 1) ** (g // 2 - 1)
    else:
        moore = 2 + k * (k - 1) ** ((g - 1) // 2 - 1) if (g-1)//2 - 1 >= 0 else 2 + k
    
    rng = random.Random(42)
    
    best_n = None
    best_edges = None
    
    # Try sizes from known best downwards? Or upwards from moore?
    # Known best for 4,10 is 320.
    # Let's try 320 first. If we find a valid graph, we return it.
    # If we have time, we try 300, 280, etc.
    
    start_sizes = [320, 300, 280, 260, 250, 245, 244]
    start_sizes = [s for s in start_sizes if s >= moore and s * k % 2 == 0]
    
    for n_try in start_sizes:
        if time.time() > deadline - 20:
            break
            
        # Try multiple random initializations
        for init in range(10):
            if time.time() > deadline - 10:
                break
                
            try:
                adj = _random_k_regular(n_try, k, rng)
            except Exception:
                continue
            
            # Check if already good
            if _find_short_cycle(adj, n_try, g - 1) is None:
                edges = []
                for u in range(n_try):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                if best_n is None or n_try < best_n:
                    best_n = n_try
                    best_edges = edges
                break # Found valid graph for this n_try
            
            # Try to improve
            improved_adj = _improve_girth(adj, n_try, k, g, rng, deadline - 5)
            
            if _find_short_cycle(improved_adj, n_try, g - 1) is None:
                edges = []
                for u in range(n_try):
                    for v in improved_adj[u]:
                        if u < v:
                            edges.append((u, v))
                if best_n is None or n_try < best_n:
                    best_n = n_try
                    best_edges = edges
                break # Found valid graph for this n_try
        
        # If we found a valid graph for this n_try, we can stop trying smaller ones?
        # No, we want the SMALLEST n.
        # So if we found a valid graph for 320, we should still try 300 if we have time.
        # But if we found one for 300, we should try 280.
        # So we iterate start_sizes in increasing order?
        # start_sizes is [320, 300, ...].
        # If we find a valid graph for 320, we store it.
        # Then we try 300. If we find one, we update best.
        # If we don't find one for 300, we keep 320.
        
    if best_n is not None and best_edges is not None:
        return best_n, best_edges
        
    # Fallback: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
