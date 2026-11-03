import time
import random
import sys
import heapq
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = float('inf')
    best_edges = None
    
    # First try cycle-elimination local search for smaller graphs
    if k == 3:
        try:
            n_local, edges_local = _cycle_elimination_search(k, g, deadline)
            if n_local < best_n:
                best_n = n_local
                best_edges = edges_local
        except Exception:
            pass
    
    # If local search didn't find anything good, try LUW construction
    if best_n == float('inf') or best_n > 600:
        min_d = max(3, g // 2 - 2)
        max_d = g + 5
        
        rule_variants = _generate_rule_variants(k, max_d)
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline - 15:
                break
            
            found_valid_for_d = False
            best_n_for_d = float('inf')
            best_edges_for_d = None
            
            for variant_idx, rule in enumerate(rule_variants):
                if time.time() > deadline - 15:
                    break
                try:
                    n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                except Exception:
                    continue
                
                if n_luw < best_n_for_d:
                    if _has_girth_at_least_fast(n_luw, edges_luw, g):
                        best_n_for_d = n_luw
                        best_edges_for_d = edges_luw
                        found_valid_for_d = True
            
            if found_valid_for_d:
                if best_n_for_d < best_n:
                    best_n = best_n_for_d
                    best_edges = best_edges_for_d
                break
    
    if best_n == float('inf'):
        # Last resort fallback
        rule_variants = _generate_rule_variants(k, max_d)
        for variant_idx, rule in enumerate(rule_variants):
            if time.time() > deadline - 5:
                break
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, max_d, rule)
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    return n_luw, edges_luw
            except Exception:
                continue
        n_luw, edges_luw = _construct_luw_with_rule(k, max_d + 3, rule_variants[0])
        return n_luw, edges_luw

    return best_n, best_edges

def _cycle_elimination_search(k, g, deadline):
    """
    Start with a random k-regular graph near the Moore bound,
    then iteratively apply 2-switches to break short cycles.
    """
    # Moore bound for k=3, g=16: 2*(3^(g/2-1) - 1) = 2*(3^7 - 1) = 4374? No wait.
    # For odd girth g=2r+1: n >= 2 * sum_{i=0}^{r-1} (k-1)^i
    # For even girth g=2r: n >= k * sum_{i=0}^{r-1} (k-1)^i
    # g=16, r=8: n >= 3 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 3 * 255 = 765
    # Actually the Moore bound for (3,16) is 3*(2^8 - 1)/(2-1) = 3*255 = 765
    # But best lower bound is 512, so let's try sizes from 512 upward
    
    moore_bound = k * (2**(g//2) - 1) // (k - 1) if k > 1 else g
    # For k=3, g=16: 3*(2^8 - 1)/2 = 3*255/2... that's not integer.
    # Correct formula: for even g=2r, n >= k * sum_{i=0}^{r-1} (k-1)^i
    # r=8, k=3: 3*(1+2+4+8+16+32+64+128) = 3*255 = 765
    # But the problem says lower bound is 512, so maybe I'm using wrong formula.
    # Let's just try a range of sizes.
    
    random.seed(42)
    
    # Try several starting sizes
    start_sizes = [512, 528, 544, 560, 576, 592, 608, 624, 640]
    
    best_n = float('inf')
    best_edges = None
    
    for n_start in start_sizes:
        if time.time() > deadline - 20:
            break
        if n_start % 2 != 0:
            continue  # Must be even for 3-regular graph
        if n_start * k % 2 != 0:
            continue  # Sum of degrees must be even
        
        # Generate random 3-regular graph
        try:
            adj = _random_regular_graph(n_start, k)
        except Exception:
            continue
        
        # Iterate: find shortest cycle, apply 2-switch to break it
        for iteration in range(5000):
            if time.time() > deadline - 20:
                break
            
            # Find shortest cycle
            cycle = _find_shortest_cycle(adj, n_start, g)
            if cycle is None:
                # No cycle shorter than g found
                edges = _adj_to_edges(adj, n_start)
                if n_start < best_n:
                    best_n = n_start
                    best_edges = edges
                break
            
            cycle_len = len(cycle)
            if cycle_len >= g:
                edges = _adj_to_edges(adj, n_start)
                if n_start < best_n:
                    best_n = n_start
                    best_edges = edges
                break
            
            # Apply a 2-switch to break this cycle
            adj = _apply_2switch(adj, cycle, n_start, k)
            if adj is None:
                break
        
        # After local search, verify girth
        if n_start < best_n:
            edges = _adj_to_edges(adj, n_start)
            if _has_girth_at_least_fast(n_start, edges, g):
                best_n = n_start
                best_edges = edges
    
    return best_n, best_edges

def _random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        raise ValueError("n*k must be even")
    
    # Create stubs
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    
    random.shuffle(stubs)
    
    # Pair stubs
    adj = [[] for _ in range(n)]
    edge_set = set()
    
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            # Self-loop, retry
            # Find a different partner
            for j in range(i+2, n*k, 2):
                if stubs[j] != u:
                    stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                    v = stubs[i+1]
                    break
            else:
                raise ValueError("Could not avoid self-loop")
        
        if (min(u,v), max(u,v)) in edge_set:
            # Multi-edge, retry
            for j in range(i+2, n*k, 2):
                if stubs[j] != u and (min(u, stubs[j]), max(u, stubs[j])) not in edge_set:
                    stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                    v = stubs[i+1]
                    break
            else:
                raise ValueError("Could not avoid multi-edge")
        
        edge_set.add((min(u,v), max(u,v)))
        adj[u].append(v)
        adj[v].append(u)
    
    return adj

def _find_shortest_cycle(adj, n, g):
    """Find a cycle shorter than g using BFS from each vertex."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] + 1 >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    if u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            cycle = [u]
                            cur = u
                            while cur != start:
                                cur = parent[cur]
                                cycle.append(cur)
                            # Add the other path
                            cycle2 = [v]
                            cur = v
                            while cur != start:
                                cur = parent[cur]
                                cycle2.append(cur)
                            # Combine
                            cycle.extend(cycle2[1:])
                            return cycle
    return None

def _apply_2switch(adj, cycle, n, k):
    """
    Apply a 2-switch to break a cycle.
    Pick two non-adjacent edges in the cycle and swap endpoints.
    """
    if len(cycle) < 4:
        return None
    
    # Find two non-adjacent edges in the cycle
    m = len(cycle)
    # cycle is a list of vertices forming a cycle
    # Edges are (cycle[i], cycle[i+1]) for i in range(m), with wrap
    
    # Try to find two non-adjacent edges
    for i in range(m):
        u1, v1 = cycle[i], cycle[(i+1) % m]
        for j in range(i+2, m):
            if j == (i+1) % m or (j % m) == i:
                continue
            u2, v2 = cycle[j], cycle[(j+1) % m]
            
            # Check if swapping creates multi-edges or self-loops
            # Current edges: (u1,v1) and (u2,v2)
            # New edges: (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)
            
            for swap_type in range(2):
                if swap_type == 0:
                    new_e1, new_e2 = (u1, u2), (v1, v2)
                else:
                    new_e1, new_e2 = (u1, v2), (v1, u2)
                
                # Check validity
                valid = True
                for a, b in new_e1:
                    if a == b:
                        valid = False
                        break
                if not valid:
                    continue
                for a, b in new_e2:
                    if a == b:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check no multi-edges
                key1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                key2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
                
                # Check if these edges already exist (other than the ones we're removing)
                if key1 == (min(u1,v1), max(u1,v1)) or key1 == (min(u2,v2), max(u2,v2)):
                    continue
                if key2 == (min(u1,v1), max(u1,v1)) or key2 == (min(u2,v2), max(u2,v2)):
                    continue
                
                # Check if key1 or key2 already in adjacency
                if new_e1[1] in adj[new_e1[0]] and new_e1[0] != u1 and new_e1[0] != u2:
                    # This edge already exists elsewhere - would create multi-edge
                    pass
                # Simpler: just check if the new edge pairs are already present
                if new_e1[1] in adj[new_e1[0]]:
                    # Edge exists - but we need to make sure it's not one of the edges we're removing
                    if not (new_e1[0] == u1 and new_e1[1] == v1) and not (new_e1[0] == v1 and new_e1[1] == u1) and \
                       not (new_e1[0] == u2 and new_e1[1] == v2) and not (new_e1[0] == v2 and new_e1[1] == u2):
                        valid = False
                        break
                if new_e2[1] in adj[new_e2[0]]:
                    if not (new_e2[0] == u1 and new_e2[1] == v1) and not (new_e2[0] == v1 and new_e2[1] == u1) and \
                       not (new_e2[0] == u2 and new_e2[1] == v2) and not (new_e2[0] == v2 and new_e2[1] == u2):
                        valid = False
                        break
                
                if valid:
                    # Apply the switch
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    adj[new_e1[0]].append(new_e1[1])
                    adj[new_e1[1]].append(new_e1[0])
                    adj[new_e2[0]].append(new_e2[1])
                    adj[new_e2[1]].append(new_e2[0])
                    return adj
    
    return None

def _adj_to_edges(adj, n):
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _get_petersen_edges():
    edges = []
    for i in range(5):
        edges.append((i, (i+1) % 5))
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i+1) % 5]))
    for i in range(5):
        edges.append((i, i+5))
    return edges

def _random_lift(base_edges, M):
    n_base = 10
    n_lift = n_base * M
    perm_map = {}
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        if key not in perm_map:
            p = list(range(M))
            random.shuffle(p)
            perm_map[key] = p
    lift_edges = []
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        p = perm_map[key]
        for j in range(M):
            u_node = u * M + j
            v_node = v * M + p[j]
            lift_edges.append((u_node, v_node))
    return lift_edges

def _has_girth_at_least_fast(n, edges, g):
    """
    Fast girth verification using BFS from each vertex with early termination.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do a limited BFS to find shortest cycle through it
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] + 1 >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    if u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _generate_rule_variants(k, max_d):
    """
    Generate multiple rule variants for LUW construction.
    Returns a list of rule dictionaries.
    """
    if k != 3:
        # For k != 3, use the original rule
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, max_d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return [rule]
    
    # For k=3, generate multiple variants
    variants = []
    
    # Variant 0: Original
    rule0 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule0[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule0)
    
    # Variant 1: Different starting pattern
    rule1 = {2: (1, 1), 3: (1, 2), 4: (2, 1)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule1[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule1)
    
    # Variant 2: Shifted pattern
    rule2 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # Use different selection
        rule2[c] = [(1, c0 - 2), (c0, 1), (c0 - 1, 1), (1, c0 + 1)][r]
    variants.append(rule2)
    
    # Variant 3: Another shift
    rule3 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule3[c] = [(1, c0 - 2), (c0 - 1, 1), (1, c0 + 1), (c0, 1)][r]
    variants.append(rule3)
    
    return variants

def _construct_luw_with_rule(k, d, rule):
    """
    Construct LUW graph with a specific rule.
    """
    q = k
    if k == 3:
        q = 3
    
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

# Keep old functions for compatibility but they're no longer called from construct
def _construct_luw_optimized(k, d):
    rule = _generate_rule_variants(k, d)[0]
    return _construct_luw_with_rule(k, d, rule)

def _construct_luw_generic(k, d):
    return _construct_luw_optimized(k, d)
# EVOLVE-BLOCK-END
