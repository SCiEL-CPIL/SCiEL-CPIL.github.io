"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 5.0

def _validate_graph(n, edges, k, g, max_vertices=None):
    """
    Validate simple k-regular graph with girth >= g.
    Returns True if valid, False otherwise.
    """
    if max_vertices is not None and n > max_vertices:
        return False
        
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check girth using non-backtracking BFS from each vertex
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], 
                 [0, 1, 2, 3], 
                 [0, 2, 3, 1], 
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

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

def _count_short_cycles_fast(adj, n, g, time_limit=2.0):
    """
    Fast girth check: returns True if girth >= g, False otherwise.
    Uses early termination and time limit.
    """
    t_start = time.time()
    max_depth = g - 2
    
    for s in range(n):
        if time.time() - t_start > time_limit:
            return True
            
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            if depth >= max_depth:
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _girth_penalty(adj, n, g, max_depth=8):
    """
    Compute a penalty score based on short cycles. Lower is better.
    Returns a float score where 0 means no short cycles found.
    """
    penalty = 0.0
    weight = 1.0
    for depth in range(3, min(g, max_depth + 1)):
        w = 1.0 / (depth - 2)
        for s in range(n):
            dq = deque()
            dist = {}
            for nb in adj[s]:
                dq.append((nb, s))
                dist[(nb, s)] = 1
            while dq:
                curr, prev = dq.popleft()
                d = dist[(curr, prev)]
                if curr == s:
                    if d + 1 < g:
                        penalty += w
                        continue
                if d >= max_depth:
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    state = (nb, curr)
                    if state not in dist:
                        dist[state] = d + 1
                        dq.append(state)
        if penalty > 1000:
            break
    return penalty

def _generate_girth_aware_4_regular(n, g, seed, max_iter=5000):
    """
    Generate a 4-regular graph biased toward high girth using
    a sequential edge-adding approach with local girth checks.
    """
    rng = random.Random(seed)
    
    # Start with a random 4-regular graph
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
    rng.shuffle(stubs)
    
    edges = set()
    valid = True
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            valid = False
            break
        edge = (min(u, v), max(u, v))
        if edge in edges:
            valid = False
            break
        edges.add(edge)
    
    if not valid:
        return None
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Now do local improvement: try to swap edges to increase girth
    best_edges = list(edges)
    best_penalty = _girth_penalty(adj, n, g, max_depth=6)
    
    if best_penalty == 0:
        return best_edges
    
    t_start = time.time()
    for it in range(max_iter):
        if time.time() - t_start > 3.0:
            break
            
        # Pick a random edge
        e1 = rng.choice(list(edges))
        u1, v1 = e1
        
        # Pick another edge not sharing vertices with e1
        candidates = [e for e in edges if len({u1, v1} & set(e)) == 0]
        if not candidates:
            continue
        e2 = rng.choice(candidates)
        u2, v2 = e2
        
        # Try swap: remove e1, e2; add (u1,u2), (v1,v2) or (u1,v2), (v1,u2)
        for new_e1, new_e2 in [((min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2))),
                                ((min(u1, v2), max(u1, v2)), (min(v1, u2), max(v1, u2)))]:
            if new_e1 == new_e2:
                continue
            if new_e1 in edges or new_e2 in edges:
                continue
                
            # Apply swap
            edges.discard(e1)
            edges.discard(e2)
            edges.add(new_e1)
            edges.add(new_e2)
            
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            adj[u2].discard(v2)
            adj[v2].discard(u2)
            
            a1, b1 = new_e1
            a2, b2 = new_e2
            adj[a1].add(b1)
            adj[b1].add(a1)
            adj[a2].add(b2)
            adj[b2].add(a2)
            
            # Check if girth improved
            if _count_short_cycles_fast(adj, n, g, time_limit=0.3):
                new_penalty = _girth_penalty(adj, n, g, max_depth=5)
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_edges = list(edges)
                    if best_penalty == 0:
                        return best_edges
                else:
                    # Revert
                    edges.discard(new_e1)
                    edges.discard(new_e2)
                    edges.add(e1)
                    edges.add(e2)
                    adj[a1].discard(b1)
                    adj[b1].discard(a1)
                    adj[a2].discard(b2)
                    adj[b2].discard(a2)
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
            else:
                # Revert
                edges.discard(new_e1)
                edges.discard(new_e2)
                edges.add(e1)
                edges.add(e2)
                adj[a1].discard(b1)
                adj[b1].discard(a1)
                adj[a2].discard(b2)
                adj[b2].discard(a2)
                adj[u1].add(v1)
                adj[v1].add(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
            break
    
    return best_edges

def _count_short_cycles(adj, n, g):
    """
    Count cycles of length < g using non-backtracking walks.
    """
    count = 0
    max_depth = g - 2
    
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    count += 1
                continue
                
            if depth >= max_depth:
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return count

def _improve_girth(n, edges, g, time_limit, rng):
    """
    Use simulated annealing with 2-switch operations to increase girth.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        
    current_cost = _count_short_cycles(adj, n, g)
    if current_cost == 0:
        return list(edges), 0
        
    best_edges = list(edges)
    best_cost = current_cost
    
    t_start = time.time()
    temp = 50.0
    cooling = 0.995
    min_temp = 0.01
    
    no_improve_count = 0
    
    while time.time() < t_start + time_limit and temp > min_temp:
        # Pick two random distinct edges
        e1 = rng.choice(edges)
        e2 = rng.choice(edges)
        if e1 == e2:
            continue
            
        u1, v1 = e1
        u2, v2 = e2
        
        # Edges must be non-adjacent
        if len({u1, v1, u2, v2}) < 4:
            continue
            
        # Try both possible 2-switches
        for new_e1, new_e2 in [((min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2))),
                                ((min(u1, v2), max(u1, v2)), (min(v1, u2), max(v1, u2)))]:
            if new_e1 == new_e2:
                continue
            if new_e1 in edge_set or new_e2 in edge_set:
                continue
                
            # Perform the switch
            edge_set.discard(e1)
            edge_set.discard(e2)
            edge_set.add(new_e1)
            edge_set.add(new_e2)
            
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            adj[u2].discard(v2)
            adj[v2].discard(u2)
            
            a1, b1 = new_e1
            a2, b2 = new_e2
            adj[a1].add(b1)
            adj[b1].add(a1)
            adj[a2].add(b2)
            adj[b2].add(a2)
            
            # Update edges list
            edges.remove(e1)
            edges.remove(e2)
            edges.append(new_e1)
            edges.append(new_e2)
            
            new_cost = _count_short_cycles(adj, n, g)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_edges = list(edges)
                no_improve_count = 0
                temp = max(temp * 1.3, 30.0)
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    temp *= 0.97
                    no_improve_count = 0
                
            if new_cost <= current_cost or rng.random() < pow(2.718, -(new_cost - current_cost) / temp):
                current_cost = new_cost
            else:
                # Revert
                edge_set.discard(new_e1)
                edge_set.discard(new_e2)
                edge_set.add(e1)
                edge_set.add(e2)
                
                adj[a1].discard(b1)
                adj[b1].discard(a1)
                adj[a2].discard(b2)
                adj[b2].discard(a2)
                
                adj[u1].add(v1)
                adj[v1].add(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                
                edges.remove(new_e1)
                edges.remove(new_e2)
                edges.append(e1)
                edges.append(e2)
            break
            
        temp *= cooling
        
    return best_edges, best_cost

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # 1. Try LUW Graph
    n_luw, edges_luw = _luw_graph(k, g)
    
    # Check if LUW graph is valid
    if _validate_graph(n_luw, edges_luw, k, g):
        return n_luw, edges_luw
        
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. If k=4, g=10, try to find a valid graph on 320 vertices
    if k == 4 and g == 10:
        target_n = 320
        rng = random.Random(42)
        
        # Use a more effective search: Greedy with backtracking
        # This is a simplified version, trying to build the graph edge by edge
        # ensuring no short cycles are formed.
        
        # For now, we rely on the improved random search from the original
        # but with a stricter time limit and more restarts if possible.
        
        # The original random search is very slow. 
        # We will attempt a few restarts of the simulated annealing.
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            time_left = deadline - time.time()
            if time_left < 5:
                break
                
            # Generate random 4-regular
            stubs = []
            for v in range(target_n):
                for _ in range(4):
                    stubs.append(v)
            rng.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
            
            if not valid:
                continue
                
            edges_list = list(edges)
            
            # Check if already valid
            if _validate_graph(target_n, edges_list, k, g):
                return target_n, edges_list
                
            # Improve with SA
            sa_time = min(time_left - 2, 10.0)
            if sa_time < 1:
                break
                
            improved_edges, cost = _improve_girth(target_n, edges_list, g, sa_time, rng)
            
            if cost == 0:
                if _validate_graph(target_n, improved_edges, k, g):
                    return target_n, improved_edges

    return best_n, best_edges

# EVOLVE-BLOCK-END
