# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 5.0

def _validate_graph(n, edges, k, g, max_vertices=None, t_start=None, deadline=None):
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
        if deadline is not None and time.time() > deadline:
            return False
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

def _luw_graph(k, g, rule_override=None):
    """
    LUW construction with optional rule override.
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
    
    # Apply rule override if provided
    if rule_override:
        for c, val in rule_override.items():
            rule[c] = val

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

def _greedy_improve(n, edges, k, g, deadline):
    """
    Try to improve the graph by removing edges and reconnecting to reduce girth violations.
    Returns improved (n, edges) or original if no improvement found.
    """
    if time.time() > deadline:
        return n, edges
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check if valid first
    if not _validate_graph(n, edges, k, g, deadline=deadline):
        return n, edges
    
    # Try random edge swaps to see if we can find a smaller valid graph
    # This is heuristic - try to find alternative constructions
    best_n = n
    best_edges = edges
    
    # Try different seed-based perturbations
    rng = random.Random(42)
    attempts = 0
    max_attempts = 50
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        # Create a perturbed rule set
        d = g - 5
        perturbed = {}
        for c in range(3, d + 1):
            if rng.random() < 0.3:
                # Pick a random valid rule
                options = [(1, c - 2), (c - 1, 1), (c, 1), (1, c + 1)]
                perturbed[c] = rng.choice(options)
        
        try:
            n_try, edges_try = _luw_graph(k, g, rule_override=perturbed)
            if n_try < best_n:
                if _validate_graph(n_try, edges_try, k, g, deadline=deadline):
                    best_n = n_try
                    best_edges = edges_try
        except Exception:
            continue
    
    return best_n, best_edges

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # Baseline: LUW
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    if k != 4 or g != 10:
        return best_n, best_edges
    
    # For k=4, g=10, try variations of the rule
    # Systematically explore rule combinations for c=3,4,5,6
    # d = g - 5 = 5, so rules needed for c=2,3,4,5
    
    # All plausible rule options for each c
    rule_options = {
        3: [(2, 1), (1, 1), (3, 1), (1, 2)],
        4: [(1, 2), (2, 1), (3, 1), (1, 3), (4, 1)],
        5: [(1, 3), (4, 1), (5, 1), (1, 6), (2, 1), (3, 1)],
    }
    
    # Try all combinations systematically
    c3_opts = rule_options[3]
    c4_opts = rule_options[4]
    c5_opts = rule_options[5]
    
    best_found = False
    for c3 in c3_opts:
        if time.time() > deadline:
            break
        for c4 in c4_opts:
            if time.time() > deadline:
                break
            for c5 in c5_opts:
                if time.time() > deadline:
                    break
                var = {3: c3, 4: c4, 5: c5}
                try:
                    n_var, edges_var = _luw_graph(k, g, rule_override=var)
                    
                    if n_var < best_n:
                        # Validate
                        if _validate_graph(n_var, edges_var, k, g, deadline=deadline):
                            best_n = n_var
                            best_edges = edges_var
                            best_found = True
                except Exception:
                    continue
    
    # If we found something better, also try greedy improvement
    if best_found and time.time() < deadline:
        n_imp, edges_imp = _greedy_improve(best_n, best_edges, k, g, deadline)
        if n_imp < best_n:
            best_n = n_imp
            best_edges = edges_imp
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
