import time
import random
from collections import deque

def _luw_construct(d, q):
    """
    Construct Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Returns (n, edges).
    """
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        # Assume q is prime
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Rule for LUW construction
    # The rule depends on d.
    # Standard LUW uses specific coefficients.
    # The seed code used a specific rule generation.
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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

def _check_girth(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    A graph has girth >= g iff there is no non-backtracking closed walk of length < g.
    For bipartite graphs (like LUW), we only need to check even lengths.
    But to be general, we check all lengths < g.
    
    Non-backtracking walk: a sequence v0, v1, ..., vk where vi != vi-2 for all i >= 2,
    and vk = v0 (closed).
    
    We use the transfer matrix approach or BFS on directed edges.
    State: (u, v) means we're at vertex v, coming from u.
    Transition: (u, v) -> (v, w) where w is a neighbor of v, w != u.
    
    We need to check if there's a cycle of length L < g in this state graph.
    A cycle in the state graph corresponds to a non-backtracking closed walk.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Build directed edge graph
    # State: (prev, curr)
    # From (prev, curr), can go to (curr, next) where next in adj[curr] and next != prev
    
    # For girth >= g, we need no non-backtracking closed walk of length 1..g-1.
    # Length 1: impossible in simple graph.
    # Length 2: u-v-u, but that's backtracking, so no.
    # Actually non-backtracking means v_{i} != v_{i-2}.
    # So a walk v0-v1-v2-...-vL=v0 with v_i != v_{i-2}.
    
    # Let's do BFS on the state space (prev, curr) for up to depth g-1.
    # If we can return to a state (a, b) that we've seen before at a smaller depth,
    # and the total walk length is < g, we have a cycle.
    
    # Actually simpler: do BFS from each directed edge (u, v), tracking depth.
    # If we reach state (v, w) where w is a neighbor of v, w != u, and we've been
    # to (v, w) before, that's a cycle in the state graph.
    # But we need the cycle length in the original graph.
    
    # Alternative: For each starting directed edge (s, t), do BFS in the non-backtracking
    # state space. If we can get back to (s, t) in L steps where L < g, then there's
    # a non-backtracking closed walk of length L+1 (since the state represents being
    # at t coming from s, and one step is the transition).
    
    # Let me think again. State (u, v) means we're at v, last came from u.
    # A transition (u,v) -> (v,w) takes 1 step in the walk.
    # A closed walk of length L in the original graph corresponds to L transitions
    # in the state graph that return to the starting state.
    
    # So: for each state (u, v), BFS in state graph. If we reach (u, v) again at
    # depth L where 1 <= L < g, then there's a non-backtracking closed walk of length L.
    
    # But we also need to handle the case where the walk doesn't return to the exact
    # same state but forms a cycle. Actually, in the state graph, a cycle of length L
    # means a non-backtracking closed walk of length L.
    
    # Optimization: We only need to check states reachable within depth g-1.
    
    # Build adjacency for state graph
    state_adj = {}
    for u in range(n):
        for v in adj[u]:
            # State (u, v)
            for w in adj[v]:
                if w != u:
                    state_adj.setdefault((u, v), []).append((v, w))
    
    # For each starting state, BFS up to depth g-1
    for start_state in state_adj:
        visited_depth = {start_state: 0}
        queue = deque([start_state])
        
        while queue:
            state = queue.popleft()
            depth = visited_depth[state]
            if depth >= g - 1:
                continue
            
            for next_state in state_adj.get(state, []):
                if next_state in visited_depth:
                    # Found a cycle in state graph
                    cycle_len = depth + 1
                    if cycle_len < g:
                        return False
                else:
                    visited_depth[next_state] = depth + 1
                    queue.append(next_state)
    
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = None
    best_edges = None
    
    # Try LUW constructions with different d
    # For g=10, try d=4 (n=512), d=5 (n=2048)
    # General rule: d = ceil((g-2)/2) - 1? 
    # LUW girth is 2d+2. So 2d+2 >= g => d >= (g-2)/2.
    
    candidates = []
    for d in range(3, 8):
        if 2 * d + 2 >= g:
            candidates.append(d)
            
    # Sort by expected size (smaller d = smaller n)
    candidates.sort()
    
    for d in candidates:
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_construct(d, k)
            # Verify girth
            if _check_girth(n, edges, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
        except Exception:
            pass
            
    # If no LUW graph worked or none found, fallback to seed logic (d=5)
    if best_n is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges = _luw_construct(d, k)
        best_n = n
        best_edges = edges
        
    return best_n, best_edges
