"""Cayley-graph construction for 4-regular girth-10 graph."""
import itertools
import math
import numpy as np
from collections import deque

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    # Moore bound for k=4, g=10 is 243
    # We search for n in [243, 400]
    for n in range(243, 401):
        # Get all factor pairs (a, b) with a*b = n, a <= b
        factors = []
        for a in range(1, int(math.isqrt(n)) + 1):
            if n % a == 0:
                b = n // a
                factors.append((a, b))
        
        for a, b in factors:
            # Group is Z_a x Z_b
            # We need to find generators u, v in this group
            # Represent elements as integers 0..n-1
            # Element i -> (i % a, i // a)
            
            # To avoid checking too many, we can iterate u, v
            # But we can prune:
            # 1. u must not be 0
            # 2. v must not be 0
            # 3. u != v, u != -v
            # 4. {u, v} must generate the group
            
            # Precompute inverses and addition tables?
            # n is small (<400), so we can just compute on fly or use small arrays.
            
            # Let's iterate u
            for u in range(1, n):
                # Check if u generates a subgroup of index 2 or 1? No, just iterate v
                for v in range(1, n):
                    # Basic checks
                    if v == u or v == n - u: # v == -u
                        continue
                    if u == n - v: # u == -v (same as above)
                        continue
                    if u == 0 or v == 0:
                        continue
                    
                    # Check if u, v generate the group
                    if not _generates_group(n, a, b, u, v):
                        continue
                    
                    # Check girth
                    if _check_girth_cayley(n, a, b, u, v, g):
                        # Construct edges
                        edges = []
                        edge_set = set()
                        for i in range(n):
                            # Neighbors: i+u, i-u, i+v, i-v
                            # In Z_a x Z_b, addition is component-wise mod a, b
                            # We need a function to add in this group
                            # But we can just use the integer representation if we define add_group(i, j)
                            
                            # Actually, it's easier to just build the graph using the group operation
                            # Let's define a helper to add two elements in Z_a x Z_b
                            # add(i, j)
                            # But we can also just use the fact that it's a Cayley graph.
                            # The edges are (i, i+u), (i, i+v) for all i.
                            # We need to map integer i to (x,y), add, map back.
                            
                            # To avoid overhead, we can precompute the neighbors for all i?
                            # Or just compute them here.
                            
                            # Let's compute neighbors for i
                            # i_u = add_group(i, u)
                            # i_v = add_group(i, v)
                            
                            i_u = _add_group(n, a, b, i, u)
                            i_v = _add_group(n, a, b, i, v)
                            
                            if i < i_u:
                                edge_set.add((i, i_u))
                            else:
                                edge_set.add((i_u, i))
                            
                            if i < i_v:
                                edge_set.add((i, i_v))
                            else:
                                edge_set.add((i_v, i))
                        
                        edges = sorted(edge_set)
                        return n, edges
                        
    return _construct_fallback(k, g)

def _add_group(n, a, b, i, j):
    """Add two elements in Z_a x Z_b represented as integers 0..n-1."""
    x1 = i % a
    y1 = i // a
    x2 = j % a
    y2 = j // a
    x_res = (x1 + x2) % a
    y_res = (y1 + y2) % b
    return x_res + y_res * a

def _neg_group(n, a, b, i):
    """Negate an element in Z_a x Z_b."""
    x = i % a
    y = i // a
    x_res = (-x) % a
    y_res = (-y) % b
    return x_res + y_res * a

def _generates_group(n, a, b, u, v):
    """Check if u and v generate Z_a x Z_b."""
    # Use BFS/DFS on the group starting from 0
    visited = [False] * n
    visited[0] = True
    queue = deque([0])
    while queue:
        curr = queue.popleft()
        for step in (u, v, _neg_group(n, a, b, u), _neg_group(n, a, b, v)):
            nxt = _add_group(n, a, b, curr, step)
            if not visited[nxt]:
                visited[nxt] = True
                queue.append(nxt)
    return all(visited)

def _check_girth_cayley(n, a, b, u, v, g):
    """
    Check if the Cayley graph of Z_a x Z_b with generators {u, -u, v, -v}
    has girth >= g.
    We do a BFS on the state space (vertex, prev_step) to find the shortest
    non-backtracking closed walk from 0.
    """
    # States: (vertex, last_step_index)
    # last_step_index: 0 for u, 1 for -u, 2 for v, 3 for -v
    # If last_step is None, we are at start.
    
    # BFS from (0, None)
    # We want to find a path back to 0 with length < g.
    # Actually, we want to find ANY non-backtracking closed walk of length < g.
    # Since it's a Cayley graph, the shortest non-backtracking closed walk at 0
    # is the girth.
    
    # Distances: dist[(vertex, last_step)]
    # We use a queue.
    # Initial state: (0, -1) where -1 means no previous step.
    # From (0, -1), we can go to (u, 0), (-u, 1), (v, 2), (-v, 3).
    
    # We stop if we reach (0, last_step) with distance < g?
    # No, we stop if we find ANY closed walk of length < g.
    # A closed walk of length L is a sequence of steps s1...sL such that sum=0.
    # Non-backtracking means s_{i+1} != -s_i.
    
    # Let's BFS.
    # dist[0, -1] = 0
    # Queue: [(0, -1, 0)]
    
    dist = {}
    queue = deque()
    queue.append((0, -1, 0))
    dist[(0, -1)] = 0
    
    steps = [u, _neg_group(n, a, b, u), v, _neg_group(n, a, b, v)]
    # Opposite steps:
    # 0 (u) -> 1 (-u)
    # 1 (-u) -> 0 (u)
    # 2 (v) -> 3 (-v)
    # 3 (-v) -> 2 (v)
    opposites = [1, 0, 3, 2]
    
    while queue:
        curr_v, prev_step, d = queue.popleft()
        
        # If we are at 0 and d > 0, we found a closed walk of length d.
        # If d < g, girth is violated.
        if curr_v == 0 and d > 0:
            if d < g:
                return False
            # If d >= g, we can stop?
            # No, there might be a shorter one? No, BFS finds shortest.
            # So if we pop (0, prev_step) with d, and d >= g, then all subsequent
            # paths to 0 will be >= d >= g.
            # Wait, we might reach 0 via different prev_steps.
            # But BFS ensures the first time we reach 0 (with any prev_step)
            # is the shortest non-backtracking closed walk.
            # So we can return True immediately if d >= g?
            # Yes, because any other closed walk would be longer or equal.
            # Actually, we just need to ensure NO closed walk of length < g.
            # So if the shortest one is >= g, we are good.
            return True
        
        for i, step in enumerate(steps):
            if i == opposites[prev_step] if prev_step != -1 else False:
                continue
            # If prev_step is -1, no restriction.
            # Actually, if prev_step is -1, we can take any step.
            
            next_v = _add_group(n, a, b, curr_v, step)
            next_state = (next_v, i)
            
            if next_state not in dist:
                dist[next_state] = d + 1
                queue.append((next_v, i, d + 1))
                
    # If we exhausted the queue without finding a closed walk of length < g,
    # and we haven't returned True yet (which implies we found one >= g),
    # it means there are no closed walks at all? Impossible for finite group.
    # Or we found one >= g and returned True.
    # If we didn't return True, it means we didn't reach 0?
    # But the group is finite and connected, so we must reach 0.
    return True

def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed."""
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
                other_idx = len(order)
                index[other] = other_idx
                order.append(other)
            else:
                other_idx = index[other]
            if vertex[0] == 0:
                edges.append((index[vertex], other_idx))
    return len(order), edges
