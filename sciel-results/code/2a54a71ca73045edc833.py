"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound_lower(k, g):
    """Return a safe lower bound on n(k,g)."""
    if g % 2 == 0:
        # Even girth: Moore bound n >= 1 + k * sum_{i=0}^{g/2-2} (k-1)^i
        # For 4-regular, g=10: 1 + 4*(1 + 3 + 9 + 27 + 81) = 1 + 4*121 = 485? 
        # Wait, that's the Moore bound for trees. For cages, the lower bound is different.
        # Actually the standard Moore bound for even girth g: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
        # For k=4, g=10: 2*(1+3+9+27+81) = 2*121 = 242. The task says best lower bound is 243.
        return 2 * sum((k-1)**i for i in range(g//2))
    else:
        # Odd girth: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        return 1 + k * sum((k-1)**i for i in range((g-1)//2))


def _verify_girth(n, edges, k, g):
    """Verify the graph is k-regular, simple, and has girth >= g using non-backtracking walks."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, j in edges:
        if i == j:
            return False
        if i > j:
            i, j = j, i
        if (i, j) in edge_set:
            return False
        edge_set.add((i, j))
        adj[i].append(j)
        adj[j].append(i)
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check girth >= g using BFS for non-backtracking walks
    # For each vertex, do a BFS tracking (current, previous) to avoid backtracking
    # The shortest non-backtracking cycle through vertex v has length = shortest path from v to v
    # in the non-backtracking sense.
    
    # Actually, a cleaner approach: for each vertex v, do a BFS in the "non-backtracking" state space.
    # State is (node, prev_node). Start from (v, -1) for all neighbors... 
    # Simpler: for each pair of adjacent vertices (u,v), find shortest non-backtracking path from v back to u.
    # The cycle length is that path length + 1.
    
    # For efficiency with small graphs, let's do: for each vertex v, BFS where state is (node, parent).
    # Initialize: for each neighbor w of v, state (w, v) at depth 1.
    # If we reach state (v, w) for some neighbor w, we have a cycle of length depth+1.
    
    for v in range(n):
        # BFS in non-backtracking state space
        visited = {}  # (node, parent) -> depth
        queue = deque()
        for w in adj[v]:
            visited[(w, v)] = 1
            queue.append((w, v, 1))
        
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= g:
                break
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor == v:
                    # Found a cycle of length depth + 1
                    if depth + 1 < g:
                        return False
                    continue
                state = (neighbor, node)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((neighbor, node, depth + 1))
    
    return True


def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction - the baseline."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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


def _construct_random_regular(n, k, seed):
    """Attempt to construct a random k-regular graph on n vertices using configuration model."""
    rng = random.Random(seed)
    
    # Configuration model: create n*k stubs
    stubs = list(range(n * k))
    # Assign stubs to vertices: vertex i gets stubs [i*k, (i+1)*k)
    # Shuffle and pair up
    rng.shuffle(stubs)
    
    # Create edges by pairing consecutive stubs
    adj = [[] for _ in range(n)]
    edge_set = set()
    valid = True
    
    for i in range(0, n * k, 2):
        u = stubs[i] // k
        v = stubs[i + 1] // k
        if u == v:
            valid = False
            break
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            valid = False
            break
        edge_set.add((a, b))
        adj[u].append(v)
        adj[v].append(u)
    
    if not valid:
        return None
    
    # Check regularity (should be automatic from construction)
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    edges = list(edge_set)
    return n, edges


def _try_improve_with_search(k, g, deadline, start_n=None):
    """Try to find a smaller valid graph by searching n values."""
    best_n = start_n
    best_edges = None
    
    if start_n is not None:
        # Get the baseline graph
        best_n, best_edges = _construct_luw(k, g)
        if not _verify_girth(best_n, best_edges, k, g):
            return best_n, best_edges
    
    lower_bound = _moore_bound_lower(k, g)
    
    # Try random regular graphs with n < best_n
    # Work from lower_bound upward
    for n in range(lower_bound, best_n):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(20):
            if time.time() > deadline - 15:
                break
            result = _construct_random_regular(n, k, seed)
            if result is None:
                continue
            n_try, edges_try = result
            if _verify_girth(n_try, edges_try, k, g):
                best_n = n_try
                best_edges = edges_try
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
    
    return best_n, best_edges


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with the LUW construction
    n, edges = _construct_luw(k, g)
    
    # Verify it's valid
    if not _verify_girth(n, edges, k, g):
        # If LUW fails for some parameters, fall back to random search
        pass
    
    # Try to improve with random regular graph search
    n, edges = _try_improve_with_search(k, g, deadline, start_n=n)
    
    return n, edges

# EVOLVE-BLOCK-END
