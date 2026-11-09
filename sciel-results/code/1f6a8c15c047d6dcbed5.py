"""Stochastic local search for 4-regular graphs with girth at least g.
Starts with random 4-regular graphs and uses 2-switches to eliminate
short cycles, minimizing vertex count."""
import time
import sys
import random
import math
from collections import deque

_DEADLINE = 115.0  # safety margin below the 120s evaluator kill


def construct(k, g):
    t_start = time.time()

    if k != 4:
        return _luw(k, g)

    # Moore bound for (4,10): 2 + 2*(3+9+27+81) = 2 + 2*120 = 246
    moore_bound = 2 + 2 * sum((k - 1) ** i for i in range(1, g // 2))
    
    best_n = None
    best_edges = None

    # Try increasing n from Moore bound
    for n in range(moore_bound, 500, 2):  # n must be even for 4-regular
        if time.time() - t_start > _DEADLINE:
            break
        
        # Try multiple random initializations with simulated annealing
        for attempt in range(5):
            if time.time() - t_start > _DEADLINE:
                break
            
            adj = _generate_random_k_regular(n, k)
            if adj is None:
                continue
            
            # Run simulated annealing local search to eliminate short cycles
            adj, penalty = _simulated_annealing(adj, n, k, g, t_start, _DEADLINE)
            
            # Check if girth >= g
            if penalty == 0:
                edges = _adjacency_to_edges(adj)
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break  # Found a valid graph for this n, move to next n

    if best_n is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    return _luw(k, g)


def _generate_random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    
    random.shuffle(stubs)
    
    # Pair up stubs
    adj = [[0] * n for _ in range(n)]
    valid = True
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            valid = False
            break
        if adj[u][v] >= 1:  # no multi-edges
            valid = False
            break
        adj[u][v] = 1
        adj[v][u] = 1
    
    if not valid:
        return None
    
    return adj


def _find_short_cycles(adj, n, g):
    """Find all cycles of length < g. Returns list of cycles (as vertex lists)."""
    cycles = []
    for start in range(n):
        # BFS/DFS to find cycles through start
        # Use non-backtracking walks
        visited = set()
        stack = [(start, [start], -1)]  # (current, path, prev)
        
        while stack:
            node, path, prev = stack.pop()
            for nb in range(n):
                if adj[node][nb] and nb != prev:
                    if nb == start and len(path) >= 3:
                        if len(path) < g:
                            cycles.append(list(path))
                    elif nb not in path:
                        stack.append((nb, path + [nb], node))
    
    return cycles


def _check_girth(adj, n, g):
    """Check if graph has girth >= g using non-backtracking walks from every vertex."""
    for start in range(n):
        # BFS with non-backtracking constraint
        # State: (node, prev)
        visited = set()
        visited.add((start, -1))
        queue = deque([(start, -1, 0)])  # (node, prev, depth)
        
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for nb in range(n):
                if adj[node][nb] and nb != prev:
                    if nb == start and depth + 1 < g:
                        return False  # Found cycle shorter than g
                    if (nb, node) not in visited:
                        visited.add((nb, node))
                        queue.append((nb, node, depth + 1))
    
    return True


def _simulated_annealing(adj, n, k, g, t_start, deadline):
    """Simulated annealing with 2-switches to minimize short cycle penalty."""
    adj = [row[:] for row in adj]
    
    # Precompute adjacency lists for faster neighbor iteration
    adj_list = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_list[i].add(j)
    
    def compute_penalty():
        """Compute total penalty: sum of 1/(len-2) for each short cycle found via BFS."""
        total = 0.0
        for start in range(n):
            visited = set()
            visited.add((start, -1))
            queue = deque([(start, -1, 0)])
            
            while queue:
                node, prev, depth = queue.popleft()
                if depth >= g - 1:
                    continue
                for nb in adj_list[node]:
                    if nb == prev:
                        continue
                    if nb == start and depth + 1 < g:
                        total += 1.0 / (depth + 1 - 2)
                        return total  # Early exit: found at least one short cycle
                    if (nb, node) not in visited:
                        visited.add((nb, node))
                        queue.append((nb, node, depth + 1))
        return total
    
    current_penalty = compute_penalty()
    
    if current_penalty == 0:
        return adj, 0
    
    T_start = 1.0
    T_end = 0.001
    cooling = 0.999
    
    best_adj = [row[:] for row in adj]
    best_penalty = current_penalty
    
    T = T_start
    no_improve = 0
    
    while time.time() - t_start < deadline - 5:
        if T < T_end:
            break
        
        # Pick a random edge
        u = random.randint(0, n - 1)
        neighbors = list(adj_list[u])
        if not neighbors:
            break
        v = random.choice(neighbors)
        
        # Find another edge (w, x) not sharing vertices with (u, v)
        # and not already connected
        found_switch = False
        for _ in range(20):
            w = random.randint(0, n - 1)
            if w in (u, v):
                continue
            w_neighbors = list(adj_list[w])
            if not w_neighbors:
                continue
            x = random.choice(w_neighbors)
            if x in (u, v) or x == w:
                continue
            if adj[u][x] or adj[v][w]:
                continue
            
            # Try the 2-switch: remove (u,v) and (w,x), add (u,x) and (v,w)
            # Check no new edges already exist
            if adj[u][x] or adj[v][w]:
                continue
            
            # Apply switch
            adj[u][v] = 0
            adj[v][u] = 0
            adj[w][x] = 0
            adj[x][w] = 0
            adj[u][x] = 1
            adj[x][u] = 1
            adj[v][w] = 1
            adj[w][v] = 1
            
            adj_list[u].remove(v)
            adj_list[v].remove(u)
            adj_list[w].remove(x)
            adj_list[x].remove(w)
            adj_list[u].add(x)
            adj_list[x].add(u)
            adj_list[v].add(w)
            adj_list[w].add(v)
            
            new_penalty = compute_penalty()
            
            # Accept if better or with probability exp(-(new-old)/T)
            delta = new_penalty - current_penalty
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_penalty = new_penalty
                no_improve = 0
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_adj = [row[:] for row in adj]
                    if best_penalty == 0:
                        return best_adj, 0
            else:
                # Revert
                adj[u][x] = 0
                adj[x][u] = 0
                adj[v][w] = 0
                adj[w][v] = 0
                adj[u][v] = 1
                adj[v][u] = 1
                adj[w][x] = 1
                adj[x][w] = 1
                
                adj_list[u].remove(x)
                adj_list[x].remove(u)
                adj_list[v].remove(w)
                adj_list[w].remove(v)
                adj_list[u].add(v)
                adj_list[v].add(u)
                adj_list[w].add(x)
                adj_list[x].add(w)
                no_improve += 1
            
            found_switch = True
            break
        
        if not found_switch:
            no_improve += 10
        
        T *= cooling
        
        if no_improve > 500:
            break
    
    return best_adj, best_penalty


def _find_one_short_cycle(adj, n, g):
    """Find one cycle of length < g using BFS from each vertex."""
    for start in range(n):
        # BFS with non-backtracking
        visited = set()
        visited.add((start, -1))
        queue = deque()
        for nb in range(n):
            if adj[start][nb]:
                queue.append((nb, start, 1, [start, nb]))
        
        while queue:
            node, prev, depth, path = queue.popleft()
            if depth >= g - 1:
                continue
            for nb in range(n):
                if adj[node][nb] and nb != prev:
                    if nb == start and depth + 1 < g:
                        return path
                    if (nb, node) not in visited:
                        visited.add((nb, node))
                        queue.append((nb, node, depth + 1, path + [nb]))
    
    return None


def _adjacency_to_edges(adj):
    """Convert adjacency matrix to edge list."""
    edges = []
    n = len(adj)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append((i, j))
    return edges


def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
    q = k
    # girth = d + 5 for odd d, so d = g - 5 for odd g, g - 4 for even g
    # Actually the paper says girth is at least d + 5.
    # For g=10, we want d+5 >= 10, so d >= 5. Use d=5.
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Build the rule
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Use BFS to find the connected component
    from collections import deque as dq
    start = (0, tuple([None] + [0] * d))

    def neighbours(vertex):
        side, x = vertex
        x = list(x)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    index = {start: 0}
    order = [start]
    edges = []

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
