import random
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using random regular graph
    generation (configuration model) with girth verification via non-backtracking walks.
    """
    # Target number of vertices - start with a reasonable estimate
    # For k=5, g=9, known best is 1116, lower bound 428
    # Try to find something in the 400-800 range
    target_n = 500
    
    max_attempts = 200
    
    for n in [target_n, target_n + 20, target_n + 40, target_n + 60, target_n + 80, target_n + 100]:
        for attempt in range(max_attempts):
            random.seed(attempt * 1000 + n)
            # Generate random k-regular graph using configuration model
            graph = _generate_random_regular(n, k)
            if graph is None:
                continue
            
            # Verify girth
            if _verify_girth(graph, g):
                # Convert adjacency to edge list
                edges = []
                for i in range(n):
                    for j in graph[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # Fallback: use a known construction or return a valid graph
    # As a last resort, use the Lazebnik-Ustimenko-Woldar approach
    return _fallback_luw(k, g)


def _generate_random_regular(n, k):
    """Generate a simple k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs: vertex i has k stubs labeled (i, 0), (i, 1), ..., (i, k-1)
    stubs = []
    for i in range(n):
        for j in range(k):
            stubs.append((i, j))
    
    # Randomly pair stubs
    for _ in range(50):
        # Shuffle stubs
        shuffled = stubs[:]
        random.shuffle(shuffled)
        
        # Pair them up
        edges_set = set()
        valid = True
        adjacency = [set() for _ in range(n)]
        
        for i in range(0, len(shuffled), 2):
            u, _ = shuffled[i]
            v, _ = shuffled[i + 1]
            if u == v:
                valid = False
                break
            if (u, v) in edges_set or (v, u) in edges_set:
                valid = False
                break
            edges_set.add((min(u, v), max(u, v)))
            adjacency[u].add(v)
            adjacency[v].add(u)
        
        if not valid:
            continue
        
        # Check all vertices have degree k
        if all(len(adj) == k for adj in adjacency):
            # Convert to list for easier access
            return [list(adj) for adj in adjacency]
    
    return None


def _verify_girth(graph, g):
    """
    Verify that the graph has girth at least g using non-backtracking walks.
    A graph has girth >= g iff there is no non-backtracking walk of length < g
    that returns to its starting vertex.
    """
    n = len(graph)
    
    # For each starting vertex, do BFS with non-backtracking constraint
    # State: (current_vertex, previous_vertex)
    # We track the shortest non-backtracking walk length to return to start
    
    # To be efficient, we do simultaneous search from all vertices
    # For girth >= g, we need to check non-backtracking walks of lengths 3 to g-1
    # (lengths 1 and 2 can't return to start in a simple graph with non-backtracking)
    
    max_depth = g - 1  # We check walks of length up to g-1
    
    # BFS state: (vertex, prev_vertex) -> set of starting vertices that can reach this state
    # Actually, let's do it per starting vertex but optimize
    
    # For each start vertex s, do non-backtracking BFS up to depth max_depth
    # and check if we return to s
    
    for s in range(n):
        # BFS from s with non-backtracking constraint
        # State: (current, prev) where prev is the vertex we came from (or -1 for start)
        # visited: set of (current, prev) states reached
        visited = set()
        queue = [(s, -1, 0)]  # (current, prev, depth)
        visited.add((s, -1))
        
        while queue:
            new_queue = []
            for current, prev, depth in queue:
                if depth >= max_depth:
                    continue
                
                for neighbor in graph[current]:
                    # Non-backtracking: can't go back to prev
                    if neighbor == prev:
                        continue
                    
                    new_state = (neighbor, current)
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    
                    # Check if we've returned to start
                    if neighbor == s and depth + 1 >= 3:
                        # Found a cycle of length depth + 1 < g
                        return False
                    
                    new_queue.append((neighbor, current, depth + 1))
            
            queue = new_queue
    
    return True


def _fallback_luw(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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
