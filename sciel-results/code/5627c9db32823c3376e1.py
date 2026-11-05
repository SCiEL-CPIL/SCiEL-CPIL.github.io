"""Rewritten program: construct k-regular graph of girth g using random Cayley graphs on symmetric groups."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Generate random Cayley graphs on symmetric groups S_n using k involutory generators.
    Verify girth via BFS from identity, checking for collisions in the ball of radius (g-1)//2.
    """
    
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    def random_involution(n):
        """Generate a random involution in S_n."""
        perm = list(range(n))
        # Randomly pair up elements
        available = list(range(n))
        random.shuffle(available)
        i = 0
        while i < n:
            if i < n - 1:
                # Decide whether to swap available[i] and available[i+1]
                if random.random() < 0.5:
                    a, b = available[i], available[i+1]
                    perm[a] = b
                    perm[b] = a
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        return tuple(perm)
    
    def compose(p, q):
        """Compose permutations p and q: (p o q)(i) = p(q(i))."""
        n = len(p)
        return tuple(p[q[i]] for i in range(n))
    
    def check_girth_cayley(generators, n, g):
        """
        Check if the Cayley graph on S_n with given generators has girth >= g.
        Returns (True, component_vertices) if girth >= g, else (False, None).
        """
        radius = (g - 1) // 2
        identity = tuple(range(n))
        
        # BFS from identity, tracking parent to detect non-backtracking collisions
        # For each vertex, track the set of "incoming edges" (previous vertex, generator)
        # If a vertex is reached by two different non-backtracking paths of length <= radius,
        # there's a cycle of length < g.
        
        visited = {}  # vertex -> set of (prev_vertex, gen_index) that led here
        visited[identity] = set()
        queue = deque([(identity, None, None, 0)])  # (vertex, prev, gen_idx, depth)
        
        while queue:
            vertex, prev, gen_idx, depth = queue.popleft()
            if depth >= radius:
                continue
            
            for gi, gen in enumerate(generators):
                # Skip backtracking
                if gi == gen_idx and prev is not None:
                    # Actually, since generators are involutions, backtracking means
                    # applying the same generator again. So if we came from prev via gen gi,
                    # we can't go back to prev via gen gi.
                    # But we should allow going to prev via a different generator.
                    pass
                
                new_vertex = compose(gen, vertex)
                
                # Check if this is backtracking
                if prev is not None and gi == gen_idx:
                    # This would go back to prev
                    if new_vertex == prev:
                        continue
                
                if new_vertex in visited:
                    # Check if this is a new non-backtracking path
                    if (vertex, gi) not in visited[new_vertex]:
                        # Collision detected: two different non-backtracking paths
                        # to the same vertex within radius
                        return (False, None)
                    else:
                        continue
                else:
                    visited[new_vertex] = set()
                
                visited[new_vertex].add((vertex, gi))
                queue.append((new_vertex, vertex, gi, depth + 1))
        
        # Count vertices in the connected component
        return (True, len(visited))
    
    # Try different group sizes and random generator sets
    best_result = None
    
    # Try S_n for n from small to large
    for n in range(6, 20):
        # Try multiple random generator sets
        for attempt in range(50):
            generators = []
            while len(generators) < k:
                gen = random_involution(n)
                if gen != tuple(range(n)):  # Not identity
                    if gen not in generators:  # Distinct generators
                        generators.append(gen)
            
            ok, count = check_girth_cayley(generators, n, g)
            if ok:
                if best_result is None or count < best_result[0]:
                    best_result = (count, generators, n)
                    if count <= mb:
                        # Can't do better than Moore bound
                        break
        if best_result is not None and best_result[0] <= mb:
            break
    
    if best_result is None:
        # Fallback: use a simple construction
        # For now, return a valid but possibly larger graph
        # Use the L-U-W construction as fallback
        def luw_construct(k, g):
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
        
        return luw_construct(k, g)
    
    count, generators, n = best_result
    
    # Build the full graph
    identity = tuple(range(n))
    visited = {}
    visited[identity] = 0
    order = [identity]
    queue = deque([identity])
    
    while queue:
        vertex = queue.popleft()
        for gen in generators:
            new_vertex = compose(gen, vertex)
            if new_vertex not in visited:
                visited[new_vertex] = len(order)
                order.append(new_vertex)
                queue.append(new_vertex)
    
    # Build edges
    edges = []
    for i, vertex in enumerate(order):
        for gen in generators:
            new_vertex = compose(gen, vertex)
            j = visited[new_vertex]
            if i < j:
                edges.append((i, j))
    
    return len(order), edges

# EVOLVE-BLOCK-END
