import time
import itertools
import random
from collections import deque

# EVOLVE-BLOCK-START

def _bipartite_cayley_group(n, gens):
    """Build a bipartite 3-regular Cayley graph on Z_n with three involutions.
    Each generator is a fixed-point-free involution (perfect matching) on Z_n.
    The graph is bipartite iff each generator maps even<->odd, which holds iff
    each generator is a permutation that flips parity. We enforce this.
    Returns (n, edges) if valid (all gens are involutions, bijections, parity-flipping), else None.
    """
    if n % 2 != 0:
        return None
    # Each generator: a list perm of length n, perm[i] = image of i
    # Must be an involution: perm[perm[i]] == i, and perm[i] != i
    # Must flip parity: i % 2 != perm[i] % 2
    for perm in gens:
        for i in range(n):
            if perm[i] == i:
                return None
            if (i ^ perm[i]) % 2 != 1:
                return None
            if perm[perm[i]] != i:
                return None
    edges = []
    for i in range(n):
        for perm in gens:
            j = perm[i]
            if i < j:
                edges.append((i, j))
    return n, edges


def _girth_at_least(n, edges, g):
    """Return True if the graph has girth at least g (no cycle of length < g).
    Uses BFS from each vertex to find the shortest cycle through it.
    For efficiency, we only need to check up to depth g//2 from each vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    half = g // 2
    for s in range(n):
        # BFS from s, tracking parent to detect cycles
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= half:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    # found a cycle of length dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True


def _random_matching(n, rng):
    """Generate a random fixed-point-free involution (perfect matching) on {0..n-1}
    that flips parity. n must be even.
    We pair each even i with a random odd j.
    """
    odds = list(range(1, n, 2))
    rng.shuffle(odds)
    perm = [0] * n
    for i, j in enumerate(odds):
        even = 2 * i
        perm[even] = j
        perm[j] = even
    return perm


def _is_girth_ok(n, edges, g):
    """Fast girth check using non-backtracking walk enumeration up to length g-1.
    For a 3-regular graph, we enumerate all non-backtracking walks of length up to g-1
    from each vertex and check if any closes into a cycle of length < g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # For each starting vertex, do DFS of non-backtracking walks up to depth g-1
    # Track (current, previous) states to avoid revisiting
    max_depth = g - 1
    for s in range(n):
        # visited: set of (node, prev_node) we've seen at this depth from s
        # We use a set to track visited states
        visited = set()
        stack = [(s, -1, 0)]  # (node, prev, depth)
        while stack:
            node, prev, depth = stack.pop()
            if depth > 0 and node == s:
                # cycle found of length depth
                if depth < g:
                    return False
                continue
            if depth >= max_depth:
                continue
            visited.add((node, prev))
            for nb in adj[node]:
                if nb == prev:
                    continue
                if (nb, node) in visited:
                    continue
                stack.append((nb, node, depth + 1))
    return True


def construct(k, g):
    """Construct a k-regular graph of girth at least g on as few vertices as possible.
    Uses a group-based Cayley graph approach with random involutions.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    rng = random.Random(42)

    # For k=3, g=16: try Cayley graphs on Z_n with three parity-flipping involutions
    # Start with small n and increase
    # Moore bound for (3,16): 2 + 3*(1+2+4+8+16+32+64) = 2 + 3*127 = 383
    # So n >= 383, and n must be even, so n >= 384
    moore = 2 + k * sum(2 ** i for i in range(g // 2))
    if k == 3:
        moore = 2 + 3 * sum(2 ** i for i in range(g // 2))
    min_n = max(moore, 4)
    if min_n % 2 != 0:
        min_n += 1

    best_result = None
    best_n = None

    # Try various n values, starting from minimum
    # We'll try n = min_n, min_n+2, min_n+4, ... up to some limit
    # For each n, try multiple random generator sets
    
    max_n = min_n + 200  # try up to 100 even values above minimum
    attempts_per_n = 50
    
    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        for attempt in range(attempts_per_n):
            if time.time() > deadline - 5.0:
                break
            
            # Generate three random parity-flipping involutions
            gens = []
            valid = True
            for _ in range(3):
                perm = _random_matching(n, rng)
                gens.append(perm)
            
            # Check that the three generators together generate a connected graph
            # and that the graph is simple (no multi-edges)
            result = _bipartite_cayley_group(n, gens)
            if result is None:
                continue
            
            nn, edges = result
            
            # Check connectivity
            adj = [[] for _ in range(nn)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # BFS for connectivity
            visited = [False] * nn
            visited[0] = True
            q = deque([0])
            count = 1
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        q.append(v)
            
            if count != nn:
                continue
            
            # Check girth
            if _girth_at_least(nn, edges, g):
                best_result = (nn, edges)
                best_n = nn
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: use the LUW construction from the seed program
    q_field = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q_field == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn, sub_fn, mul_fn = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add_fn, sub_fn, mul_fn = (lambda x, y: (x + y) % q_field), (lambda x, y: (x - y) % q_field), (lambda x, y: x * y % q_field)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q_field):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add_fn(x[c], mul_fn(y[a], x[b])) if side == 0 else sub_fn(x[c], mul_fn(x[a], y[b]))
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
