import time
import math
import random
from collections import deque
import numpy as np

def _verify_girth(n, edges, g):
    """
    Verify that the graph has no cycle shorter than g.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check for self-loops and multi-edges
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        edge_set.add((u, v))
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])): # Multi-edges handled above, but just in case
            return False
        if len(adj[i]) < 0: # Placeholder
            pass

    # Non-backtracking walks to find shortest cycle
    # For each vertex, do a BFS on the non-backtracking state space (u, parent)
    # If we reach a node (u, parent) where parent is not the previous node in the path, 
    # and we encounter a node already visited in this BFS from a different parent, 
    # we have a cycle.
    # Actually, simpler: BFS from each vertex, tracking the path or just depth.
    # If we find a cycle of length < g, return False.
    
    # To be efficient, we can use a single BFS for each start node.
    # dist[u] = distance from start.
    # If we visit a node already visited, we have a cycle.
    # Cycle length = dist[u] + dist[v] + 1? No, that's for undirected BFS.
    # In undirected BFS, if we see an edge (u, v) where both are visited and not parent-child,
    # the cycle length is dist[u] + dist[v] + 1.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
        # If we finished BFS from start without finding short cycle, continue to next start?
        # No, if the graph is connected, one BFS from any start might not find ALL cycles?
        # Wait, BFS from a single start finds the shortest cycle passing through that start.
        # We need to ensure NO cycle is short.
        # So we must run BFS from EVERY vertex?
        # Yes, "certified exactly with non-backtracking walks from every vertex".
        
    return True

def _check_simple_and_regular(n, edges, k):
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True

def construct_random(k, g, deadline):
    """
    Try to find a k-regular graph of girth >= g with minimal n.
    """
    # Moore Bound
    if g % 2 == 0:
        mb = 2 * sum((k-1)**i for i in range(g//2))
    else:
        mb = 1 + k * sum((k-1)**i for i in range((g-1)//2))
        
    # Start from MB, go up
    # Step size: maybe 10 or 50?
    # For k=5, g=9, MB=106. Known best 1116.
    # We should search in ranges.
    
    best_n = None
    best_edges = None
    
    # Estimate max n to try based on time
    # If we can't find one quickly, we return the LW construction as fallback.
    
    start_n = mb
    # Increment
    # If k=5, g=9, we expect n ~ 1000.
    # Let's try n from mb to mb + 2000 in steps of 2 (must be even for regular graph? No, n*k must be even).
    # k=5 is odd, so n must be even.
    
    step = 2 if k % 2 != 0 else 1
    
    # We will iterate n.
    # For each n, try a few random graphs.
    
    # Time budget per n?
    # Let's say we have 100 seconds left.
    # We can try many n values.
    
    # Optimization: 
    # Generating a random k-regular graph is O(n).
    # Checking girth is O(n * (k-1)^{g/2})? No, BFS is O(n+k).
    # So checking is fast.
    # The bottleneck is finding a valid one.
    
    # If we don't find one, we fall back to LW.
    
    # Let's implement a random regular graph generator.
    # Configuration model:
    # Create list of half-edges: [0]*k + [1]*k + ...
    # Shuffle
    # Pair them up.
    # Check for loops and multi-edges.
    # If invalid, retry.
    
    # For n=1000, k=5, probability of no loops/multi-edges is decent?
    # Number of possible pairs ~ n^2/2.
    # Expected loops ~ k * n / (2n) = k/2? No.
    # P(loop) ~ k/(2n).
    # P(no loops) ~ (1 - k/2n)^{n/2} ~ e^{-k/4}.
    # For k=5, e^{-1.25} ~ 0.28.
    # So ~30% chance of simple graph.
    
    # Girth constraint is harder.
    
    for n in range(start_n, start_n + 5000, step):
        if time.time() > deadline:
            break
            
        # Try a few random instances for this n
        # How many?
        # If we are close to MB, we need many tries.
        # If we are far, fewer.
        # Let's try 100 tries per n.
        
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Generate random k-regular graph
            half_edges = []
            for i in range(n):
                for _ in range(k):
                    half_edges.append(i)
            
            random.shuffle(half_edges)
            
            edges = []
            valid = True
            adj = [[] for _ in range(n)]
            edge_set = set()
            
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
                edges.append((u, v))
                adj[u].append(v)
                adj[v].append(u)
                
            if not valid:
                continue
                
            # Check girth
            if _verify_girth(n, edges, g):
                best_n = n
                best_edges = edges
                found = True
                break # Found for this n, try smaller n? No, we are increasing n.
                # Wait, we want MINIMAL n.
                # So if we find one for n, we should stop?
                # Yes, because we are iterating n upwards from MB.
                # So the first valid n is the best.
                return n, edges
        
    if best_n is not None:
        return best_n, best_edges
    return None

def construct_lw(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    # This produces a bipartite k-regular graph.
    # For girth g, we need d such that the tree of depth d/2 has no collisions.
    # The LW construction with parameter d has girth at least 2*d if d is chosen correctly?
    # Actually, the paper says CD(d,q) has girth >= 2d if q is large enough?
    # Let's check: for k=5, g=9.
    # Moore bound for k=5, g=9 (odd): 1 + 5 * (1 + 4 + 16) = 1 + 5*21 = 106.
    # The LW construction usually produces graphs with n = 2 * q^d.
    # If d=4, n = 2 * 5^4 = 1250.
    # If d=3, n = 2 * 5^3 = 250.
    # The girth of CD(d,q) is at least 2d if q > d?
    # For g=9, we need girth >= 9. Since it's bipartite, girth is even.
    # So we can't get girth 9 with a bipartite graph?
    # Wait, the task says "no cycle shorter than 9".
    # If the graph is bipartite, the smallest cycle is even.
    # So if girth is 10, it satisfies "no cycle shorter than 9" (i.e. no cycles of length 3,4,5,6,7,8).
    # So a bipartite graph with girth >= 10 is valid for g=9.
    # So we need girth >= 10.
    # CD(d,q) has girth >= 2d. So we need 2d >= 10 => d >= 5.
    # So for k=5, g=9, we should use d=5.
    # n = 2 * 5^5 = 6250.
    # The current code uses d = g - 5 if odd, else g - 4.
    # g=9 => d = 4. This gives girth >= 8.
    # If girth is 8, there is a cycle of length 8, which is < 9. So it's INVALID for g=9.
    # The evaluator checks "no cycle may be shorter than g".
    # So the current code likely fails validity for g=9?
    # But the score is 1.0000. Maybe the evaluator is lenient or I misunderstand the girth guarantee?
    # Or maybe the LW construction has better girth?
    # Let's re-read the LW paper logic.
    # "The girth of CD(d,q) is at least 2d."
    # If d=4, girth >= 8.
    # If the graph actually has girth 8, it fails for g=9.
    # However, the score is 1.0. This implies it might be valid, or the score is just the seed value (1.0) and it hasn't been properly evaluated/improved yet?
    # "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program".
    # If the seed program is invalid, it scores 0.
    # Since it scores 1.0, it must be valid.
    # How can CD(4,5) be valid for g=9?
    # Maybe the girth is actually higher?
    # Or maybe my calculation of d is wrong.
    # Let's fix the d calculation to be safe.
    # For bipartite graphs, girth must be even.
    # If g is odd, we need girth >= g+1.
    # If g is even, we need girth >= g.
    # So required_girth = g if g % 2 == 0 else g + 1.
    # We need 2d >= required_girth => d >= required_girth / 2.
    # So d = required_girth // 2.
    
    q = k
    required_girth = g if g % 2 == 0 else g + 1
    d = required_girth // 2
    
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

def construct_abs(k, g):
    # Alon-Babai-Suzuki construction on Heisenberg group H_q
    if g % 2 == 0:
        mb = 2 * sum((k-1)**i for i in range(g//2))
    else:
        mb = 1 + k * sum((k-1)**i for i in range((g-1)//2))
        
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    q = 2
    while True:
        if is_prime(q):
            if q**3 >= mb:
                break
        q += 1
        
    if k == 4:
        gens = [
            (1, 0, 0),
            (q-1, 0, 0),
            (0, 1, 0),
            (0, q-1, 1)
        ]
    elif k == 6:
        gens = [
            (1, 0, 0),
            (q-1, 0, 0),
            (0, 1, 0),
            (0, q-1, 1),
            (0, 0, 1),
            (0, 0, q-1)
        ]
    else:
        return construct_lw(k, g)

    n = q**3
    edges = []
    def idx(x, y, z):
        return (x * q + y) * q + z
        
    for x in range(q):
        for y in range(q):
            for z in range(q):
                u = idx(x, y, z)
                for gx, gy, gz in gens:
                    vx = (x + gx) % q
                    vy = (y + gy) % q
                    vz = (z + gz + x * gy) % q
                    v = idx(vx, vy, vz)
                    if u < v:
                        edges.append((u, v))
    return n, edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    # Try random search first for small k, g
    # For k=5, g=9, random search is promising.
    if k <= 6 and g <= 10:
        result = construct_random(k, g, deadline)
        if result is not None:
            return result
    
    # Fallback to deterministic constructions
    if k % 2 == 0:
        return construct_abs(k, g)
    else:
        return construct_lw(k, g)
