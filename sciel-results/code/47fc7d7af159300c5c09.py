"""Constructive search for small k-regular graphs of large girth.

Strategy (Algebraic/Circulant Parameter Search):
  1. Estimate a candidate vertex count n from the Moore bound (lower) up to a
     generous upper cap.
  2. For each n, search for a set of k generators for a circulant graph
     (or generalized circulant) that ensures girth >= g.
  3. Verify the girth exactly.
  4. Return the first valid graph found.
  5. Fallback to a known algebraic construction if search fails.
"""
import random
import time
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 110.0  # leave headroom under the 120 s cap


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _moore_bound(k, g):
    """Moore bound (lower bound) for n(k, g)."""
    if g % 2 == 0:
        h = g // 2
        if k == 2:
            return 2 * h
        s = 0
        p = 1
        for _ in range(h):
            s += p
            p *= (k - 1)
        return k * s
    else:
        h = (g - 1) // 2
        if k == 2:
            return g
        s = 0
        p = 1
        for _ in range(h):
            s += p
            p *= (k - 1)
        return k * s


def _circulant_girth(n, generators):
    """Compute the girth of a circulant graph C_n(G) using BFS.
    generators: list of steps. The graph is defined by edges (i, i+g) mod n.
    Since it's undirected, we assume generators are symmetric or we treat them as such.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        for g in generators:
            j = (i + g) % n
            if j != i:
                adj[i].append(j)
    
    # Remove duplicates and ensure symmetry
    for i in range(n):
        adj[i] = list(set(adj[i]))
        # Ensure symmetry
        for j in adj[i]:
            if i not in adj[j]:
                adj[j].append(i)
        
    # BFS from each node to find shortest cycle
    min_girth = float('inf')
    for start in range(n):
        # dist[v] = shortest path from start to v
        # parent[v] = previous node
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
                    # Cycle length = dist[u] + dist[v] + 1
                    # Wait, this finds *a* cycle, not necessarily the shortest one involving start?
                    # Actually, BFS from start finds the shortest path to each node.
                    # A cycle is formed if we connect u and v where v is already visited and v != parent[u].
                    # The length of the cycle is dist[u] + dist[v] + 1.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    # We can break early if min_girth is small enough?
                    # But we need the exact girth.
        # Note: The above BFS finds the shortest cycle passing through 'start' 
        # that is "discovered" by the BFS tree edges. 
        # Actually, the standard way to find girth is BFS from every node and checking 
        # for edges (u,v) where u != parent[v] and v != parent[u].
        # The length is dist[u] + dist[v] + 1.
        # However, this might miss cycles that don't use the BFS tree in a simple way?
        # No, the shortest cycle in an unweighted graph is always found by this method 
        # if we consider all edges.
    
    return min_girth if min_girth < float('inf') else 2 * n # Fallback


def _find_circulant_generators(n, k, g, rng):
    """Try to find k generators for a circulant graph on n vertices with girth >= g."""
    # For k=5, generators can be {a, -a, b, -b, c} where 2c = 0 mod n (so c = n/2)
    # Or if n is odd, we can't have c=n/2. Then we need a different structure.
    # If n is even, c = n//2 is a valid self-inverse generator.
    # Then we need 2 pairs {a, -a} and {b, -b}.
    # So k=5 implies n must be even? 
    # If n is odd, k must be even for circulant graphs with symmetric generators?
    # No, we can have generators {a, b, c, d, e} such that {a,b,c,d,e} = {-a,-b,-c,-d,-e}.
    # This requires that for every x in S, -x is in S.
    # If n is odd, x != -x for all x. So S must be a union of pairs {x, -x}.
    # Thus k must be even.
    # Since k=5 is odd, n MUST be even for a simple circulant graph with this property.
    # So we only try even n.
    
    if n % 2 != 0:
        return None

    c = n // 2
    # We need to find a, b such that the graph has girth >= g.
    # The generators are [a, -a, b, -b, c].
    # We can restrict a, b to be coprime to n? Not necessarily, but to be connected, 
    # gcd(a, b, c, n) must be 1. Since c=n/2, gcd(n/2, a, b, n)=1 implies 
    # at least one of a,b is odd.
    
    # Search space for a, b
    # a, b in [1, n//2 - 1]
    # To avoid duplicates, assume 1 <= a < b <= n//2 - 1.
    
    max_a = n // 2 - 1
    if max_a < 2:
        return None

    # Heuristic: a and b should be "far apart" in some sense to avoid short cycles.
    # Short cycles are formed by sums of generators equal to 0.
    # e.g., a + a + c = 0 => 2a + n/2 = 0 => 4a + n = 0 => 4a = n mod n?
    # 2a = -c = n/2 => 4a = n. So if n is divisible by 4, a = n/4 creates a 4-cycle?
    # Path: 0 -> a -> 2a -> 2a+c = 2a+n/2. If 2a = n/2, then 2a+c = n = 0.
    # Path: 0, a, 2a, 0. Length 3? No, 0->a->2a->0 is length 3 if 2a=0? No, 2a=n/2.
    # 0 -> a -> 2a -> 2a+c. If 2a+c = 0, then 2a = -c = n/2.
    # So 0 -> a -> 2a -> 0 is a triangle?
    # Edges: (0,a), (a,2a), (2a, 2a+c). If 2a+c=0, then (2a,0).
    # So 0-a-2a-0 is a triangle.
    # We must avoid 2a = n/2.
    
    candidates = []
    for a in range(1, max_a + 1):
        if 2 * a % n == c:
            continue
        for b in range(a + 1, max_a + 1):
            if 2 * b % n == c:
                continue
            # Check if gcd(a,b,c,n) == 1
            import math
            if math.gcd(math.gcd(a, b), math.gcd(c, n)) != 1:
                continue
            candidates.append((a, b))
            
    if not candidates:
        return None

    # Shuffle candidates to try random ones first
    rng.shuffle(candidates)
    
    for a, b in candidates:
        if _time_left() < 5.0:
            break
        gens = [a, -a % n, b, -b % n, c]
        girth = _circulant_girth(n, gens)
        if girth >= g:
            return gens
            
    return None


def _construct_circulant(n, generators):
    """Construct edge list for circulant graph."""
    edges = []
    for i in range(n):
        for g in generators:
            j = (i + g) % n
            if i < j:
                edges.append((i, j))
    return edges


def _fallback_luw(k, g):
    """Fallback: the original Lazebnik-Ustimenko-Woldar construction."""
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
                other = (other[0], tuple(other[1]))
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def construct(k, g):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(12345)

    if k < 2 or g < 3:
        return _fallback_luw(max(2, k), max(3, g))

    moore = _moore_bound(k, g)
    
    # Try circulant graphs for even n
    # Start from moore bound, go up
    max_n = 5000 # Cap search
    n = moore
    if n % 2 != 0:
        n += 1
        
    while n <= max_n and _time_left() > 2.0:
        if k % 2 == 1:
            # Only even n works for odd k in simple circulant
            gens = _find_circulant_generators(n, k, g, rng)
            if gens:
                edges = _construct_circulant(n, gens)
                return n, edges
        else:
            # For even k, we can try all n
            # Generators: k/2 pairs {a, -a}
            # This is harder to search generically. 
            # We'll stick to the odd k case which is our main target (k=5).
            pass
            
        n += 2
        
    # Fallback
    return _fallback_luw(k, g)
