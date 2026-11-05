"""Construct a 3-regular graph of girth >= g using circulant graph search."""
import math
import sys
from collections import deque

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _next_prime(n):
    if n <= 2:
        return 2
    candidate = n if n % 2 else n + 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate

def _compute_girth_circulant(n, a, b):
    """Compute exact girth of 3-regular circulant graph C_n(a, b, -a-b) using non-backtracking BFS."""
    c = (-a - b) % n
    if a == 0 or b == 0 or c == 0:
        return 0
    if len({a, b, c}) != 3:
        return 0
    
    # Build adjacency as tuples for speed
    adj = [((v + a) % n, (v + b) % n, (v + c) % n) for v in range(n)]
    
    best_girth = float('inf')
    
    # Non-backtracking BFS from each directed edge
    for start_u in range(n):
        if best_girth <= g:
            break
        for start_v in adj[start_u]:
            # BFS from (start_v, start_u) at depth 1
            visited = set()
            queue = deque()
            queue.append((start_v, start_u, 1))
            visited.add((start_v, start_u))
            
            found = False
            while queue:
                curr, prev, depth = queue.popleft()
                
                if curr == start_u and depth >= 3:
                    if depth < best_girth:
                        best_girth = depth
                    found = True
                    break
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    state = (nxt, curr)
                    if state not in visited:
                        visited.add(state)
                        if depth + 1 >= best_girth:
                            continue
                        queue.append((nxt, curr, depth + 1))
            
            if found and best_girth <= g:
                break
    
    return best_girth if best_girth != float('inf') else n

def _construct_circulant(n, a, b):
    """Build edges for 3-regular circulant graph with generators a, b, -a-b."""
    c = (-a - b) % n
    edges = []
    for v in range(n):
        for w in [(v + a) % n, (v + b) % n, (v + c) % n]:
            if v < w:
                edges.append((v, w))
    return edges

def _lazebnik_ustimenko_woldar(k, g):
    """Lazebnik-Ustimenko-Woldar construction for 3-regular graphs of high girth."""
    q = 3
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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

def construct(k, g):
    """Construct a 3-regular graph with girth >= g."""
    assert k == 3, "This implementation only handles k=3"
    
    # Moore bound for 3-regular girth g: n >= 2*(2)^(g/2-1) for even g
    if g % 2 == 0:
        moore_bound = 2 * (2 ** (g // 2 - 1))
    else:
        moore_bound = 2 * (2 ** ((g - 1) // 2)) + 1
    
    # Try to find a circulant graph
    best_n = None
    best_edges = None
    
    # First, get a baseline from L-U-W construction
    try:
        luw_n, luw_edges = _lazebnik_ustimenko_woldar(k, g)
        best_n = luw_n
        best_edges = luw_edges
    except:
        pass
    
    # Search for smaller circulant graphs
    # n must be odd for 3-regular circulant with a+b+c=0
    n_start = max(moore_bound + 1, 101)
    
    # Try odd numbers (not just primes) starting from moore bound
    # For circulant C_n(a,b,-a-b), we need gcd(a,n)=gcd(b,n)=gcd(c,n)=1 for connectivity
    # Actually for girth we just need the graph to be well-formed
    
    max_n_to_try = min(best_n - 1 if best_n else 2000, 2000)
    
    for n in range(n_start if n_start % 2 else n_start + 1, max_n_to_try + 1, 2):
        if best_n is not None and n >= best_n:
            break
        
        # Try generator pairs
        # Heuristic: small a, b around n/3 to n/2
        candidates = []
        
        # Small a values
        for a in range(1, min(10, n // 6)):
            # b should be such that c = (-a-b)%n is distinct and nonzero
            # Try b in a range
            for b in range(a + 1, min(a + 30, n // 2)):
                c = (-a - b) % n
                if c == 0 or c == a or c == b:
                    continue
                # Avoid duplicates: assume a < b < c or a < c < b
                if min(a, b, c) == a and max(a, b, c) > b:
                    candidates.append((a, b))
        
        # Also try structured candidates
        for a in [1, 2, 3, 5]:
            for frac in [3, 4, 5, 6, 7, 8]:
                b = n // frac
                if b > a and b < n // 2:
                    c = (-a - b) % n
                    if c not in (0, a, b):
                        candidates.append((a, b))
        
        # Deduplicate
        candidates = list(set(candidates))
        
        found_for_n = False
        for a, b in candidates:
            girth = _compute_girth_circulant(n, a, b)
            if girth >= g:
                edges = _construct_circulant(n, a, b)
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                found_for_n = True
                break
        
        if found_for_n:
            # Found a valid graph for this n; continue to see if smaller n works
            # But we're going up in n, so this is the smallest for this n
            # We can break if we've found something good enough
            if best_n is not None and best_n <= moore_bound * 1.5:
                break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Ultimate fallback
    return _lazebnik_ustimenko_woldar(k, g)
