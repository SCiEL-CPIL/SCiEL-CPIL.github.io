"""
Cage problem n(4,10): construct a 4-regular graph of girth >= 10.

Strategy:
1. Try to find a 4-regular Cayley graph on a cyclic group Z_m with girth >= g.
   This is fast to verify using non-backtracking walk enumeration.
2. If no cyclic construction is found in time, fall back to the Lazebnik-
   Ustimenko-Woldar (LUW) construction, which gives a valid (but larger) graph.

The evaluator scores n (lower is better), so we want the smallest n possible.
The Moore bound for (4,10) is 243. The smallest known is 320.
"""
import math
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (i, j) tuples.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    if k != 4:
        return _construct_fallback(k, g)
    
    # Try cyclic Cayley graphs Z_m with S = {a, -a, b, -b}
    best_n = None
    best_edges = None
    
    # Moore bound for (4,10) is 243
    moore_bound = _moore_bound(k, g)
    
    # Try m from moore_bound upward
    for m in range(moore_bound, moore_bound + 200):
        if time.time() > deadline - 30:
            break
        
        # For efficiency, try small a values first (a=1 is often good)
        # We need gcd(a,b,m)=1 for connectedness
        # And a, b should be "small" to avoid short relations
        
        found = False
        # Try a from 1 to min(m//3, 20) - keep a small for efficiency
        max_a = min(m // 3, 30)
        for a in range(1, max_a + 1):
            if time.time() > deadline - 30:
                break
            if math.gcd(a, m) != 1:
                continue
            
            # Try b from a+1 to m//2
            # But we can prune: if b is too close to a or m-a, skip
            max_b = m // 2
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 20:
                    break
                
                # Quick necessary checks
                if (a + b) % m == 0:
                    continue
                if (2 * a) % m == 0 or (2 * b) % m == 0:
                    continue
                if math.gcd(math.gcd(a, b), m) != 1:
                    continue
                
                # Full girth check
                if _check_girth_cyclic(m, a, b, g):
                    # Construct edges
                    edge_set = set()
                    for i in range(m):
                        for s in [a, b]:
                            j = (i + s) % m
                            if i < j:
                                edge_set.add((i, j))
                            else:
                                edge_set.add((j, i))
                    edges = sorted(edge_set)
                    
                    if best_n is None or m < best_n:
                        best_n = m
                        best_edges = edges
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: LUW construction
    return _construct_fallback(k, g)


def _moore_bound(k, g):
    """Compute the Moore bound for a k-regular graph of girth g."""
    if g % 2 == 0:
        # Even girth: n >= k * (k-1)^(g/2 - 1) / (k-1) + ... 
        # Actually: n >= 2 * (k-1)^(g/2 - 1) * k / 2 + ... 
        # Standard formula: for even g = 2r, n >= k * sum_{i=0}^{r-1} (k-1)^i
        r = g // 2
        total = 0
        for i in range(r):
            total += (k - 1) ** i
        return k * total
    else:
        # Odd girth: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        r = (g - 1) // 2
        total = 1
        for i in range(r):
            total += (k - 1) ** i
        return k * total


def _check_girth_cyclic(m, a, b, g):
    """
    Check that the Cayley graph of Z_m with S={a,-a,b,-b} has girth >= g.
    Uses BFS on non-backtracking states to detect closed walks of length < g.
    """
    steps = [a, -a, b, -b]
    inv = {a: -a, -a: a, b: -b, -b: b}
    
    # For each length L from 2 to g-1, check if any non-backtracking
    # walk of length L sums to 0 mod m.
    
    # Use iterative deepening with state (position, last_step)
    # Start: for each s in steps, state is (s % m, s)
    # After L-1 more steps, check if position == 0
    
    # For L=2: (s1, s2) with s2 != -s1, s1+s2=0 mod m
    #   s2 = -s1, but s2 != -s1 is violated. So no L=2.
    
    for L in range(2, g):
        # BFS: state = (pos, last_step)
        # After 1 step: {(s%m, s) for s in steps}
        # After L steps: expand L-1 times
        
        # Use a set for visited states at each level to avoid explosion
        # But we need to track the walk, not just visited states
        # Actually, we just need to know if ANY non-backtracking walk of length L
        # sums to 0. So we can do BFS on (pos, last_step) states.
        
        if L == 1:
            continue
        
        # Level 1: all (s%m, s) for s in steps
        current = set()
        for s in steps:
            current.add((s % m, s))
        
        for _ in range(L - 1):
            next_level = set()
            for (pos, last) in current:
                for s in steps:
                    if s == inv[last]:
                        continue
                    new_pos = (pos + s) % m
                    next_level.add((new_pos, s))
            
            # Check if any state has pos == 0
            for (pos, last) in next_level:
                if pos == 0:
                    return False
            
            current = next_level
    
    return True


def _construct_fallback(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction: a q-regular bipartite graph
    with girth at least d+5 for odd d, where q = k.
    """
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
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
