"""Voltage lift construction with algebraic voltage assignment and fast girth verification."""
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

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

def _find_primitive_root(m):
    """Find a primitive root modulo m (for prime m, returns smallest generator of Z_m^*)."""
    if m <= 1:
        return 1
    if m == 2:
        return 1
    # Factor phi(m)
    phi = m - 1  # assuming m is prime
    # For prime m, primitive root g has order m-1
    # g is primitive if g^((m-1)/p) != 1 mod m for all prime factors p of m-1
    factors = []
    temp = phi
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    
    for g in range(2, m):
        ok = True
        for p in factors:
            if pow(g, phi // p, m) == 1:
                ok = False
                break
        if ok:
            return g
    return 1  # fallback

def _petersen_edges():
    """Return edge list of Petersen graph (10 vertices, 3-regular)."""
    # Standard Petersen: outer pentagon 0-1-2-3-4-0, inner star 5-7-9-6-8-5, spokes i-(i+5)
    edges = []
    # Outer pentagon
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner star (5-7-9-6-8-5)
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i + 2) % 5]))
    # Spokes
    for i in range(5):
        edges.append((i, i + 5))
    return edges

def _build_voltage_lift(m, voltage_pattern):
    """
    Build a voltage lift of the Petersen graph over Z_m.
    voltage_pattern: dict mapping edge index -> voltage (in Z_m)
    Returns (n, edges) where n = 10*m.
    """
    base_edges = _petersen_edges()
    n = 10 * m
    edges = []
    
    for idx, (u, v) in enumerate(base_edges):
        volt = voltage_pattern.get(idx, 0)
        for coset in range(m):
            eu = u * m + coset
            ev = v * m + (coset + volt) % m
            edges.append((eu, ev))
    
    return n, edges

def _verify_girth_fast(n, edges, min_girth):
    """
    Verify girth >= min_girth using non-backtracking BFS from each vertex.
    Uses proper cycle detection: if we reach (v, u) at distance d from start,
    and (v, u) was previously reached at distance d', then there's a cycle
    of length d + d' + 1 (if d' > 0) or we need to check more carefully.
    
    Actually, for non-backtracking walks: a cycle of length L exists iff
    there's a closed non-backtracking walk of length L. We detect this by
    checking if from start, we can reach (start, prev) at distance L-1 for
    some prev, meaning a non-backtracking walk of length L from start back
    to start (with the last step coming from prev).
    
    Simpler approach: BFS non-backtracking from each vertex, tracking
    distance to each (node, parent) state. If we encounter a state that was
    already visited at a smaller distance, the cycle length is dist_new + dist_old + 1... 
    Actually no. Let me think again.
    
    Non-backtracking BFS from vertex s: state is (current, previous). 
    We start with (s, -1) at distance 0.
    From (u, p) at distance d, we go to (v, u) at distance d+1 for each neighbor v != p of u.
    
    A cycle of length L exists if there's a non-backtracking walk of length L from s back to s.
    That means we reach state (s, p) at distance L-1 for some neighbor p of s.
    
    So: for each start s, do non-backtracking BFS. If we reach (s, p) at distance d where d+1 >= min_girth,
    we're fine. But if we reach (s, p) at distance d where d+1 < min_girth, we found a cycle.
    
    Wait, that's not right either. The non-backtracking walk of length L from s to s with last step from p
    means: s -> ... -> p -> s, which is a cycle of length L if the walk is non-backtracking.
    
    But actually any cycle gives a non-backtracking closed walk. So:
    - Girth < min_girth iff there exists s, p (neighbor of s), and L < min_girth such that
      there's a non-backtracking walk of length L-1 from s to p (not going through s in the middle? no...).
    
    Let me just use: BFS non-backtracking from s. Track dist[(v, u)] = minimum distance to reach state (v, u).
    If we ever find that (s, p) is reachable at distance d where d + 1 < min_girth, then there's a cycle of length d+1.
    
    But we also need to handle: what if the shortest non-backtracking walk from s to (s,p) has length d,
    but there's a shorter cycle that doesn't start/end at s? That's handled by checking from every s.
    
    Actually, a cleaner method: for each vertex s, do non-backtracking BFS up to depth (min_girth-1)//2.
    If at any point we reach a state (v, u) where v == s and the distance is >= 1, 
    we have a cycle of length distance + 1... no wait.
    
    Let me use the standard approach: 
    For each start s:
      dist[(s, -1)] = 0
      For each state (u, p) at distance d:
        For each neighbor v of u with v != p:
          if (v, u) not in dist: dist[(v, u)] = d+1
          elif d+1 < min_girth: 
            # We found (v,u) at distance d+1, but it was already at distance dist[(v,u)]
            # This means there's a cycle of length (d+1) + dist[(v,u)] + 1? No...
            # Actually this means two different non-backtracking walks reach (v,u),
            # giving a closed non-backtracking walk of length (d+1) + dist[(v,u)].
            # But that's not necessarily a simple cycle.
    
    OK let me just use the simplest correct method: 
    For each start s, do BFS on the non-backtracking graph (states = directed edges + source).
    Check if we can return to s via a non-backtracking walk of length < min_girth.
    
    State: (current_node, previous_node). Source state: (s, -1).
    Target states: (s, p) for each neighbor p of s, at distance d where d+1 < min_girth.
    
    If any target state is reachable at distance d with d+1 < min_girth, girth < min_girth.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = min_girth - 1  # We need to check cycles of length 1..min_girth-1
    
    for s in range(n):
        # BFS non-backtracking from s
        # State: (node, parent), distance
        # Start: (s, -1) at dist 0
        # We want to find if (s, p) is reachable at distance d where d+1 < min_girth
        # i.e., d < min_girth - 1, i.e., d <= min_girth - 2
        
        visited = {}  # (node, parent) -> distance
        visited[(s, -1)] = 0
        queue = deque([(s, -1, 0)])
        
        found_short = False
        while queue:
            node, parent, dist = queue.popleft()
            if dist >= min_girth - 1:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                new_state = (nb, node)
                if new_state in visited:
                    # Already reached this state; check if we found a cycle
                    if nb == s and dist + 1 < min_girth:
                        return False
                else:
                    visited[new_state] = dist + 1
                    if nb == s and dist + 1 < min_girth:
                        return False
                    if dist + 1 < min_girth - 1:
                        queue.append((nb, node, dist + 1))
    
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # safety margin (evaluator kills at 150s)
    
    if k != 3:
        # Fallback: simple approach for other k values
        # For now, return a trivial construction or raise
        # We focus on k=3, g=16
        if k == 4:
            # 4-regular: use a circulant or similar
            pass
        # Return something valid even if not optimal
        # For k=3, g arbitrary, try voltage lifts
        pass
    
    # Moore bound for k=3, g=16:
    # For girth g (even), Moore bound: n >= k * (k-1)^(g/2 - 1) / (k-1) ... 
    # Actually for 3-regular, girth 16: n >= 2 * 3 * 2^7 = 384? 
    # Moore bound for cage n(k,g): for even girth g, n >= k*(k-1)^(g/2 - 1)
    # For k=3, g=16: n >= 3 * 2^7 = 384. But n must be even (3-regular => even n).
    # Best lower bound is 512 per the task description.
    moore_bound = 3 * (2 ** 7)  # 384, but use 512 as stated lower bound
    moore_bound = 512
    
    # Try voltage lifts of Petersen graph over Z_m
    # Petersen has 10 vertices, 15 edges
    # Lift over Z_m gives 10m vertices
    # We need 10m >= moore_bound, so m >= 51.2, so m >= 52
    # But 10m must be even (it is, since 10m is always even)
    # And 3-regular: 10m vertices, 15m edges, degree = 15m*2/(10m) = 3. Good.
    
    base_edges = _petersen_edges()
    num_base_edges = len(base_edges)  # 15
    
    best_n = None
    best_edges = None
    
    # Scan m from minimum upward
    min_m = (moore_bound + 9) // 10  # ceil(512/10) = 52
    
    # For each m, try different voltage assignments
    # Strategy: use powers of a primitive root
    # The key insight: assign voltages so that non-trivial closed walks have large order
    
    # Try m values, prioritizing primes and values with good multiplicative structure
    m_candidates = []
    for m in range(min_m, min_m + 200):
        m_candidates.append(m)
    
    # Sort: prefer primes, then values with many prime factors in m-1
    def m_score(m):
        if _is_prime(m):
            return (0, m)  # primes first
        # Count distinct prime factors of m-1
        temp = m - 1
        cnt = 0
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                cnt += 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            cnt += 1
        return (1, m, -cnt)
    
    m_candidates.sort(key=m_score)
    
    for m in m_candidates:
        if time.time() > deadline:
            break
        
        # Try several voltage patterns
        # Pattern 1: all voltages = 1 (trivial, gives disjoint copies)
        # Pattern 2: voltages = powers of primitive root
        # Pattern 3: voltages = i * c for some constant c
        
        # First, try the "powers of primitive root" pattern
        if _is_prime(m) and m > 2:
            g_root = _find_primitive_root(m)
            # Assign edge i voltage = g_root^i mod m
            voltages = {}
            for i in range(num_base_edges):
                voltages[i] = pow(g_root, i, m)
            
            n, edges = _build_voltage_lift(m, voltages)
            
            if n < (best_n if best_n else float('inf')):
                # Verify girth
                if _verify_girth_fast(n, edges, g):
                    best_n = n
                    best_edges = edges
                    # Keep going to find smaller? No, we're scanning m upward, so this is smallest m
                    # But different m might give same n... no, n = 10m is monotonic in m
                    break
        
        # Pattern: constant voltage c
        for c in [1, 2, 3]:
            if time.time() > deadline:
                break
            voltages = {i: c for i in range(num_base_edges)}
            n, edges = _build_voltage_lift(m, voltages)
            if n < (best_n if best_n else float('inf')):
                if _verify_girth_fast(n, edges, g):
                    best_n = n
                    best_edges = edges
                    break
        
        if best_n is not None:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: if no valid lift found, try LUW-like construction
    # For k=3, g=16: LUW with d=11 gives girth >= 16
    # But LUW gives 2*3^11 = 354294 vertices, which is valid but large
    
    def luw_construct(q, d):
        """LUW construction."""
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q

        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = options[r]

        def neighbours(side, x):
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, tuple([0] * (d + 1)))
        index = {start: 0}
        order = [start]
        edges = []

        for vertex in order:
            if time.time() > deadline:
                break
            side, x = vertex
            for other in neighbours(side, x):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[vertex], index[other]))

        n = len(order)
        return n, edges

    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    d = max(3, d)
    n, edges = luw_construct(3, d)
    return n, edges

# EVOLVE-BLOCK-END
