"""
Combined cage solver: Circulant search and LUW fallback.
The primary strategy is to search for a circulant graph C_n(S) with n even,
S = {a, -a, b, -b, n/2}, which has degree 5.
This allows us to search n close to the Moore bound.
If no circulant graph is found within the time limit, we fall back to the
Lazebnik-Ustimenko-Woldar construction, which is guaranteed to have
girth >= 9 for appropriate parameters, though it may have more vertices.
"""
import time
import math
import random
from collections import deque

_DEADLINE = 115.0  # Safety margin before 120s limit
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _nonback_girth(adj, n):
    """
    Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g.
    """
    if n == 0:
        return 0
    # If graph is a tree or disconnected, girth is infinity (or 0 if we treat isolated nodes, but here regular)
    # We look for shortest cycle.
    # Non-backtracking BFS from every edge (u, v).
    # dist[(u, v)] = length of shortest non-backtracking walk from some start edge to (u, v).
    # Actually, standard method:
    # For each edge (u, v), start BFS from v with parent u.
    # If we reach u again, we have a cycle.
    
    # Optimization: We can run this in parallel for all edges or just sequentially.
    # Given n is small (likely < 500), sequential is fine.
    
    # To be efficient, we can use the "girth by non-backtracking walks" algorithm.
    # Initialize dist[u][v] = 1 for all edges (u,v).
    # Actually, let's use the standard approach:
    # For each vertex s, do BFS but track parents to detect cycles.
    # Or, for each edge (u,v), start BFS from v, forbidding going back to u.
    # If we reach u, the cycle length is dist + 1.
    
    # Let's implement the edge-based BFS.
    min_girth = float('inf')
    
    # Adjacency list of lists
    # adj[u] is list of neighbors
    
    for u in range(n):
        for v in adj[u]:
            # Start BFS from v, parent u
            # dist[x] = distance from v
            # parent[x] = previous node
            dist = [-1] * n
            parent = [-1] * n
            dist[v] = 1
            parent[v] = u
            q = deque([v])
            
            while q:
                curr = q.popleft()
                for w in adj[curr]:
                    if w == parent[curr]:
                        continue
                    if w == u:
                        # Found cycle: v -> ... -> curr -> u -> v
                        # Length is dist[curr] + 1 (edge curr-u) + 1 (edge u-v)?
                        # Path: v (dist 1) -> ... -> curr (dist d).
                        # Edge curr-u closes the loop to u.
                        # The cycle is v -> ... -> curr -> u -> v.
                        # Length = dist[curr] + 1 (curr to u) + 1 (u to v)?
                        # Wait. dist[v]=1.
                        # Path v -> ... -> curr has length dist[curr].
                        # Edge curr-u adds 1.
                        # Edge u-v adds 1.
                        # Total = dist[curr] + 2?
                        # Let's trace:
                        # v (1) -> x (2) -> ... -> curr (d).
                        # curr -> u (d+1).
                        # u -> v (d+2).
                        # Yes, cycle length is dist[curr] + 2.
                        cycle_len = dist[curr] + 2
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        # We can prune this branch? No, there might be shorter cycles from other paths?
                        # Actually, BFS finds shortest path to curr.
                        # So this is the shortest cycle containing edge (u,v) and passing through curr?
                        # We just need the global min.
                        # We don't need to continue BFS from this specific path if we found a cycle?
                        # But there might be other cycles.
                        # However, if min_girth is already small, we can stop early?
                        # For now, just update min.
                    elif dist[w] == -1:
                        dist[w] = dist[curr] + 1
                        parent[w] = curr
                        q.append(w)
            # Early exit if we found a very small cycle?
            if min_girth <= 3:
                return min_girth
                
    return min_girth

def _is_valid_circulant(n, a, b):
    """
    Check if circulant graph C_n({a, -a, b, -b, n/2}) has girth >= 9.
    n must be even.
    """
    if n % 2 != 0:
        return False, None
    
    # Construct adjacency
    # To save memory/time, we can compute neighbors on the fly or build a small list.
    # n is small, so building adj list is fine.
    adj = [[] for _ in range(n)]
    shifts = [a, -a % n, b, -b % n, n // 2]
    # Ensure shifts are unique
    # a, -a, b, -b, n/2
    # If a == b, invalid.
    # If a == -b, invalid (i.e., a+b == 0 mod n).
    # If 2a == 0 mod n, invalid (a == n/2).
    
    s_set = set(shifts)
    if len(s_set) != 5:
        return False, None
        
    for i in range(n):
        for s in s_set:
            j = (i + s) % n
            adj[i].append(j)
            
    # Check degree
    # Should be 5 by construction if s_set has size 5.
    
    # Check girth
    girth = _nonback_girth(adj, n)
    if girth >= 9:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return True, edges
    return False, None

def _circulant_search(k, g):
    """
    Search for a circulant graph with degree k and girth >= g.
    For k=5, we assume n is even and S={a, -a, b, -b, n/2}.
    """
    if k != 5:
        return None
        
    # Moore bound
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # Search n from moore upwards
    # n must be even
    # Start at first even >= moore
    start_n = moore if moore % 2 == 0 else moore + 1
    
    # Limit search space to keep time reasonable
    # If we don't find it in n < moore + 200, we give up on circulant
    max_n = start_n + 200
    
    for n in range(start_n, max_n + 1, 2):
        if _elapsed() > _DEADLINE:
            return None
            
        # Search for a, b
        # a in 1 .. n/2 - 1
        # b in 1 .. n/2 - 1
        # a != b
        # a != n - b (i.e., a + b != n)
        # a != n/2, b != n/2
        
        # Optimization:
        # We can iterate a, then b.
        # To speed up, we can check for small cycles analytically?
        # For now, brute force with early termination in _is_valid_circulant.
        
        for a in range(1, n // 2):
            if _elapsed() > _DEADLINE:
                return None
            for b in range(a + 1, n // 2):
                # Check if a and b are distinct and not negatives of each other
                # b is in (a, n/2), so b != a.
                # b != -a mod n  => b != n - a.
                # Since b < n/2 and a < n/2, n - a > n/2. So b != n - a is always true.
                # So just checking b > a is enough for distinctness and non-negative?
                # Wait, S = {a, n-a, b, n-b, n/2}.
                # We need these 5 to be distinct.
                # a != n-a => 2a != 0 => a != n/2 (satisfied by range)
                # b != n-b => b != n/2 (satisfied)
                # a != b (satisfied by b > a)
                # a != n-b => a + b != n.
                # n-a != b => n - a != b => a + b != n.
                # n-a != n-b => a != b.
                # n-a != n/2 => a != n/2.
                
                if a + b == n:
                    continue
                    
                valid, edges = _is_valid_circulant(n, a, b)
                if valid:
                    return n, edges
                    
    return None

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    """
    q = k
    # For k=5, g=9, d=4 works.
    # General rule: d = max(3, (g-1)//2 + 1) ?
    # For g=9, (9-1)//2 + 1 = 5.
    # Let's check d=4 vs d=5.
    # Girth of CD(d, q) is at least 2*min(d, 2q-1) + 1.
    # For q=5, 2q-1 = 9.
    # If d=4, min(4, 9) = 4. Girth >= 2*4 + 1 = 9.
    # So d=4 is sufficient.
    # Number of vertices = q^(d+1).
    # d=4 -> 5^5 = 3125.
    # d=3 -> 5^4 = 625. Girth >= 2*3+1 = 7. (Too small for g=9).
    # So d=4 is the minimum for g=9 with q=5.
    
    d = 4
    if g > 9:
        # Increase d if needed
        # 2*min(d, 2q-1) + 1 >= g
        # min(d, 2q-1) >= (g-1)//2
        # If (g-1)//2 <= 2q-1, then d = (g-1)//2.
        # Else d = 2q-1.
        required = (g - 1) // 2
        if required <= 2 * q - 1:
            d = required
        else:
            d = 2 * q - 1
            
    # Ensure d >= 3 for the rule definitions
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
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r_idx = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]
    
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
            yield (1 - side), tuple(y)
    
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

def construct(k, g):
    global _start_time
    _start_time = time.time()
    
    # Try circulant search first
    result = _circulant_search(k, g)
    if result is not None:
        return result
        
    # Fallback to LUW
    return _luw_construct(k, g)
