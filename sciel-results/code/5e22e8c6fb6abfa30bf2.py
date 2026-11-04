import time
import random
import math
from collections import deque
from itertools import product

def _luw_construct(k, g):
    # Fallback to LUW
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

def _check_girth_fast(n, edges, target_girth, deadline=None):
    """Check girth >= target_girth using BFS from each vertex with early termination."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For girth check: verify no cycle shorter than target_girth
    # Use BFS from each vertex, tracking parent to avoid backtracking
    # A cycle of length L exists iff there's a non-backtracking walk of length L-1 from v back to v
    # More efficiently: BFS from each vertex, if we find a neighbor that was visited at depth >= target_girth - 1, 
    # but actually we need to check for cycles.
    
    # Better approach: for each vertex, do BFS and check if any vertex at distance d has an edge to a vertex at distance d-1 (not its parent)
    # This indicates a cycle of length 2*d
    
    # Most efficient: check all non-backtracking walks up to length target_girth-1
    # For each vertex v, count non-backtracking walks of length L ending at v for L = 1..target_girth-1
    # If count > 0 for L = target_girth-1... no wait, a cycle of length C means a non-backtracking walk of length C from v back to v.
    
    # Actually the standard approach: girth >= g iff for every vertex v, there are no non-backtracking walks of length < g from v back to v.
    # Equivalently: for each vertex v, BFS with state (node, prev) and check if we can reach (v, anything) in < g steps.
    
    # For efficiency, let's use the adjacency matrix power approach or just BFS per vertex.
    # With n up to ~10000, BFS per vertex is O(n * (n + e)) which might be too slow.
    
    # Alternative: use the fact that girth >= g iff the number of non-backtracking walks of length L from any vertex back to itself is 0 for L < g.
    # We can compute this via matrix multiplication of the non-backtracking matrix.
    
    # For now, use per-vertex BFS with early termination, but only if n is small enough.
    # For larger n, use a sampling approach or accept the LUW guarantee.
    
    if n > 5000:
        # Too large for per-vertex BFS in reasonable time; use a different check
        # Check via edge-based BFS
        for u, v in edges:
            visited = [-1] * n
            queue = deque()
            queue.append((u, 0, -1))
            visited[u] = 0
            
            found_dist = -1
            while queue:
                node, dist, parent = queue.popleft()
                if node == v:
                    found_dist = dist
                    break
                if dist >= target_girth:
                    break
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if visited[neighbor] == -1:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1, node))
            
            if found_dist != -1:
                cycle_len = found_dist + 1
                if cycle_len < target_girth:
                    return False
        return True
    else:
        # Per-vertex BFS for small graphs
        for start in range(n):
            # State: (node, prev_node)
            # Distance from (start, -1)
            visited = {}
            queue = deque()
            queue.append((start, -1, 0))
            visited[(start, -1)] = 0
            
            while queue:
                node, prev, dist = queue.popleft()
                if dist >= target_girth - 1:
                    continue
                for neighbor in adj[node]:
                    if neighbor == prev:
                        continue
                    new_state = (neighbor, node)
                    if new_state not in visited:
                        visited[new_state] = dist + 1
                        queue.append((neighbor, node, dist + 1))
                        if neighbor == start and dist + 1 < target_girth:
                            return False
        return True

def _cayley_construct(k, g, time_limit):
    """
    Construct a 3-regular Cayley graph over a finite abelian group Z_m1 x Z_m2 x ... 
    with symmetric generating set S = {a, -a, b, -b, c, -c} but we need exactly 3 generators
    for 3-regularity, so S = {a, b, c} where a=-a, b=-b, c=-c (i.e., each has order 2)
    OR S = {a, -a, b} where we pick carefully.
    
    Actually for 3-regular Cayley: we need |S| = 3 with S = S^{-1} and e not in S.
    Options:
    1. S = {a, b, c} where each of a, b, c has order 2 (involution)
    2. S = {a, -a, b} where b has order 2 and a has order > 2
    
    For option 1: group must be elementary 2-group or have elements of order 2.
    For option 2: more flexible.
    
    Let's try option 2: Cayley graph on Z_m with S = {a, -a, b} where b = m/2 (order 2).
    This gives a 3-regular graph on m vertices.
    
    The girth is the shortest word in {a, -a, b} that equals 0 in Z_m, with the constraint
    that consecutive letters cannot be inverses (non-backtracking).
    
    Actually for Cayley graphs, the girth is the length of the shortest non-trivial relation
    in the generators, where "non-trivial" means the word is not empty and doesn't have
    adjacent inverse pairs.
    
    For Z_m with generators {a, -a, b}:
    - b has order 2, so b*b = 0, giving a 2-cycle? No, b is an involution so the edge from v to v+b 
      and back is just traversing the same edge, which is a 2-cycle in the graph sense but 
      in terms of non-backtracking walks, b followed by b is allowed if b != -b.
      Wait, in a Cayley graph, the edge from v to v+b is the same as from v+b to v (undirected).
      So going v -> v+b -> v is a 2-cycle (backtracking).
      
    For non-backtracking: we can't go v -> v+a -> v (that's backtracking on edge a).
    We CAN go v -> v+a -> v+a+b -> ... as long as we don't immediately reverse.
    
    The shortest non-trivial non-backtracking word in {a, -a, b} that equals 0 in Z_m.
    
    Let's search over m and generator choices.
    """
    if k != 3 or g != 16:
        return None
    
    start_time = time.time()
    best_result = None
    best_n = float('inf')
    
    # Try Cayley graphs on Z_m with S = {a, -a, b} where b = m//2 (requires m even)
    # or more generally S = {a, -a, c} where c is some element of order 2.
    
    # For Z_m, elements of order 2: only m//2 if m is even.
    # So S = {a, -a, m//2} for even m.
    
    # The girth is the minimum length of a non-backtracking word w in {a, -a, m//2}
    # such that sum(w) ≡ 0 (mod m), where sum means interpreting a as a, -a as -a, m//2 as m//2.
    # Non-backtracking means no adjacent pair is (x, -x) for x in {a, -a} and (m//2, m//2) is 
    # actually backtracking since it's the same edge.
    
    # Wait: in the Cayley graph, the neighbors of v are v+a, v-a, v+m//2.
    # The edge between v and v+m//2 is undirected, so going v -> v+m//2 -> v is backtracking.
    # Similarly v -> v+a -> v is backtracking (via -a).
    # But v -> v+a -> v+a+m//2 is fine (different edge).
    
    # So non-backtracking constraint: if we just used generator s, we can't use s^{-1} next.
    # For a: inverse is -a. For m//2: inverse is m//2 itself (since 2*(m//2) = m ≡ 0).
    # So after using m//2, we can't use m//2 again.
    
    # Let's do BFS in the word space to find the shortest non-backtracking word summing to 0 mod m.
    
    for m in range(64, 2000, 2):  # even m, start from reasonable size
        if time.time() - start_time > time_limit - 5:
            break
        
        a_val = m // 2  # the involution
        
        # We need to choose a (the other generator). Try various a values.
        # For good girth, a should be "large" relative to m.
        # Try a = 1, a = 2, a = 3, etc., and also a = m//2 ± 1, etc.
        
        candidates_a = [1, 2, 3, 4, 5, m//2 - 1, m//2 - 2, m//2 - 3, m//2 - 4, m//2 - 5]
        candidates_a = [x for x in candidates_a if 0 < x < m and x != a_val]
        
        for a in candidates_a:
            if time.time() - start_time > time_limit - 5:
                break
            
            # BFS in state space (current_sum, last_generator) to find shortest word summing to 0
            # last_generator: 0 = used a, 1 = used -a, 2 = used a_val, -1 = start
            # We want the shortest path from (0, -1) to (0, any) with length >= 1
            # that doesn't use (0, -1) as intermediate.
            
            # Actually, we want non-trivial words, so length >= 1, and the word shouldn't be 
            # just a single generator that's 0 (none are since a, -a, a_val are all non-zero mod m).
            
            # BFS: state = (sum_mod_m, last_gen)
            # last_gen in {-1, 0, 1, 2} where 0=a, 1=-a, 2=a_val
            # Transition from (s, lg):
            #   if lg != 0: can add a -> (s+a)%m, 0
            #   if lg != 1: can add -a -> (s-a)%m, 1  
            #   if lg != 2: can add a_val -> (s+a_val)%m, 2
            #   (if lg == 0, can't add -a; if lg == 1, can't add a; if lg == 2, can't add a_val)
            
            visited = {}  # (sum, last_gen) -> distance
            queue = deque()
            queue.append((0, -1, 0))
            visited[(0, -1)] = 0
            
            girth_found = None
            while queue:
                s, lg, dist = queue.popleft()
                if dist >= g - 1:  # We're looking for words of length < g that sum to 0
                    break
                
                # Try adding each generator
                options = []
                if lg != 1:  # can use a (inverse of -a is a, so if last was -a, can't use a)
                    options.append((a, 0))
                if lg != 0:  # can use -a
                    options.append((-a, 1))
                if lg != 2:  # can use a_val
                    options.append((a_val, 2))
                
                for delta, new_lg in options:
                    new_s = (s + delta) % m
                    new_state = (new_s, new_lg)
                    if new_state not in visited:
                        visited[new_state] = dist + 1
                        if new_s == 0 and dist + 1 >= 1:
                            # Found a word of length dist+1 summing to 0
                            if new_lg != -1:  # non-trivial
                                girth_found = dist + 1
                                queue.clear()
                                break
                        queue.append((new_s, new_lg, dist + 1))
            
            if girth_found is not None:
                if girth_found >= g:
                    # This Cayley graph has girth >= g!
                    n = m
                    if n < best_n:
                        # Build the edge list
                        edges = []
                        for v in range(n):
                            for gen in [a, -a, a_val]:
                                u = (v + gen) % n
                                if v < u:
                                    edges.append((v, u))
                        best_n = n
                        best_result = (n, edges)
                        # Don't break, keep searching for smaller m
            elif girth_found is None and dist < g - 1:
                # No word of length < g-1 found, need to check length g-1
                pass
    
    return best_result

def _try_voltage_graph(k, g, time_limit):
    if k != 3 or g != 16:
        return None
        
    start_time = time.time()
    m = 234
    base_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    
    while time.time() - start_time < time_limit:
        voltages = [random.randint(0, m-1) for _ in range(6)]
        
        g_val = m
        for v in voltages:
            g_val = math.gcd(g_val, v)
            if g_val == 1:
                break
        if g_val != 1:
            continue
            
        n = 4 * m
        edges = []
        for i, (u, v) in enumerate(base_edges):
            vol = voltages[i]
            for offset in range(m):
                u_idx = u * m + offset
                v_idx = v * m + (offset + vol) % m
                edges.append((u_idx, v_idx))
                
        if _check_girth_fast(n, edges, g):
            return n, edges
            
    return None

def construct(k, g):
    deadline = time.time() + 140  # 10s safety margin
    
    # Try Cayley graph construction first (structural change)
    result = _cayley_construct(k, g, 60)
    if result and result[0] < 10000:
        return result
    
    # Try voltage graph
    result = _try_voltage_graph(k, g, 40)
    if result:
        return result
        
    # Fallback to LUW
    return _luw_construct(k, g)
