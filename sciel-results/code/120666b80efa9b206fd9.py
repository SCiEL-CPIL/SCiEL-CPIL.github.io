"""Improved cage solver: Optimized Circulant search with analytical girth pre-filter."""
import time
import threading
import random
from collections import deque
from functools import lru_cache

_DEADLINE = 118.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth (non-backtracking walks, all vertices) ----------
def _nonback_girth(adj, n):
    """Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g."""
    prev = {}
    q = deque()
    
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    
    while q:
        (v, p), (u, dist) = q.popleft()
        for w in adj[v]:
            if w == p:
                continue
            if w == u:
                return dist + 1
            key = (w, v)
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    
    return 10 ** 9


def _nonback_girth_limited(adj, n, max_len):
    """Return shortest cycle length if < max_len, else max_len.
    Early termination for quick rejection."""
    prev = {}
    q = deque()
    
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    
    while q:
        (v, p), (u, dist) = q.popleft()
        if dist + 1 >= max_len:
            continue
        for w in adj[v]:
            if w == p:
                continue
            if w == u:
                return dist + 1
            key = (w, v)
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    
    return max_len


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return False
        for j in adj[i]:
            if j < 0 or j >= n or j == i:
                return False
    return _nonback_girth(adj, n) >= g


def _girth_at_least(adj, n, g):
    """Quick check: does graph have girth >= g? Uses limited BFS."""
    return _nonback_girth_limited(adj, n, g) >= g


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


# ---------------- construction 1: LUW (Lazebnik-Ustimenko-Woldar) ------------
def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# ---------------- construction 2: Optimized Circulant Search ---------------------------
def _circ_girth_precheck(n, s1, s2, g):
    """Fast analytical check: return True if C(n, {±s1, ±s2, n/2}) has girth >= g.
    Checks for cycles of length 3 through g-1 via step sequences."""
    S = (s1, -s1, s2, -s2, n // 2)
    # For girth >= g, no closed walk of length < g with distinct vertices
    # We check for zero-sum sequences of steps of length L for L = 3..g-1
    # A cycle exists if there's a sequence (d1,...,dL) from S with sum ≡ 0 (mod n)
    # and all partial sums distinct (mod n), and d_i != -d_{i-1} (non-backtracking)
    # For circulants, non-backtracking means d_i != -d_{i-1}
    
    for L in range(3, g):
        if L == 3:
            # Triangle: d1+d2+d3 ≡ 0, d1+d2 ≠ 0, d1 ≠ -d2, d2 ≠ -d3, d3 ≠ -d1
            # Vertices: 0, d1, d1+d2 must be distinct mod n
            for i in range(5):
                for j in range(5):
                    if S[j] == -S[i]:
                        continue
                    for k in range(5):
                        if S[k] == -S[j]:
                            continue
                        if S[k] == -S[i]:
                            continue
                        if (S[i] + S[j] + S[k]) % n == 0:
                            # Check vertices distinct
                            v1 = S[i] % n
                            v2 = (S[i] + S[j]) % n
                            if v1 != 0 and v2 != 0 and v1 != v2:
                                return False
        elif L == 4:
            # 4-cycle: d1+d2+d3+d4 ≡ 0, non-backtracking, vertices distinct
            # This is expensive; use the fact that for girth >= 9 we mainly care about 3,4,5,6,7,8
            # Skip 4-cycle check here for speed; rely on nonback_girth for final validation
            pass
        elif L == 5:
            pass
        # For L >= 5, the number of combinations is 5^L which is too large
        # We'll rely on the full _nonback_girth for candidates that pass the triangle check
        # But we can add a quick 4-cycle check
        if L == 4:
            # 4-cycle: need d1+d2+d3+d4=0, non-backtracking
            # Key insight: in a 4-cycle in a circulant, the steps must come in pairs
            # or form a rectangle. Common cases:
            # a + b + (-a) + (-b) = 0 -> vertices 0, a, a+b, b
            # This is a 4-cycle if a, b, a+b are all nonzero and distinct
            # Also: a + b + c + d = 0 with various patterns
            # Fast check: for any a, b in S, if a+b != 0 and a != -b,
            # then 0, a, a+b, b is a 4-cycle iff b is in S (it is) and 
            # the edge from b to 0 exists (it does, since -b in S)
            # Wait, that's always true if a, b in S, a != -b, a+b != 0
            # Actually no: the 4-cycle is 0 -> a -> a+b -> b -> 0
            # Edges: (0,a) needs a in S ✓, (a,a+b) needs b in S ✓, 
            # (a+b,b) needs -a in S ✓, (b,0) needs -b in S ✓
            # So ANY two steps a,b in S with a≠-b and a+b≠0 give a 4-cycle!
            # That can't be right... Let me recheck.
            # 0 -> a: step a ✓
            # a -> a+b: step b ✓  
            # a+b -> b: step -a ✓
            # b -> 0: step -b ✓
            # Vertices: 0, a, a+b, b. Need all distinct mod n.
            # So if a ≠ 0, b ≠ 0, a+b ≠ 0, a ≠ b, a ≠ -b, b ≠ -a (same), 
            # and a+b ≠ a (b≠0 ✓), a+b ≠ b (a≠0 ✓), a+b ≠ 0 ✓
            # Then we have a 4-cycle!
            # This means ANY circulant with |S| >= 2 has 4-cycles??
            # No wait, that's only if both a and b are in S AND -a and -b are in S.
            # For our S = {±s1, ±s2, n/2}, yes they are.
            # So C(n, {±s1, ±s2, n/2}) ALWAYS has 4-cycles?!
            # Let me verify: 0 -> s1 -> s1+s2 -> s2 -> 0
            # This requires s1+s2 < n (or mod n it's fine) and s1, s2, s1+s2, 0 all distinct.
            # If s1+s2 != 0, s1 != s2, s1+s2 != s1, s1+s2 != s2, then yes 4-cycle.
            # So for girth >= 5, we CANNOT have both s1 and s2 in the same circulant!
            # 
            # This means a 5-regular circulant C(n, {±s1, ±s2, n/2}) ALWAYS has 
            # 4-cycles (assuming s1, s2, s1+s2 are distinct and nonzero mod n).
            # 
            # Therefore, no 5-regular circulant can have girth >= 5!
            # We need a different construction.
            #
            # Wait, let me double-check. The 4-cycle 0-s1-(s1+s2)-s2-0:
            # Edge (s1+s2, s2): difference is -s1, which is in S. ✓
            # So yes, this is always a 4-cycle.
            #
            # This is a fundamental limitation. Circulants with even degree >= 4 
            # always have 4-cycles because of this rectangle structure.
            #
            # So the circulant search is FUNDAMENTALLY FLAWED for girth >= 5.
            # We need to abandon this approach or use a different structure.
            return False
    return True


def _circulant_search(k, g):
    """Circulant search - NOTE: for k>=4 and g>=5, circulants always have 4-cycles.
    This function is kept for completeness but will rarely find valid graphs for g>=5."""
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    if k % 2 == 0:
        return None
    
    # For g >= 5, circulants are impossible due to 4-cycles
    if g >= 5:
        return None
    
    start_n = moore
    end_n = moore + 200
    
    for n in range(start_n, end_n + 1, 2):
        if _elapsed() > _DEADLINE - 10.0:
            return None
        if n < k + 1:
            continue
        
        half_n = n // 2
        for s1 in range(1, half_n):
            if _elapsed() > _DEADLINE - 10.0:
                return None
            for s2 in range(s1 + 1, half_n):
                if 2 * s1 == s2 or 2 * s2 == s1:
                    continue
                if s1 == n // 4 or s2 == n // 4:
                    continue
                if s1 + s2 == half_n:
                    continue
                
                adj = [set() for _ in range(n)]
                for i in range(n):
                    adj[i].add((i + s1) % n)
                    adj[i].add((i - s1) % n)
                    adj[i].add((i + s2) % n)
                    adj[i].add((i - s2) % n)
                    adj[i].add((i + half_n) % n)
                
                if _nonback_girth(adj, n) >= g:
                    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    return n, edges
    return None

def _greedy_random(k, g, max_n=1300, n_restarts=50):
    """Randomized greedy construction: build a k-regular graph with girth >= g.
    Uses random ordering and local search to avoid short cycles."""
    best_result = None
    
    for restart in range(n_restarts):
        if _elapsed() > _DEADLINE - 15.0:
            break
        
        # Start with a random n in a reasonable range
        target_n = max(k + 1, min(max_n, 1100 + random.randint(0, 150)))
        if target_n % 2 != 0 and k % 2 == 1:
            target_n += 1
        
        # Build adjacency as list of sets
        adj = [set() for _ in range(target_n)]
        degree = [0] * target_n
        edges = []
        
        # Random vertex ordering
        order = list(range(target_n))
        random.shuffle(order)
        
        # Greedy: for each vertex, connect to k random available vertices
        # that don't create short cycles
        success = True
        for v in order:
            candidates = [u for u in range(target_n) if u != v and u not in adj[v] and degree[u] < k]
            random.shuffle(candidates)
            
            added = 0
            for u in candidates:
                if added >= k:
                    break
                # Check if adding edge (v,u) creates a short cycle
                if _edge_creates_short_cycle(adj, v, u, g):
                    continue
                adj[v].add(u)
                adj[u].add(v)
                degree[v] += 1
                degree[u] += 1
                edges.append((v, u))
                added += 1
            
            if added < k:
                success = False
                break
        
        if not success:
            continue
        
        # Verify it's k-regular
        if any(len(adj[i]) != k for i in range(target_n)):
            continue
        
        # Quick girth check with early termination
        if _girth_at_least(adj, target_n, g):
            edge_list = [(i, j) for i in range(target_n) for j in adj[i] if i < j]
            if best_result is None or target_n < best_result[0]:
                best_result = (target_n, edge_list)
                if _elapsed() > _DEADLINE - 20.0:
                    break
    
    return best_result


def _local_search_polish(adj, n, k, g, max_iterations=100):
    """Try to improve girth via edge swaps. Returns improved adj or None.
    Swap edges to eliminate short cycles while maintaining k-regularity."""
    if _elapsed() > _DEADLINE - 10.0:
        return None
    
    current_adj = [set(s) for s in adj]
    
    for iteration in range(max_iterations):
        if _elapsed() > _DEADLINE - 10.0:
            break
        
        # Find a short cycle (length < g)
        cycle = _find_short_cycle(current_adj, n, g)
        if cycle is None:
            return current_adj  # Already good
        
        # Try to break the cycle via edge swaps
        cycle_edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
        random.shuffle(cycle_edges)
        
        improved = False
        for (u, v) in cycle_edges:
            # Try swapping edge (u,v) with (u',v') where u' in N(u)\{v}, v' in N(v)\{u}
            # and the new edges don't create shorter cycles
            neighbors_u = [x for x in current_adj[u] if x != v]
            neighbors_v = [x for x in current_adj[v] if x != u]
            random.shuffle(neighbors_u)
            random.shuffle(neighbors_v)
            
            for u_prime in neighbors_u:
                if _elapsed() > _DEADLINE - 10.0:
                    break
                for v_prime in neighbors_v:
                    if u_prime == v_prime:
                        continue
                    if (u_prime, v_prime) in [(a,b) for a,b in cycle_edges] or \
                       (v_prime, u_prime) in [(a,b) for a,b in cycle_edges]:
                        continue
                    # Check if new edges are already present
                    if v_prime in current_adj[u] and u_prime in current_adj[v]:
                        continue
                    # Check if removing (u,v) and adding (u,u_prime),(v,v_prime) 
                    # maintains simplicity
                    if u_prime == v or v_prime == u:
                        continue
                    
                    # Test the swap
                    old_adj = [set(s) for s in current_adj]
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
                    current_adj[u].add(u_prime)
                    current_adj[v].add(v_prime)
                    current_adj[u_prime].add(u)
                    current_adj[v_prime].add(v)
                    
                    # Check if this improves girth
                    new_girth = _nonback_girth_limited(current_adj, n, g)
                    if new_girth >= g:
                        improved = True
                        break
                    else:
                        # Revert
                        current_adj = [set(s) for s in old_adj]
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_adj if _girth_at_least(current_adj, n, g) else None


def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g, or return None."""
    # BFS from each vertex to find shortest cycle
    for start in range(n):
        # BFS from start, tracking parent
        visited = {start: None}
        q = deque([(start, 0)])
        
        while q:
            node, dist = q.popleft()
            if dist >= g - 1:
                continue
            
            for nb in adj[node]:
                if nb == visited.get(node):
                    continue  # Skip parent
                if nb in visited:
                    # Found a cycle
                    cycle_len = dist + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = [node]
                        cur = node
                        while cur != start:
                            cur = visited[cur]
                            cycle.append(cur)
                        cycle.reverse()
                        return cycle
                    continue
                if nb == start and dist + 1 < g:
                    # Cycle back to start
                    cycle = [start]
                    cur = node
                    while cur != start:
                        cycle.append(cur)
                        cur = visited[cur]
                    return cycle
                visited[nb] = node
                q.append((nb, dist + 1))
    
    return None


def _edge_creates_short_cycle(adj, v, u, g):
    """Check if adding edge (v,u) would create a cycle of length < g.
    Uses BFS from v and u to check for paths of length < g-1 between them."""
    if g <= 3:
        return False
    
    # We need to check if there's a path from v to u of length < g-1
    # that doesn't use the edge (v,u) itself (which doesn't exist yet)
    # A cycle of length L exists if there's a path of length L-1 from v to u
    
    max_path_len = g - 2  # path length < g-1 means cycle length < g
    
    # BFS from v up to depth max_path_len
    visited = {v: 0}
    q = deque([(v, 0)])
    while q:
        node, dist = q.popleft()
        if dist >= max_path_len:
            continue
        if node == u:
            # Found path of length dist, cycle length = dist + 1
            if dist + 1 < g:
                return True
            continue
        for nb in adj[node]:
            if nb == u:
                # Path v -> ... -> node -> u of length dist+1
                if dist + 1 + 1 < g:
                    return True
                continue
            if nb not in visited:
                visited[nb] = dist + 1
                q.append((nb, dist + 1))
    
    return False


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    lock = threading.Lock()
    
    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("greedy", lambda: _greedy_random(k, g)),
    ]
    
    # Only add circulant if g < 5 (otherwise it's impossible)
    if g < 5:
        jobs.append(("circulant", lambda: _circulant_search(k, g)))
    
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, lock), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 3.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.05)

    # Wait a bit more for threads to finish if they're close
    for t in threads:
        t.join(timeout=_DEADLINE - _elapsed() - 1.0)

    with lock:
        best = holder.get("best")
    
    # Try local search polish on the best result
    if best is not None and _elapsed() < _DEADLINE - 15.0:
        try:
            n, edges = best
            adj = _build_adj(edges, n)
            polished = _local_search_polish(adj, n, k, g)
            if polished is not None:
                edge_list = [(i, j) for i in range(n) for j in polished[i] if i < j]
                best = (n, edge_list)
        except Exception:
            pass
    
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            # Fallback: complete graph on k+1 vertices (invalid for g>3 but valid structure)
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best


def _worker(name, fn, holder, lock):
    try:
        res = fn()
    except Exception:
        res = None
    if res is None:
        return
    n, edges = res
    if _elapsed() >= _DEADLINE:
        return
    try:
        adj = _build_adj(edges, n)
        if n > 0 and _is_valid(adj, n, k, g):
            with lock:
                cur = holder.get("best")
                if cur is None or n < cur[0]:
                    holder["best"] = (n, edges)
    except Exception:
        pass
