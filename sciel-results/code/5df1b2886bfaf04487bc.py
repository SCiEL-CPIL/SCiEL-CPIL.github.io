"""Stronger program: LUW baseline + optimized girth check + targeted search for smaller k-regular girth-g graphs."""
import time
import random
from collections import deque, defaultdict

_DEADLINE = 112  # safety margin before 120s wall-clock kill

def _time_left():
    return _DEADLINE - time.time()

def _moore_lower(k, g):
    """Moore bound: minimum possible n for k-regular graph with girth g."""
    if g % 2 == 0:
        r = g // 2
        # n >= k * (k-1)^(r-1) / 2 * ... simplified: for even girth g=2r
        # n >= k * ((k-1)^(r-1) - 1) / (k-2) * 2 / k ... use standard form
        # For girth g (even): n >= k * sum_{i=0}^{r-1} (k-1)^i = k * ((k-1)^r - 1)/(k-2)
        # Actually the Moore bound for k-regular, girth g:
        # if g even (g=2r): n >= k * ((k-1)^(r-1) - 1) / (k-2) * 2 ... let me use the tree bound
        # A tree of depth r-1 from each vertex: number of distinct vertices >= 2 * sum_{i=0}^{r-1}(k-1)^i
        # = 2 * ((k-1)^r - 1)/(k-2) for k>2
        r = g // 2
        if k <= 2:
            return g
        return 2 * ((k - 1) ** r - 1) // (k - 2)
    else:
        # g odd (g=2r+1): center vertex, tree of depth r
        # n >= 1 + k * sum_{i=0}^{r-1}(k-1)^i = 1 + k * ((k-1)^r - 1)/(k-2)
        r = (g - 1) // 2
        if k <= 2:
            return g
        return 1 + k * ((k - 1) ** r - 1) // (k - 2)

def _check_girth(adj, n, g):
    """Check no cycle shorter than g using non-backtracking BFS from each vertex.
    Returns True if girth >= g. Uses integer-encoded states for speed."""
    if n == 0:
        return True
    max_depth = (g - 1) // 2
    # Encode (node, prev) as node * (n+1) + (prev+1), where prev=-1 encodes to 0
    stride = n + 1
    for start in range(n):
        visited = set()
        queue = deque()
        # start with prev=-1 -> encoded as start*stride + 0
        queue.append(start * stride)
        visited.add(start * stride)
        depth = 0
        while queue and depth <= max_depth:
            next_queue = deque()
            for _ in range(len(queue)):
                state = queue.popleft()
                node = state // stride
                prev_enc = state % stride  # 0 means prev=-1, else prev=prev_enc-1
                for nb in adj[node]:
                    # Skip backtracking
                    if prev_enc != 0 and nb == prev_enc - 1:
                        continue
                    new_state = nb * stride + (node + 1)
                    if new_state in visited:
                        # Found a cycle of length depth+2 (the edge closes the loop)
                        if depth + 2 < g:
                            return False
                        continue
                    visited.add(new_state)
                    next_queue.append(new_state)
            queue = next_queue
            depth += 1
    return True

def _check_degree(adj, n, k):
    """Check every vertex has degree exactly k."""
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True

def _check_simple(adj, n):
    """Check no loops, no repeated edges."""
    for i in range(n):
        if i in adj[i]:
            return False
        s = set(adj[i])
        if len(s) != len(adj[i]):
            return False
    return True

def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if not _check_simple(adj, n):
        return False
    if not _check_degree(adj, n, k):
        return False
    if not _check_girth(adj, n, g):
        return False
    return True

def _projective_plane_construct(k, g):
    """Construct a k-regular graph with high girth using projective plane of order q.
    Uses the incidence graph of the projective plane (which is (q+1)-regular),
    then removes edges to reduce degree to exactly k while preserving girth."""
    # Find smallest prime power >= k
    def is_prime_power(m):
        if m < 2:
            return 0
        for p in range(2, m + 1):
            pp = p
            while pp < m:
                pp *= p
            if pp == m:
                return p
        return 0
    
    q = k
    while is_prime_power(q) == 0:
        q += 1
    
    if q == 2:
        # Projective plane of order 2: Fano plane, 7 points, 7 lines
        # Incidence graph is 3-regular on 14 vertices (Heawood graph, girth 6)
        # For k=3, girth must be >= 6, but we need higher girth
        # Fano plane incidence graph has girth 6, so only valid for g <= 6
        if k == 3 and g <= 6:
            # Heawood graph edges
            edges = [(0,7),(0,8),(0,9),(1,7),(1,10),(1,11),(2,7),(2,12),(2,13),
                     (3,8),(3,10),(3,13),(4,8),(4,11),(4,12),(5,9),(5,10),(5,12),
                     (6,9),(6,11),(6,13)]
            return 14, edges
        return None, None
    
    # Build finite field GF(q)
    if q == 3:
        def gf_add(a, b): return (a + b) % q
        def gf_sub(a, b): return (a - b) % q
        def gf_mul(a, b): return (a * b) % q
        def gf_inv(a):
            for x in range(q):
                if gf_mul(a, x) == 1:
                    return x
            return 0
    elif q == 5:
        def gf_add(a, b): return (a + b) % q
        def gf_sub(a, b): return (a - b) % q
        def gf_mul(a, b): return (a * b) % q
        def gf_inv(a):
            for x in range(q):
                if gf_mul(a, x) == 1:
                    return x
            return 0
    else:
        # General prime power (for small q, use modular arithmetic if prime)
        def gf_add(a, b): return (a + b) % q
        def gf_sub(a, b): return (a - b) % q
        def gf_mul(a, b): return (a * b) % q
        def gf_inv(a):
            for x in range(q):
                if gf_mul(a, x) == 1:
                    return x
            return 0
    
    # Projective plane of order q has q^2 + q + 1 points and same number of lines
    # Points: (x, y) for x,y in GF(q), plus (t, 1) for t in GF(q), plus (0, 1) [infinity]
    # Actually standard: points are 1D subspaces of GF(q)^3
    # Lines are 2D subspaces of GF(q)^3
    
    # Represent points as canonical vectors:
    # Type 0: (1, x, y) for x,y in GF(q) -> q^2 points
    # Type 1: (0, 1, t) for t in GF(q) -> q points  
    # Type 2: (0, 0, 1) -> 1 point
    # Total: q^2 + q + 1 points
    
    # Lines: a line is defined by ax + by + cz = 0 (not all zero)
    # Canonical: if a != 0: (1, b/a, c/a); if a=0, b!=0: (0, 1, c/b); if a=b=0: (0,0,1)
    # Same parametrization as points!
    
    # Incidence: point (a,b,c) is on line (a',b',c') iff aa' + bb' + cc' = 0
    
    def point_coords(idx):
        """Convert point index to (a,b,c) coordinates."""
        if idx < q * q:
            x = idx % q
            y = idx // q
            return (1, x, y)
        elif idx < q * q + q:
            t = idx - q * q
            return (0, 1, t)
        else:
            return (0, 0, 1)
    
    def line_coords(idx):
        """Convert line index to (a',b',c') coordinates (same as point)."""
        return point_coords(idx)
    
    n_pts = q * q + q + 1
    n_lines = q * q + q + 1
    total_vertices = n_pts + n_lines
    
    # Build incidence: point i is connected to line j iff dot product = 0
    adj = [set() for _ in range(total_vertices)]
    for i in range(n_pts):
        pa, pb, pc = point_coords(i)
        for j in range(n_lines):
            la, lb, lc = line_coords(j)
            if gf_add(gf_add(gf_mul(pa, la), gf_mul(pb, lb)), gf_mul(pc, lc)) == 0:
                adj[i].add(n_pts + j)
                adj[n_pts + j].add(i)
    
    # This is (q+1)-regular. We need to reduce to k-regular.
    # Strategy: for each vertex, remove (q+1-k) edges.
    # To preserve girth, we should remove edges that are "far apart" in the graph.
    # Simple approach: randomly remove edges, then verify girth.
    
    if k >= q + 1:
        # Already k-regular or higher; but we need exactly k
        # If k == q+1, we're done
        if k == q + 1:
            edges = []
            for i in range(total_vertices):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return total_vertices, edges
        # k > q+1 is impossible with this construction
        return None, None
    
    if k == q + 1:
        edges = []
        for i in range(total_vertices):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return total_vertices, edges
    
    # Need to reduce from (q+1)-regular to k-regular
    # Remove (q+1-k) edges per vertex
    remove_per_vertex = q + 1 - k
    
    # Try random edge removal
    import random
    for attempt in range(50):
        adj_copy = [set(s) for s in adj]
        # For each vertex, randomly select (q+1-k) neighbors to remove
        removed_edges = set()
        valid = True
        for i in range(total_vertices):
            if len(adj_copy[i]) < k:
                valid = False
                break
            neighbors = list(adj_copy[i])
            to_remove = random.sample(neighbors, remove_per_vertex)
            for nb in to_remove:
                if (min(i,nb), max(i,nb)) not in removed_edges:
                    removed_edges.add((min(i,nb), max(i,nb)))
                    adj_copy[i].discard(nb)
                    adj_copy[nb].discard(i)
        
        if not valid:
            continue
        
        # Check all degrees are exactly k
        ok = True
        for i in range(total_vertices):
            if len(adj_copy[i]) != k:
                ok = False
                break
        if not ok:
            continue
        
        # Check girth
        adj_list = [list(s) for s in adj_copy]
        if _check_girth(adj_list, total_vertices, g):
            edges = []
            for i in range(total_vertices):
                for j in adj_copy[i]:
                    if i < j:
                        edges.append((i, j))
            return total_vertices, edges
    
    return None, None

def _lazebnik_construct(k, g):
    """LUW graph construction - fast, always valid for prime k or k=4."""
    q = k
    # d such that girth >= d + 5 (for odd d) or d + 4 (for even d)
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
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

def _greedy_search(k, g, time_budget=60.0):
    """Greedy randomized search for a smaller k-regular graph with girth >= g.
    Uses a girth-aware BFS to pick stub pairs that avoid short cycles."""
    start_time = time.time()
    best_n = None
    best_edges = None
    
    lower = _moore_lower(k, g)
    # Try a range of n values, starting from lower bound
    max_target = min(lower + 200, 600)  # cap to keep search tractable
    for offset in range(0, 300):
        if time.time() - start_time > time_budget:
            break
        target_n = lower + offset
        if target_n < k + 1:
            continue
        if k * target_n % 2 != 0:
            continue
        if target_n > max_target:
            break
        
        # Try multiple random initializations with girth-aware configuration
        attempts = 8 if target_n <= lower + 50 else 4
        for attempt in range(attempts):
            if time.time() - start_time > time_budget:
                break
            
            adj = _random_k_regular_girth_aware(target_n, k, g)
            if adj is None:
                continue
            
            # Verify girth
            if _check_girth(adj, target_n, g):
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if best_n is None or target_n < best_n:
                    best_n = target_n
                    best_edges = edges
                break  # Found valid for this target_n
    
    if best_n is not None:
        return best_n, best_edges
    return None, None

def _random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model + simplification."""
    if n < k + 1 or k * n % 2 != 0:
        return None
    if k >= n:
        return None
    
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    # Randomly pair stubs
    for _ in range(100):  # try 100 times to get a simple graph
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            # Convert to lists
            return [list(s) for s in adj]
    return None

def _random_k_regular_girth_aware(n, k, g, max_attempts=30):
    """Generate a random k-regular graph with girth awareness using BFS-based stub pairing.
    At each step, pick a stub, then choose a partner stub that is far enough (BFS distance >= g-2)
    to avoid creating a short cycle."""
    if n < k + 1 or k * n % 2 != 0:
        return None
    if k >= n:
        return None
    
    max_attempts = min(max_attempts, 40)
    
    for attempt in range(max_attempts):
        # Track remaining stubs per vertex
        stubs_per_vertex = [list(range(k)) for _ in range(n)]
        adj = [set() for _ in range(n)]
        
        valid = True
        total_pairs = n * k // 2
        
        for pair_idx in range(total_pairs):
            # Find a vertex with remaining stubs
            candidates = [i for i in range(n) if stubs_per_vertex[i]]
            if not candidates:
                break
            u = random.choice(candidates)
            stub_u = stubs_per_vertex[u].pop()
            
            # BFS from u to find vertices at distance >= g-2 (to ensure cycle >= g)
            # We need: any path from u to v of length L gives cycle of length L+1
            # So we need L+1 >= g, i.e., L >= g-1
            # But we also need to avoid the direct edge creating a 2-cycle (u==v)
            min_dist = g - 2  # if dist(u,v) >= g-2, then cycle length >= g-1... 
            # Actually: path of length d from u to v + new edge = cycle of length d+1
            # We need d+1 >= g, so d >= g-1
            # But for girth check, we need no cycle of length < g
            # So we need d >= g-1 for the path, meaning BFS distance >= g-1
            # However, the BFS distance in the CURRENT graph (without the new edge)
            # If BFS dist(u,v) >= g-1, then adding (u,v) gives cycle >= g
            # But if there's a shorter path we missed... BFS gives shortest, so if BFS dist >= g-1, safe.
            
            # For efficiency, do a limited BFS
            target_dist = g - 1  # need BFS distance >= this
            # Actually we need: shortest path from u to v in current graph >= g-1
            # If no path exists, that's fine too (distance = infinity)
            
            # Collect candidate vertices by BFS depth
            visited = {u: 0}
            frontier = [u]
            eligible = []
            depth = 0
            while frontier and depth < target_dist:
                next_frontier = []
                for node in frontier:
                    for nb in adj[node]:
                        if nb not in visited:
                            visited[nb] = depth + 1
                            next_frontier.append(nb)
                frontier = next_frontier
                depth += 1
            
            # Vertices at distance >= target_dist (or unreachable) are eligible
            # Unreachable vertices: those not in visited
            # Vertices at distance >= target_dist: those in visited with dist >= target_dist
            # But our BFS only went to depth target_dist-1, so visited has dist 0..target_dist-1
            # Eligible: vertices NOT in visited (unreachable) OR in visited with dist >= target_dist
            # Since we stopped at depth target_dist-1, all visited have dist < target_dist
            # So eligible = vertices not in visited AND != u, with remaining stubs
            
            eligible_vertices = [i for i in range(n) if i != u and i not in visited and stubs_per_vertex[i]]
            
            if not eligible_vertices:
                # Fallback: try any vertex with remaining stubs (may create short cycle)
                eligible_vertices = [i for i in range(n) if i != u and stubs_per_vertex[i]]
                if not eligible_vertices:
                    valid = False
                    break
            
            v = random.choice(eligible_vertices)
            stub_v = stubs_per_vertex[v].pop()
            
            if v in adj[u]:
                valid = False
                break
            
            adj[u].add(v)
            adj[v].add(u)
        
        if valid:
            adj_list = [list(s) for s in adj]
            if _check_girth(adj_list, n, g):
                return adj_list
    return None

def _would_create_short_cycle(adj, n, u, v, g):
    """Check if adding edge (u,v) to current adj creates a cycle of length < g."""
    # BFS from u to v, avoiding the edge (u,v) itself
    # If we find a path of length < g-1, then adding (u,v) creates a cycle of length <= g-1 < g
    max_path_len = g - 2  # path length + 1 for the new edge < g
    if max_path_len < 1:
        return False
    visited = {u: 0}
    queue = deque([(u, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist >= max_path_len:
            continue
        for nb in adj[node]:
            if nb == v and dist + 1 <= max_path_len:
                # Found path from u to v of length dist+1
                # Cycle length = dist + 1 + 1 = dist + 2
                if dist + 2 < g:
                    return True
            if nb not in visited:
                visited[nb] = dist + 1
                queue.append((nb, dist + 1))
    return False

def _repair_girth(adj, n, k, g, elapsed, budget):
    """Try to repair a k-regular graph to have girth >= g by swapping edges.
    Uses 2-switch operations to eliminate short cycles."""
    if elapsed > budget:
        return None
    remaining = budget - elapsed
    max_iters = min(2000, int(remaining * 20))
    
    adj = [list(s) for s in adj]  # copy
    # Convert to sets for faster lookup
    adj_sets = [set(s) for s in adj]
    
    for _ in range(max_iters):
        if time.time() - (budget - remaining) > budget:
            break
        # Find a short cycle
        cycle = _find_short_cycle(adj, n, g)
        if cycle is None:
            return [list(s) for s in adj_sets]
        # Try to fix by 2-switch: remove edges (u,v) and (w,x), add (u,w) and (v,x)
        # Pick two edges from the cycle that are non-adjacent
        if len(cycle) < 4:
            # Cycle too short, try random 2-switch
            break
        # Pick random pair of non-adjacent edges in cycle
        edge_pairs = []
        for i in range(len(cycle)):
            for j in range(i + 2, len(cycle)):
                if j - i == 1 or (i == 0 and j == len(cycle) - 1):
                    continue
                edge_pairs.append((i, j))
        if not edge_pairs:
            break
        i, j = random.choice(edge_pairs)
        u1, v1 = cycle[i], cycle[(i + 1) % len(cycle)]
        u2, v2 = cycle[j], cycle[(j + 1) % len(cycle)]
        # Remove edges (u1,v1) and (u2,v2)
        adj_sets[u1].discard(v1)
        adj_sets[v1].discard(u1)
        adj_sets[u2].discard(v2)
        adj_sets[v2].discard(u2)
        # Try adding (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)
        found = False
        for pairing in [((u1, u2), (v1, v2)), ((u1, v2), (v1, u2))]:
            (a, b), (c, d) = pairing
            if a == b or c == d:
                continue
            if b in adj_sets[a] or d in adj_sets[c]:
                continue
            # Tentatively add
            adj_sets[a].add(b)
            adj_sets[c].add(d)
            adj_sets[b].add(a)
            adj_sets[d].add(c)
            # Check degrees are still k
            if all(len(adj_sets[x]) == k for x in range(n)):
                # Check girth
                adj_list = [list(s) for s in adj_sets]
                if _check_girth(adj_list, n, g):
                    found = True
                    break
            # Revert
            adj_sets[a].discard(b)
            adj_sets[c].discard(d)
            adj_sets[b].discard(a)
            adj_sets[d].discard(c)
        if not found:
            # Revert removals
            adj_sets[u1].add(v1)
            adj_sets[v1].add(u1)
            adj_sets[u2].add(v2)
            adj_sets[v2].add(u2)
    
    if _check_girth([list(s) for s in adj_sets], n, g):
        return [list(s) for s in adj_sets]
    return None

def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g if one exists. Returns list of vertices or None."""
    max_depth = (g - 1) // 2
    for start in range(n):
        # BFS with parent tracking
        parent = {start: None}
        depth = {start: 0}
        queue = deque([(start, -1)])
        while queue:
            node, prev = queue.popleft()
            if depth[node] >= max_depth:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb in parent:
                    # Found a cycle
                    cycle_len = depth[node] + depth[nb] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = [node]
                        cur = node
                        while cur != nb:
                            cur = parent[cur]
                            cycle.append(cur)
                        cur = nb
                        while cur != node:
                            cur = parent[cur]
                            cycle.append(cur)
                        return cycle
                else:
                    parent[nb] = node
                    depth[nb] = depth[node] + 1
                    queue.append((nb, node))
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g, minimizing n."""
    start_time = time.time()
    
    # Always get LUW as baseline
    luw_n, luw_edges = _lazebnik_construct(k, g)
    best_n, best_edges = luw_n, luw_edges
    
    # Try projective plane construction
    if time.time() - start_time < _DEADLINE - 10:
        pp_n, pp_edges = _projective_plane_construct(k, g)
        if pp_n is not None and pp_n < best_n:
            # Verify
            adj_check = [[] for _ in range(pp_n)]
            for u, v in pp_edges:
                adj_check[u].append(v)
                adj_check[v].append(u)
            if _is_valid(adj_check, pp_n, k, g):
                best_n, best_edges = pp_n, pp_edges
    
    # Try greedy search with remaining time
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 6  # 6s safety margin
    if remaining > 3:
        greedy_n, greedy_edges = _greedy_search(k, g, time_budget=remaining)
        if greedy_n is not None and greedy_n < best_n:
            best_n, best_edges = greedy_n, greedy_edges
    
    # Final safety check: verify the result is valid
    if time.time() - start_time < _DEADLINE - 2:
        adj_check = [[] for _ in range(best_n)]
        for u, v in best_edges:
            adj_check[u].append(v)
            adj_check[v].append(u)
        if not _is_valid(adj_check, best_n, k, g):
            # Fallback to LUW
            best_n, best_edges = luw_n, luw_edges
    
    return best_n, best_edges
