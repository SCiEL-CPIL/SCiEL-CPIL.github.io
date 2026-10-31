"""
Construct a k-regular graph with girth at least g.
Primary: systematic circulant search C_n(1,a,b) with algebraic pre-filters.
Secondary: voltage lift over Z_m with exhaustive small-m search.
Fallback: LUW construction.
"""
import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Systematic circulant C_n(1,a,b) search
        # This is vertex-transitive so girth check from one vertex suffices.
        # Algebraic 4-cycle, 6-cycle, and 8-cycle filters eliminate most candidates fast.
        result = _find_circulant_systematic(3, 16, deadline)
        if result is not None:
            return result

        # Strategy 2: Voltage lift over Z_m with optimized search
        if not check_deadline():
            result2 = _voltage_lift_search(3, 16, deadline)
            if result2 is not None:
                return result2

        # Strategy 3: LUW fallback
        for d in [9, 11, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)


def _find_circulant_systematic(k, g, deadline):
    """
    Systematic search for C_n(1,a,b) with girth >= g.
    Vertex-transitive: check girth from vertex 0 only.
    Algebraic pre-filters for 4-cycles, 6-cycles, and 8-cycles.
    """
    min_n = 512
    max_n = 940

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 3:
            return None

        half = n // 2
        # Try a near n/3 first, but also spread out
        a_candidates = list(range(max(3, n // 5), half))
        a_candidates.sort(key=lambda a: abs(a - n // 3))

        for a in a_candidates:
            if time.time() > deadline - 3:
                return None

            # Quick filter: 2a mod n should not be 0, 1, n-1
            two_a = (2 * a) % n
            if two_a in (0, 1, n - 1):
                continue
            # Also check 3a mod n
            three_a = (3 * a) % n
            if three_a in (0, 1, n - 1):
                continue

            b_start = a + 1
            b_end = half

            for b in range(b_start, b_end):
                if time.time() > deadline - 3:
                    return None

                # Algebraic 4-cycle check
                if _has_4cycle_fast(n, a, b):
                    continue

                # Algebraic 6-cycle check
                if _has_6cycle_fast(n, a, b):
                    continue

                # Algebraic 8-cycle check
                if _has_8cycle_fast(n, a, b):
                    continue

                # Full girth verification via non-backtracking BFS from vertex 0
                if _girth_check_circ(n, a, b, g):
                    # Build edges
                    edges = []
                    for i in range(n):
                        for shift in (1, a, b):
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return None


def _has_4cycle_fast(n, a, b):
    """
    Check if C_n(1,a,b) has a 4-cycle.
    A 4-cycle exists iff there exist steps s1,s2,s3,s4 from {±1,±a,±b}
    summing to 0 mod n, with no immediate backtracking, and 4 distinct vertices.
    
    Precompute the set of generators and check all 4-combinations.
    """
    # Generators: ±1, ±a, ±b mod n
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    
    # For each ordered 4-tuple of generators (with no immediate backtracking),
    # check if sum ≡ 0 mod n and vertices are distinct.
    # To be fast, precompute all pairs (g1, g2) where g2 != -g1 mod n
    # Then for each such pair, check if -(g1+g2) can be written as g3+g4
    # with g4 != -g3.
    
    # Build lookup: for each value v mod n, which generators equal v?
    # Actually, just iterate: 6^4 = 1296 checks, each O(1). Fast enough.
    
    for i in range(6):
        g1 = gens[i]
        for j in range(6):
            g2 = gens[j]
            if g2 == (n - g1) % n:
                continue  # backtrack
            s12 = (g1 + g2) % n
            for k in range(6):
                g3 = gens[k]
                if g3 == (n - g2) % n:
                    continue  # backtrack
                s123 = (s12 + g3) % n
                g4 = (n - s123) % n
                # Check if g4 is in gens and g4 != -g3
                if g4 == (n - g3) % n:
                    continue  # backtrack
                # Check if g4 is actually one of the generators
                if g4 in gens:
                    # Verify 4 distinct vertices: 0, g1, g1+g2, g1+g2+g3
                    v1 = g1
                    v2 = s12
                    v3 = s123
                    if v1 != 0 and v2 != 0 and v3 != 0 and v1 != v2 and v2 != v3 and v1 != v3:
                        return True
    return False


def _has_6cycle_fast(n, a, b):
    """
    Check if C_n(1,a,b) has a 6-cycle.
    A 6-cycle through 0: 0 -> v1 -> v2 -> v3 -> v4 -> v5 -> 0
    BFS to depth 3. If two vertices at depth 2 share a common neighbor at depth 3,
    or a vertex at depth 3 has two neighbors at depth 2 with different depth-1 parents,
    we have a 6-cycle (or shorter, which is also bad).
    """
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    
    dist = {0: 0}
    parent = {0: -1}
    level = [0]
    
    for depth in range(3):
        next_level = []
        for u in level:
            for g in gens:
                v = (u + g) % n
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level
    
    # For each vertex at depth 3, check its neighbors at depth 2
    # If two distinct depth-2 neighbors have different depth-1 parents -> 6-cycle
    for v in level:
        depth2_neighbors = []
        for g in gens:
            w = (v + g) % n
            if w in dist and dist[w] == 2:
                depth2_neighbors.append(w)
        
        if len(depth2_neighbors) >= 2:
            # Check if any two have different parents at depth 1
            parents_seen = set()
            for w in depth2_neighbors:
                p = parent[w]
                if p in parents_seen:
                    # Same parent: 0->p->w1->v->w2->p->0 is a 5-cycle
                    return True
                parents_seen.add(p)
            if len(parents_seen) >= 2:
                # Different parents: 0->p1->w1->v->w2->p2->0 is a 6-cycle
                return True
    
    return False


def _has_8cycle_fast(n, a, b):
    """
    Check if C_n(1,a,b) has an 8-cycle.
    BFS to depth 4. If two vertices at depth 4 are adjacent -> 9-cycle (not 8).
    For 8-cycle: vertices at depth 3 sharing a neighbor at depth 4, or
    vertices at depth 4 with a common neighbor at depth 3 (different from parents).
    
    Simpler: check if any vertex at depth 4 has two neighbors at depth 3
    whose depth-2 parents are different -> 8-cycle.
    """
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    
    dist = {0: 0}
    parent = {0: -1}
    level = [0]
    
    for depth in range(4):
        next_level = []
        for u in level:
            for g in gens:
                v = (u + g) % n
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level
    
    # For each vertex at depth 4, check neighbors at depth 3
    for v in level:
        depth3_neighbors = []
        for g in gens:
            w = (v + g) % n
            if w in dist and dist[w] == 3:
                depth3_neighbors.append(w)
        
        if len(depth3_neighbors) >= 2:
            # Check depth-2 parents of these depth-3 neighbors
            parents_seen = set()
            for w in depth3_neighbors:
                p2 = parent[w]  # depth 2
                p1 = parent[p2]  # depth 1
                key = (p2, p1)
                if key in parents_seen:
                    return True
                parents_seen.add(key)
            if len(parents_seen) >= 2:
                return True
    
    return False


def _girth_check_circ(n, a, b, g):
    """
    Non-backtracking BFS from vertex 0 in C_n(1,a,b).
    Returns True if girth >= g (no cycle shorter than g).
    Vertex-transitive so checking from 0 suffices.
    Optimized with arrays instead of dict for speed.
    """
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    
    # State: (current_vertex, prev_vertex) -> encode as single int
    # prev can be -1 (start), so encode prev as prev+1 (0 means no prev)
    # state_id = u * (n+1) + (prev+1)
    max_state = n * (n + 1)
    best_depth = [-1] * max_state
    queue = deque()
    
    start_id = 0 * (n + 1) + 0  # u=0, prev=-1 -> prev+1=0
    best_depth[start_id] = 0
    queue.append((0, -1, 0))  # (u, prev, depth)
    
    max_len = g - 1  # we look for cycles of length < g, i.e., <= g-1
    
    while queue:
        u, prev, d = queue.popleft()
        
        if d >= max_len:
            continue
        
        for g_step in gens:
            v = (u + g_step) % n
            if v == prev:
                continue  # no backtracking
            
            nd = d + 1
            
            if nd > max_len:
                continue
            
            state_id = v * (n + 1) + (u + 1)
            
            if best_depth[state_id] == -1 or nd < best_depth[state_id]:
                best_depth[state_id] = nd
                
                if v == 0 and u != -1:
                    # Found a cycle of length nd
                    if nd < g:
                        return False
                    # else: cycle is long enough, don't expand
                else:
                    queue.append((v, u, nd))
    
    return True


def _voltage_lift_search(k, g, deadline):
    """
    Voltage lift of a small base graph over Z_m.
    Try the 4-vertex cycle (K_{2,2} = C4) as base: lift gives C_{4m} which is bipartite.
    Better: use the dodecahedron (20 vertices, 5-regular) - no, need 3-regular.
    Use the Petersen graph (10 vertices, 3-regular) as base.
    """
    # Petersen graph
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
    
    best_result = None
    best_n = float('inf')
    
    # m from 51 upward (n = 10*m >= 510)
    for m in range(51, 150):
        if time.time() > deadline - 5:
            break
        
        # Try a few structured voltage assignments
        for trial in range(8):
            if time.time() > deadline - 3:
                break
            
            # Generate voltage assignment
            voltages = {}
            for idx, (u, v) in enumerate(petersen_edges):
                if trial == 0:
                    vol = (idx * 7 + 3) % m
                elif trial == 1:
                    vol = (u * 3 + v * 7 + idx) % m
                elif trial == 2:
                    vol = (idx * idx + 2 * idx + 1) % m
                elif trial == 3:
                    vol = (u * v + idx * 3) % m
                elif trial == 4:
                    vol = (idx * 13 + 11) % m
                elif trial == 5:
                    vol = (u * 11 + v * 17 + 1) % m
                elif trial == 6:
                    vol = (idx * 19 + u * 5) % m
                else:
                    # Pseudo-random
                    seed = (idx * 2654435761 + trial * 12345) % (2**31)
                    vol = (seed * 1103515245 + 12345) % (2**31) % m
                
                voltages[(u, v)] = vol
                voltages[(v, u)] = (-vol) % m
            
            # Check girth via non-backtracking BFS on base graph with voltage tracking
            if _lift_girth_check_fast(petersen_adj, voltages, m, g):
                n = 10 * m
                if n < best_n:
                    edges = []
                    for u, v in petersen_edges:
                        vol = voltages[(u, v)]
                        for i in range(m):
                            a = u * m + i
                            b = v * m + ((i + vol) % m)
                            if a < b:
                                edges.append((a, b))
                            else:
                                edges.append((b, a))
                    if len(edges) == 15 * m:
                        best_n = n
                        best_result = (n, edges)
                    break  # Found valid for this m
    
    return best_result


def _lift_girth_check_fast(petersen_adj, voltages, m, g):
    """
    Check if voltage lift has girth >= g.
    A cycle of length L in the lift = closed non-backtracking walk of length L
    in base with total voltage 0 mod m.
    Check all starts (Petersen is vertex-transitive, but voltage breaks symmetry).
    """
    for start in range(10):
        if _lift_bfs_from_fast(start, petersen_adj, voltages, m, g):
            return False
    return True


def _lift_bfs_from_fast(start, petersen_adj, voltages, m, g):
    """
    Non-backtracking BFS from 'start' tracking voltage.
    Returns True if short cycle found.
    """
    max_len = g - 1
    visited = set()
    queue = deque()
    
    start_state = (start, -1, 0)
    visited.add(start_state)
    queue.append((start, -1, 0, 0))
    
    while queue:
        u, prev, vol, depth = queue.popleft()
        
        if depth >= max_len:
            continue
        
        for v in petersen_adj[u]:
            if v == prev:
                continue
            
            new_vol = (vol + voltages[(u, v)]) % m
            new_state = (v, u, new_vol)
            
            if new_state in visited:
                continue
            
            visited.add(new_state)
            new_depth = depth + 1
            
            if v == start and new_vol == 0 and new_depth >= 2:
                if new_depth < g:
                    return True
            
            if new_depth < max_len:
                queue.append((v, u, new_vol, new_depth))
    
    return False


# (removed: _find_circulant_v2, _has_4cycle, _has_6cycle, _full_girth_check_circ
#  replaced by _find_circulant_systematic, _has_4cycle_fast, _has_6cycle_fast, _girth_check_circ)


def _bipartite_cage(k, g, deadline):
    """
    Construct a bipartite k-regular graph with girth >= g.
    For k=3, g=16, bipartite requires n >= 1022, so this is not competitive
    with the non-bipartite Moore bound of 510. Return None.
    """
    return None


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False

    return True


def _bfs_short_cycle(src, adj, n, g):
    """
    BFS from src to find if there's a cycle shorter than g.
    Returns True if a short cycle is found.
    """
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])

    max_depth = g // 2

    while queue:
        u = queue.popleft()
        depth = dist[u]

        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue

        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True

    return False

# EVOLVE-BLOCK-END
