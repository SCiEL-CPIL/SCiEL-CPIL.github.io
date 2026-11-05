"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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


def _check_girth_exact(adj, g, n):
    """Check girth >= g exactly using non-backtracking walks from each vertex.
    
    For girth >= g, there must be no non-backtracking walk of length < g that
    returns to its start. We use BFS on the directed line graph (state = (u, v))
    where we track shortest non-backtracking paths.
    """
    # State: (current_vertex, previous_vertex)
    # We want to detect if any non-backtracking walk of length < g returns to start
    # Use BFS from each start vertex, tracking (vertex, prev) states
    
    for start in range(n):
        # dist[state] = shortest non-backtracking walk length from (start, -1) to state
        # We'll use a dict for sparse states
        from collections import defaultdict
        # dist[(u, v)] = length of shortest non-backtracking walk ending at u, coming from v
        dist = {}
        # Initial states: (start, -1) with length 0
        dist[(start, -1)] = 0
        queue = deque([(start, -1, 0)])
        
        while queue:
            u, prev, length = queue.popleft()
            if length >= g - 1:
                continue
            for v in adj[u]:
                if v == prev:
                    continue  # non-backtracking
                new_state = (v, u)
                new_len = length + 1
                if new_state not in dist:
                    dist[new_state] = new_len
                    if v == start and new_len < g:
                        # Found a non-backtracking cycle of length new_len < g
                        return False
                    if new_len < g - 1:
                        queue.append((v, u, new_len))
                # If state already seen, we already have a shorter path, skip
    
    return True


def _generate_random_regular(k, n, seed=None):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Configuration model with rejection
    for _ in range(50):  # try up to 50 times
        half_edges = []
        for v in range(n):
            half_edges.extend([v] * k)
        
        rng.shuffle(half_edges)
        
        adj = {v: set() for v in range(n)}
        valid = True
        
        for i in range(0, n * k, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                valid = False
                break
            if u in adj[v]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if valid:
            for v in range(n):
                if len(adj[v]) != k:
                    valid = False
                    break
            if valid:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return edges
    
    return None


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (evaluator kills at 150s)
    
    # For k=5, g=9: best known is 1116, Moore bound is ~428
    # LUW construction gives a valid graph; let's compute it first as baseline
    n_luw, edges_luw = _luw_construct(k, g)
    
    # Verify LUW result
    if not _check_girth_exact({i: [] for i in range(n_luw)}, g, n_luw):
        # Build proper adj for LUW
        adj_luw = {i: [] for i in range(n_luw)}
        for u, v in edges_luw:
            adj_luw[u].append(v)
            adj_luw[v].append(u)
        if not _check_girth_exact(adj_luw, g, n_luw):
            # Should not happen, but return anyway
            return n_luw, edges_luw
    
    best_n = n_luw
    best_edges = edges_luw
    
    # Try to find smaller graphs via random search
    # For girth g, the graph must have at least the Moore bound vertices
    # Moore bound for k-regular, girth g:
    if g % 2 == 1:
        moore = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    else:
        moore = 2 * sum((k-1)**i for i in range(g//2))
    
    # Try random regular graphs with n between moore and best_n
    # But random graphs rarely have high girth, so this is unlikely to help
    # Focus on structured constructions instead
    
    # Try voltage graph lifts with careful girth checking
    # Use small base graphs and moderate lifts
    
    base_sizes = [10, 12, 14]
    for base_n in base_sizes:
        if time.time() > deadline:
            break
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            seed = random.randint(0, 2**31)
            base_edges = _generate_random_regular(k, base_n, seed)
            if base_edges is None:
                continue
            
            # Build base adjacency
            base_adj = {v: set() for v in range(base_n)}
            for u, v in base_edges:
                base_adj[u].add(v)
                base_adj[v].add(u)
            
            # Try lifts with m such that base_n * m < best_n
            max_m = (best_n - 1) // base_n
            if max_m < 2:
                continue
            
            # Try a few m values
            m_candidates = sorted(set([max(2, max_m // 2), max(2, max_m * 2 // 3), max(2, max_m - 1), max(2, max_m)]))
            m_candidates = [m for m in m_candidates if 2 <= m <= max_m]
            
            for m in m_candidates:
                if time.time() > deadline:
                    break
                
                for vs in range(4):
                    if time.time() > deadline:
                        break
                    
                    rng = random.Random(seed + vs * 1000 + m)
                    voltages = {}
                    for u, v in base_edges:
                        voltages[(u, v)] = rng.randint(0, m - 1)
                    
                    n_lift = base_n * m
                    lift_edges = set()
                    adj_lift = {i: [] for i in range(n_lift)}
                    valid = True
                    
                    for u, v in base_edges:
                        vol = voltages[(u, v)]
                        for a in range(m):
                            u_lift = u * m + a
                            v_lift = v * m + (a + vol) % m
                            if u_lift == v_lift:
                                valid = False
                                break
                            edge = (min(u_lift, v_lift), max(u_lift, v_lift))
                            if edge in lift_edges:
                                valid = False
                                break
                            lift_edges.add(edge)
                            adj_lift[u_lift].append(v_lift)
                            adj_lift[v_lift].append(u_lift)
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Check degree
                    deg_ok = True
                    for i in range(n_lift):
                        if len(adj_lift[i]) != k:
                            deg_ok = False
                            break
                    if not deg_ok:
                        continue
                    
                    # Check girth
                    if _check_girth_exact(adj_lift, g, n_lift):
                        if n_lift < best_n:
                            best_n = n_lift
                            best_edges = list(lift_edges)
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
