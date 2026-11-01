"""
Constructs a 3-regular graph with girth at least g using voltage lifts of the Petersen graph.
"""
import time
import random
from collections import deque
import itertools

def _verify_girth_fast(adj, n, g):
    """
    Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    # We check for cycles of length < g.
    # A non-backtracking walk of length L starting at s and ending at s (with L >= 3)
    # implies a cycle of length <= L.
    # We perform BFS on the state space (v, prev).
    # State (v, prev) means we are at v, came from prev.
    # Distance is the length of the walk.
    
    for start in range(n):
        # dist[state] = length of walk
        # state is (v, prev)
        dist = {}
        dq = deque()
        # Initial states: (start, -1) is not a walk, it's the start.
        # From start, we can go to any neighbor w.
        # State (w, start) has length 1.
        for w in adj[start]:
            state = (w, start)
            dist[state] = 1
            dq.append(state)
        
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            
            # If we reach start again, we found a cycle.
            # The walk was start -> ... -> v -> start?
            # No, the state is (v, prev).
            # If v == start, the walk length is d.
            # But we started with (w, start) of length 1.
            # If we see (start, u) with length d, it means a walk of length d+1?
            # Let's trace:
            # (w, start) len 1.
            # Next is (x, w) len 2.
            # If x == start, state is (start, w) len 2.
            # This corresponds to walk start -> w -> start. Length 2.
            # But simple graphs have no cycles of length 2.
            # So we should not count this.
            # We only care about cycles of length >= 3.
            
            if v == start:
                # The walk length is d.
                # But this state (start, prev) was reached from (prev, start)?
                # No, (start, prev) means we are at start, came from prev.
                # The previous state was (prev, something).
                # Actually, if v == start, the walk is start -> ... -> prev -> start.
                # Length is d.
                # If d < g, we have a cycle of length d (or less).
                # However, if d == 2, it's a backtrack.
                # In a simple graph, we don't have 2-cycles.
                # So if d >= 3 and d < g, return False.
                if d < g and d >= 3:
                    return False
                # If d == 1, impossible (no self-loops).
                # If d == 2, backtrack. Ignore.
                continue

            if d >= g - 1:
                # No need to expand further, any cycle found will be >= g
                continue
            
            for w in adj[v]:
                if w == prev:
                    continue
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _build_lift(base_edges, m, shifts):
    """
    Build a voltage graph lift over Z_m.
    base_edges: list of (u,v) on the base graph
    m: lift size
    shifts: list of voltage shifts, one per base edge
    Returns (n, adj_list, edge_list) or None if invalid.
    """
    n_base = max(max(u for u,v in base_edges), max(v for u,v in base_edges)) + 1
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    
    # Map edge index to shift
    # base_edges must correspond to shifts order
    
    for idx, (u, v) in enumerate(base_edges):
        vol = shifts[idx] % m
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
            else:
                # Multiple edge?
                # This can happen if the base graph has multiple edges or if shifts cause collisions.
                # Petersen is simple, so no multiple edges in base.
                # Collision in lift: (u, i) -- (v, i+vol) and (u, i') -- (v, i'+vol')
                # If u=u', i=i', then v=v', i+vol = i'+vol' => vol=vol'.
                # So no multiple edges unless base has multiple edges.
                pass

    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    adj_list = [list(s) for s in adj]
    return n, adj_list, edges

def _get_petersen_walks():
    """
    Precompute all closed non-backtracking walks of length < 16 in the Petersen graph.
    Returns a list of vectors c_W (length 15) representing edge counts.
    """
    # Petersen edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    walks = []
    max_len = 15
    
    # Start from each edge
    for start_u, start_v in edges:
        # Initial state: at start_v, came from start_u, via edge 0 (index in edges list? No, we need consistent indexing)
        # Let's use edge indices 0..14.
        # Start walk: edge e0.
        # State: (current_node, prev_node, edge_idx_list)
        # We want to return to start_u.
        
        # We'll do a DFS/BFS up to depth 15.
        # State: (node, prev, path_edges)
        # To avoid explosion, we prune if we can't return to start?
        # Distance in base graph is small.
        
        def dfs(node, prev, path, depth):
            if depth == 0:
                return
            if node == start_u and depth > 0:
                # Found a closed walk of length (max_len - depth + 1)?
                # No, path contains the edges taken.
                # Length is len(path).
                if 3 <= len(path) <= max_len:
                    # Count edges
                    c = [0] * 15
                    for e in path:
                        c[e] += 1
                    # Only add if not already in walks? 
                    # Different walks can have same edge counts.
                    # We just need the set of all possible counts.
                    walks.append(tuple(c))
            
            if depth >= max_len:
                return
            
            for next_node, e_idx in adj[node]:
                if next_node == prev:
                    continue
                dfs(next_node, node, path + [e_idx], depth - 1)
        
        # Start: we took edge e0 to get to start_v.
        # Path so far: [e0].
        # Current node: start_v. Prev: start_u.
        # We need to return to start_u.
        dfs(start_v, start_u, [edges.index((start_u, start_v)) if (start_u, start_v) in edges else edges.index((start_v, start_u))], max_len - 1)
        
    # Remove duplicates
    return list(set(walks))

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    if k != 3:
        # Fallback for other k
        # LUW construction
        return _luw_construct(k, g)

    # Petersen graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    e_count = 15
    
    # Precompute walks
    walks = _get_petersen_walks()
    
    # Search for m
    rng = random.Random(42)
    
    # Try m values
    # n = 10m. Moore bound 510 -> m >= 51.
    # Best known 936 -> m ~ 94.
    
    for m in range(52, 200):
        if time.time() > deadline - 5:
            break
            
        n = 10 * m
        
        # Randomized Local Search for shifts
        best_shifts = None
        best_cost = float('inf')
        
        # Initialize with random shifts
        for restart in range(50):
            if time.time() > deadline - 2:
                break
                
            shifts = [rng.randint(0, m - 1) for _ in range(e_count)]
            
            # Evaluate cost
            cost = 0
            for c in walks:
                s = sum(c[i] * shifts[i] for i in range(e_count)) % m
                if s == 0:
                    cost += 1
                    
            if cost == 0:
                best_shifts = shifts
                break
                
            # Local Search
            current_shifts = shifts[:]
            current_cost = cost
            T = 1.0
            for step in range(1000):
                if time.time() > deadline - 1:
                    break
                    
                i = rng.randint(0, e_count - 1)
                old_val = current_shifts[i]
                new_val = rng.randint(0, m - 1)
                if old_val == new_val:
                    continue
                    
                current_shifts[i] = new_val
                
                # Recalculate cost (expensive if done naively, but walks list is small ~1000?)
                # Number of walks is small.
                new_cost = 0
                for c in walks:
                    s = sum(c[j] * current_shifts[j] for j in range(e_count)) % m
                    if s == 0:
                        new_cost += 1
                
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < (2.718 ** (-delta / T)):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_shifts = current_shifts[:]
                        if new_cost == 0:
                            break
                else:
                    current_shifts[i] = old_val
                T *= 0.99
                
            if best_cost == 0:
                break
                
        if best_cost == 0:
            result = _build_lift(petersen_edges, m, best_shifts)
            if result is not None:
                n_r, adj, edges = result
                # Verify to be safe
                if _verify_girth_fast(adj, n_r, g):
                    return n_r, edges

    # Fallback to LUW
    if time.time() < deadline - 10:
        try:
            n_luw, edges_luw = _luw_construct(3, g)
            return n_luw, edges_luw
        except:
            pass

    # Last resort: return a large LUW graph
    return _luw_construct(3, g)

def _luw_construct(k, g, d_override=None):
    """
    LUW construction for k=3.
    """
    q = k
    if d_override is not None:
        d = d_override
    else:
        # Ensure girth >= g
        # Girth of LUW is at least 2d+1
        # So 2d+1 >= g => d >= (g-1)/2
        d = (g - 1 + 1) // 2  # ceil((g-1)/2)
        if d < 3:
            d = 3
            
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
        
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
            
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
