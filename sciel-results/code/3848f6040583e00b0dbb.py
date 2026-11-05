import time
import random
from collections import deque, defaultdict
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    """
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin

    if k == 3:
        # Try the new tree-merging approach first
        result = _construct_tree_merge(g, deadline, start_time)
        if result is not None:
            return result
        # Fallback to voltage lift if tree merge fails
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        # Fallback for other k (not primary target)
        return _construct_luw(k, g, d=7)

def _construct_tree_merge(g, deadline, start_time):
    """
    Constraint satisfaction approach: build non-backtracking tree of depth 7,
    then merge boundary vertices using bipartite matching to form a 3-regular graph.
    
    For girth g, we need no cycles shorter than g.
    The non-backtracking tree from a root has depth (g-1)//2 before potential self-intersections.
    For g=16, depth = 7 (since 2*7 = 14 < 16, but 2*8 = 16 would create a cycle of length 16 which is allowed).
    
    Actually, let's think more carefully:
    - A non-backtracking walk of length L from root back to root creates a cycle of length L.
    - We need no cycles of length < g, so no non-backtracking closed walk of length < g.
    - The tree of depth d from root contains all non-backtracking walks of length <= d.
    - If two boundary vertices at depth d are merged, they create a cycle of length 2*d.
    - We need 2*d >= g, so d >= g/2.
    - For g=16, d >= 8. But we also need to ensure no shorter cycles form.
    
    Let me reconsider: The approach is to build a tree of depth d where d = g//2 - 1 or similar,
    then match boundary vertices. The key constraint is that matching two vertices at depths
    d1 and d2 creates a cycle of length d1 + d2 + 2 (the path from root to each plus the matching edge).
    Actually, if we match vertex v at depth d_v and vertex w at depth d_w, the cycle length is
    d_v + d_w + 2 (going from v up to their LCA, which is root if depths are equal, then down to w, plus the matching edge... no).
    
    Let me use a simpler approach: build the full non-backtracking tree of depth d where 2*d >= g,
    then find a perfect matching on the boundary that doesn't create any cycle shorter than g.
    
    For a 3-regular graph with girth g, the Moore bound gives n >= 2 * (3^(g/2 - 1) - 1) + 2 = 2 * 3^(g/2 - 1).
    For g=16: n >= 2 * 3^7 = 4374. That's too large.
    
    The Moore bound for 3-regular girth 16 is actually:
    n >= 2 * (3^(g/2 - 1) - 1) + 2 for even g
    = 2 * (3^7 - 1) + 2 = 2 * 2186 + 2 = 4374
    
    But the best known is 936, so the Moore bound is loose. Let me use a different strategy.
    
    Actually, the Moore bound formula for k-regular girth g:
    If g is even: n >= k * (k-1)^(g/2 - 1) / 2 + ... 
    Let me just use: n >= 2 * (k-1)^(g/2 - 1) for k=3, g=16: 2 * 2^7 = 256.
    
    Wait, the standard Moore bound for (k,g):
    - If g is even: n >= k * sum_{i=0}^{g/2-1} (k-1)^i = k * ((k-1)^(g/2) - 1) / (k-2) for k>2
    - For k=3, g=16: n >= 3 * (2^8 - 1) / 1 = 3 * 255 = 765
    
    Hmm, that doesn't match the stated lower bound of 512. Let me just proceed with the algorithm
    and see what n we get.
    
    Strategy:
    1. Build non-backtracking tree of depth d = (g-1)//2 from a root.
    2. The boundary vertices (at depth d) need to be matched among themselves or with other boundary vertices
       from other "roots" to form a 3-regular graph.
    3. Actually, for a single tree, each internal vertex has degree 2 (parent + children) except root has degree 3 (children only, but we need 3 children for 3-regular).
    
    Let me try a different approach: 
    - Start with multiple copies of the tree
    - Match boundary vertices between copies to form a 3-regular graph
    - Check that no cycle shorter than g is formed
    """
    if g < 4:
        return None
    
    # For 3-regular graph with girth g:
    # Build a "skeleton" tree of depth d where d = g // 2 - 1
    # Each leaf at depth d needs to be connected to form 3-regularity
    # 
    # Actually, let's use the following approach:
    # 1. Generate a random 3-regular graph candidate using a tree-based construction
    # 2. Check girth
    # 3. If girth is too small, modify edges
    #
    # This is essentially a local search / simulated annealing approach on 3-regular graphs.
    
    # Let's try: start with a random 3-regular graph, then use edge swaps to increase girth
    n_target = 2 * (3 ** ((g - 2) // 2))  # Rough estimate, will adjust
    
    # For g=16: n_target = 2 * 3^7 = 4374, too large
    # Let's use a smaller n and try to find one with girth >= g
    # Start with n = 2 * 3^((g-2)//2 - 1) or similar
    
    # Actually, let's just try various n values and use simulated annealing
    # to find a 3-regular graph with girth >= g
    
    best_result = None
    best_n = float('inf')
    
    # Try n values from Moore bound estimate upward
    # Moore bound for k=3, g=16: n >= 2 * (3^(g/2-1) - 1) + 2 = 2*(3^7-1)+2 = 4374
    # But best known is 936, so let's start lower
    # Let's try n = 2 * 3^m for various m
    
    for m in range(3, 9):  # 3^3=27 to 3^8=6561
        n = 2 * (3 ** m)
        if n > 20000:
            break
        if time.time() > deadline:
            break
        
        result = _sa_3regular_girth(n, g, deadline, start_time)
        if result is not None:
            n_found, edges = result
            if n_found < best_n:
                best_n = n_found
                best_result = (n_found, edges)
    
    return best_result

def _sa_3regular_girth(n, g, deadline, start_time):
    """
    Simulated annealing to find a 3-regular graph on n vertices with girth >= g.
    """
    if time.time() > deadline:
        return None
    
    # Generate random 3-regular graph
    edges = _random_3regular(n)
    if edges is None:
        return None
    
    # Compute current girth
    current_girth = _compute_girth(n, edges)
    
    if current_girth >= g:
        return (n, edges)
    
    # Simulated annealing: try edge swaps to increase girth
    T_start = 5.0
    T_end = 0.01
    alpha = 0.995
    max_iters = 5000
    
    best_girth = current_girth
    best_edges = edges[:]
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick two random edges to swap
        e1 = random.randint(0, len(edges) - 1)
        e2 = random.randint(0, len(edges) - 1)
        if e1 == e2:
            continue
        
        u1, v1 = edges[e1]
        u2, v2 = edges[e2]
        
        # Try swapping: (u1,v1) and (u2,v2) -> (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)
        new_edges1 = (u1, u2)
        new_edges2 = (v1, v2)
        
        # Check validity: no loops, no duplicate edges
        if u1 == u2 or v1 == v2:
            continue
        
        edge_set = set()
        for a, b in edges:
            if a < b:
                edge_set.add((a, b))
            else:
                edge_set.add((b, a))
        
        ne1 = (min(new_edges1[0], new_edges1[1]), max(new_edges1[0], new_edges1[1]))
        ne2 = (min(new_edges2[0], new_edges2[1]), max(new_edges2[0], new_edges2[1]))
        
        if ne1 in edge_set or ne2 in edge_set or ne1 == ne2:
            continue
        
        # Perform swap
        edges[e1] = new_edges1
        edges[e2] = new_edges2
        
        # Compute new girth
        new_girth = _compute_girth(n, edges)
        
        delta = new_girth - current_girth
        
        if delta > 0:
            current_girth = new_girth
            if new_girth > best_girth:
                best_girth = new_girth
                best_edges = edges[:]
        else:
            # Accept with probability exp(delta / T)
            T = T_start * (alpha ** i)
            if random.random() < np.exp(delta / T):
                current_girth = new_girth
            else:
                # Revert
                edges[e1] = (u1, v1)
                edges[e2] = (u2, v2)
        
        if current_girth >= g:
            return (n, edges)
    
    # Return best found if it meets girth requirement
    if best_girth >= g:
        return (n, best_edges)
    
    return None

def _random_3regular(n):
    """Generate a random 3-regular graph on n vertices. Returns list of edges or None if impossible."""
    if n < 4 or n % 2 != 0:
        return None
    
    # Use configuration model
    vertices = list(range(n))
    half_edges = []
    for v in vertices:
        for _ in range(3):
            half_edges.append(v)
    
    random.shuffle(half_edges)
    
    edges = []
    edge_set = set()
    
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v:
            return None  # Loop
        if (u, v) in edge_set or (v, u) in edge_set:
            return None  # Multi-edge
        edge_set.add((u, v))
        edges.append((u, v))
    
    return edges

def _compute_girth(n, edges):
    """Compute the girth (shortest cycle length) of a graph."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS to find shortest cycle through it
    min_girth = float('inf')
    
    for start in range(n):
        # BFS with parent tracking
        dist = [-1] * n
        parent = [-1] * n
        queue = deque()
        
        for nb in adj[start]:
            dist[nb] = 1
            parent[nb] = start
            queue.append(nb)
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    
    return min_girth if min_girth != float('inf') else float('inf')

def _construct_petersen_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Uses Simulated Annealing to find valid voltages.
    """
    # Petersen graph edges (10 vertices, 15 edges)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Spokes
    ]
    
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    # We need to find constraints on voltages such that no such walk has total voltage 0 mod m
    coeff_vectors = set()
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        # DFS to find all non-backtracking closed walks of length < g
        # State: (current_v, prev_v, length, counts)
        stack = [(start_v, -1, 0, [0]*15)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(n_base):
                if next_v == prev_v:
                    continue # Non-backtracking
                if next_v == v:
                    continue # No loops
                
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                
                new_counts = counts.copy()
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    # Found a non-backtracking closed walk
                    coeff_vectors.add(tuple(new_counts))
                
                stack.append((next_v, v, new_l, new_counts))
    
    # Normalize constraints to remove redundancy (c and -c are equivalent)
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        # Find first non-zero
        for i in range(15):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    
    # Precompute coefficient matrix for faster evaluation
    # C is a numpy array of shape (num_constraints, 15)
    if not coeff_list:
        # If no constraints, any voltage works, but we need m >= 2
        m = 2
        voltages = [1] * 15
        n_candidate = 10 * m
        edges = _build_lifted_graph(pet_edges, voltages, m)
        return n_candidate, edges

    C = np.array(coeff_list, dtype=np.int64)
    num_constraints = len(C)
    
    # Search for m and voltages
    best_n = float('inf')
    best_edges = None
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = []
    candidate = 2
    while len(primes) < 200:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    for m in primes:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        # Simulated Annealing
        # Energy: number of constraints violated (sum % m == 0)
        # We want to minimize energy to 0.
        
        def calculate_energy(voltages_np):
            # voltages_np is shape (15,)
            # C is shape (num_constraints, 15)
            # sums = C @ voltages_np
            sums = C @ voltages_np
            # Check which sums are 0 mod m
            # Note: numpy modulo can be negative? No, Python's % is always non-negative for positive modulus.
            # Numpy's % follows C standard, which for positive modulus is non-negative.
            violated = (sums % m == 0)
            return np.sum(violated)
        
        found = False
        # Parameters for SA
        T_start = 10.0
        T_end = 0.01
        alpha = 0.99
        max_iters = 5000
        restarts = 5
        
        for restart in range(restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            voltages = [random.randint(1, m-1) for _ in range(15)]
            v_np = np.array(voltages, dtype=np.int64)
            current_energy = calculate_energy(v_np)
            
            if current_energy == 0:
                found = True
                break
                
            T = T_start
            
            for i in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to change
                j = random.randint(0, 14)
                old_val = v_np[j]
                new_val = random.randint(1, m-1)
                if new_val == old_val:
                    continue
                    
                v_np[j] = new_val
                new_energy = calculate_energy(v_np)
                
                delta_e = new_energy - current_energy
                
                if delta_e < 0:
                    # Accept
                    current_energy = new_energy
                else:
                    # Accept with probability exp(-delta_e / T)
                    if random.random() < np.exp(-delta_e / T):
                        current_energy = new_energy
                    else:
                        # Reject
                        v_np[j] = old_val
                        
                T *= alpha
                
                if current_energy == 0:
                    found = True
                    break
                if T < T_end:
                    break
            
            if found:
                break
        
        if found:
            # Final verification
            voltages_final = v_np.tolist()
            edges = _build_lifted_graph(pet_edges, voltages_final, m)
            if _has_girth_at_least(n_candidate, edges, g):
                return n_candidate, edges
            # If verification fails, it's a bug, but we assume SA found a valid energy=0 state
            # which implies girth >= g theoretically.
            # Let's just return it if energy is 0, trusting the math.
            # But to be safe, we already checked _has_girth_at_least.
            # If it fails, we continue searching.
            
    # Fallback
    return _construct_luw(k=3, g=g, d=11)

def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edges = []
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue
            
            vol = voltages[e_idx]
            u_edge, v_edge = pet_edges[e_idx]
            if u == u_edge and v == v_edge:
                sign = 1
            else:
                sign = -1
            
            for layer in range(m):
                u_node = layer * n_base + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * n_base + v
                
                if u_node < v_node:
                    edges.append((u_node, v_node))
                else:
                    edges.append((v_node, u_node))
                
    edges = list(set(edges))
    return edges

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
    For k=3, uses q=3 arithmetic.
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
    
    # Build rule table with variant-dependent offsets
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
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
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    A graph has girth >= g iff there are no non-backtracking closed walks of length < g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Non-backtracking closed walks of length < g
    max_len = g - 1
    
    for start in range(n):
        # State: (current_vertex, prev_vertex, length)
        # Start with all neighbors of start as length-1 walks
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start and l > 0:
                # Found a non-backtracking closed walk of length l < g
                return False
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    
    return True
