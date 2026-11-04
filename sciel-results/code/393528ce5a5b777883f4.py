import time
import random
from collections import defaultdict
from collections import deque
import math

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Structural Change: Voltage graph lifting over K_{3,3} (bipartite) with 
    Simulated Annealing to optimize voltage assignment for girth.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    if k != 3:
        # Fallback for other k, keep simple LUW
        return _construct_luw(k, g, d=7)
    
    # Primary approach: Voltage graph over K_{3,3}
    # K_{3,3} is 3-regular and bipartite.
    # Vertices 0..5. Edges: (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)
    # This base graph has girth 4.
    
    # Precompute reduced closed walks in K_{3,3} up to length g-1
    # Since K_{3,3} is bipartite, only even lengths matter.
    # We need to check lengths 4, 6, ..., g-2 (if g is even) or g-1 (if g is odd, but g=16 is even)
    
    base_adj = {
        0: [3, 4, 5],
        1: [3, 4, 5],
        2: [3, 4, 5],
        3: [0, 1, 2],
        4: [0, 1, 2],
        5: [0, 1, 2]
    }
    
    # Edge indices for voltage assignment
    # We assign a voltage to each edge. Since the graph is undirected, 
    # we can orient edges arbitrarily and use -voltage for the reverse direction.
    # Let's define directed edges with consistent orientation for walk sums.
    # Actually, for a closed walk, the sum of voltages must be 0.
    # If we traverse an edge in one direction, we add its voltage. In reverse, we subtract.
    
    # Let's list all edges with an index
    edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i # We need to know direction for sign
    
    # To handle direction in sum:
    # If we go u->v where (u,v) is in edges, add voltage[i].
    # If we go v->u, subtract voltage[i].
    
    # Precompute all reduced closed walks of length L < g
    # A reduced walk does not traverse the same edge in both directions consecutively.
    # In a simple graph, reduced means no immediate backtracking.
    
    closed_walks = []
    max_len = g - 1
    
    # Generate walks
    # Start from each vertex
    for start in range(6):
        stack = [(start, -1, 0, [])] # (current_node, prev_edge_idx, length, path_of_edge_indices_with_sign)
        # Instead of storing path, store the coefficient vector of edge indices
        # But we need to check if sum(v_e * coeff_e) == 0.
        # It's better to store the list of (edge_idx, sign)
        
        while stack:
            v, prev_e, l, coeffs = stack.pop()
            
            if l > 0 and v == start:
                # Found a closed walk
                # Check if it is reduced (no immediate backtracking)
                # The generation method above (avoiding prev_e) ensures no immediate backtracking
                # But we must ensure we don't count the same walk multiple times?
                # For voltage check, we just need the set of distinct coefficient vectors.
                # However, different walks can have the same coefficient vector.
                # We can just store the tuple of coefficients.
                closed_walks.append(coeffs)
            
            if l >= max_len:
                continue
            
            for next_v in base_adj[v]:
                # Find edge index between v and next_v
                # There is exactly one edge in K_{3,3} between any pair in different partitions
                e_idx = edge_index[(v, next_v)]
                
                # Check if this is backtracking
                # Backtracking means next_v is the vertex we came from, and e_idx is the edge we used.
                # In our stack, prev_e is the edge index used to arrive at v.
                # If next_v is the neighbor we came from, we would be using the same edge.
                # Since K_{3,3} is simple, if next_v is the previous node, e_idx == prev_e.
                if e_idx == prev_e:
                    continue
                
                # Determine sign
                # If (v, next_v) is in original edge list orientation, sign is +1
                # Else sign is -1
                # We defined edges as (u,v) with u < v? No, we defined specific tuples.
                # Let's check if (v, next_v) matches the stored direction.
                # We stored (0,3) as idx 0. So (0,3) is +, (3,0) is -.
                # How to check?
                # We can store a set of directed edges for + sign.
                pass # Logic below

    # Better approach for precomputation:
    # Store walks as tuples of (edge_idx, sign)
    # Then convert to coefficient vector.
    
    # Let's redo precomputation cleanly
    closed_walk_coeff_vectors = set()
    
    # Directed edges for sign determination
    # If (u,v) is in the list 'edges', then u->v is +1, v->u is -1.
    # We need a fast lookup.
    pos_dir = set()
    for u, v in edges:
        pos_dir.add((u, v))
        
    def get_sign(u, v):
        if (u, v) in pos_dir:
            return 1
        else:
            return -1
            
    for start in range(6):
        stack = [(start, -1, 0, {})] # (node, prev_edge, len, {edge_idx: sum_of_signs})
        
        while stack:
            v, prev_e, l, coeff_dict = stack.pop()
            
            if l > 0 and v == start:
                # Convert dict to tuple for set
                # Sort by edge index
                vec = tuple(coeff_dict.get(i, 0) for i in range(9))
                # Filter out zero vectors (shouldn't happen for l>0)
                if any(x != 0 for x in vec):
                    closed_walk_coeff_vectors.add(vec)
            
            if l >= max_len:
                continue
            
            for next_v in base_adj[v]:
                e_idx = edge_index[(v, next_v)]
                
                if e_idx == prev_e:
                    continue
                
                sign = get_sign(v, next_v)
                
                new_dict = coeff_dict.copy()
                new_dict[e_idx] = new_dict.get(e_idx, 0) + sign
                # Optimization: if any coefficient becomes 0, we can remove it to keep dict small
                # But for simplicity, we keep it.
                
                stack.append((next_v, e_idx, l + 1, new_dict))

    # Convert to list of tuples for fast access
    coeff_list = list(closed_walk_coeff_vectors)
    
    # Now, search for voltage assignments
    # We want v_e in Z_m such that for all c in coeff_list, sum(c_i * v_i) != 0 mod m
    
    best_n = float('inf')
    best_edges = None
    
    # Try small primes
    # Moore bound for n(3,16) is 512.
    # 6 * m >= 512 => m >= 85.33
    # Primes: 89, 97, 101, 103, 107, 109, 113, 127, 131...
    
    primes = [89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307]
    
    # Simulated Annealing for each prime
    for m in primes:
        if time.time() > deadline:
            break
            
        # Initial random assignment
        voltages = [random.randint(1, m-1) for _ in range(9)]
        
        # Calculate initial cost (number of violated constraints)
        def calc_cost(vents):
            cost = 0
            for c in coeff_list:
                s = sum(c[i] * vents[i] for i in range(9))
                if s % m == 0:
                    cost += 1
            return cost
            
        current_cost = calc_cost(voltages)
        current_voltages = voltages.copy()
        
        # Annealing parameters
        T = 10.0
        T_min = 0.01
        cooling = 0.99
        steps = 2000 # Adjust based on time
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Pick a random edge to change
            idx = random.randint(0, 8)
            old_val = current_voltages[idx]
            new_val = random.randint(1, m-1)
            current_voltages[idx] = new_val
            
            new_cost = calc_cost(current_voltages)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                # Accept
            else:
                current_voltages[idx] = old_val
                
            T *= cooling
            
            if current_cost == 0:
                break
                
            if T < T_min:
                break
        
        if current_cost == 0:
            # Valid assignment found
            n = 6 * m
            edges = []
            for u in range(6):
                for w in base_adj[u]:
                    e_idx = edge_index[(u, w)]
                    vol = current_voltages[e_idx]
                    # Determine sign for this direction
                    sign = get_sign(u, w)
                    # If sign is 1, w_node = (v + vol) % m
                    # If sign is -1, w_node = (v - vol) % m
                    # Actually, let's define the lift carefully.
                    # Vertex in lift: (u, v) where u in 0..5, v in 0..m-1.
                    # Edge between (u, v) and (w, v') if (u,w) is edge in base.
                    # v' = (v + sign * vol) % m.
                    
                    for v in range(m):
                        u_node = v * 6 + u
                        target_v = (v + sign * vol) % m
                        w_node = target_v * 6 + w
                        if u_node < w_node:
                            edges.append((u_node, w_node))
                        else:
                            edges.append((w_node, u_node))
            
            edges = list(set(edges))
            
            # Verify girth just in case (should be fast for n ~ 500-600)
            # Using BFS with depth limit
            if _has_girth_at_least(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    if n <= 936: # Target improvement
                        return n, edges
                        
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to LUW if voltage graph fails
    return _construct_luw(k, g, d=7)

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using BFS with depth limit.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = (g - 1) // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            
            if dist[u] >= max_depth:
                continue
            
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and v != start:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _construct_luw(k, g, d=None, rule_variant=0):
    """
    LUW construction fallback.
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

