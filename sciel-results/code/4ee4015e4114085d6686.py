import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    # 1. Identify forbidden diagonal elements (2a is a square)
    forbidden_diag = set()
    for a in range(1, N + 1):
        s = 2 * a
        r = int(math.isqrt(s))
        if r * r == s:
            forbidden_diag.add(a)
            
    # Valid vertices
    vertices = [i for i in range(1, N + 1) if i not in forbidden_diag]
    v_index = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    
    # 2. Build conflict graph (u+v is square)
    # Squares up to 256+256=512
    squares = set()
    for k in range(2, 23): # 2^2=4 to 22^2=484. 23^2=529>512
        squares.add(k * k)
        
    # adjacency list for conflict graph
    # adj[i] is set of indices j such that vertices[i] + vertices[j] is a square
    adj_conflict = [set() for _ in range(n)]
    for i in range(n):
        u = vertices[i]
        for j in range(i + 1, n):
            v = vertices[j]
            if (u + v) in squares:
                adj_conflict[i].add(j)
                adj_conflict[j].add(i)
                
    # 3. Build complement graph for Max Clique
    # We want a set of vertices with NO edges in conflict graph.
    # This is a Clique in the complement graph.
    # adj_clique[i] is set of j such that i and j are NOT in conflict (and i!=j)
    # Note: In complement graph, i and j are connected if they CAN coexist.
    adj_clique = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if j not in adj_conflict[i]:
                adj_clique[i].add(j)
                
    # 4. Max Clique Algorithm (Tomita / Branch and Bound)
    # We sort vertices by degree in descending order to improve pruning
    # degrees = [len(adj_clique[i]) for i in range(n)]
    # order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    # remap = {old: new for new, old in enumerate(order)}
    
    # It's often faster to work with the original indices but sorted list for branching
    # Let's use a standard recursive MIS/Clique solver
    
    best_clique = []
    best_size = 0
    
    # Simple Greedy Initialization
    # Pick vertex with max degree, add to clique, remove non-neighbors
    # This is for Clique in G_complement
    def greedy_clique(pivots):
        # pivots is a list of available vertices
        if not pivots:
            return []
        # Pick vertex with max degree in induced subgraph?
        # Or just first one?
        # Let's sort pivots by degree in the induced subgraph
        # For speed, just pick the one with highest global degree among pivots
        v = max(pivots, key=lambda x: len(adj_clique[x]))
        clique = [v]
        # Neighbors of v in clique graph
        neighbors = adj_clique[v]
        # New pivots must be in pivots AND neighbors
        new_pivots = [u for u in pivots if u in neighbors]
        clique.extend(greedy_clique(new_pivots))
        return clique
        
    init_clique = greedy_clique(list(range(n)))
    best_size = len(init_clique)
    best_clique = init_clique
    
    # Branch and Bound
    # We will use a recursive function
    # To avoid recursion depth issues and overhead, we can use a stack or careful recursion
    # Given n ~ 245, recursion depth is bounded by clique size, which is < 245.
    
    # Pre-compute bitmasks? n=245 is too large for standard int bitmasks in Python efficiently if we do bit ops on 245 bits.
    # Python ints are arbitrary precision, so bit ops on 245-bit ints are fast (C level).
    # Let's represent sets of vertices as Python integers (bitmasks).
    
    # Map vertices to 0..n-1 indices
    # adj_mask[i] is bitmask of neighbors in clique graph
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj_clique[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
        
    # Sort vertices by degree descending for better branching
    # We need to remap indices to 0..n-1 based on this order?
    # Or just keep original indices and use a list of available vertices?
    # Bitmasks are faster if we use the original 0..n-1 mapping.
    # Branching: Pick a vertex v from P (available).
    # Include v: New P = P & adj_mask[v].
    # Exclude v: New P = P & ~(1 << v).
    
    # To optimize, we should order the vertices in P.
    # A common heuristic is to pick the vertex with the highest degree in the subgraph induced by P.
    
    def expand(clique, P):
        nonlocal best_size, best_clique
        if not P:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
            
        # Pruning: If len(clique) + bit_count(P) <= best_size, return
        if len(clique) + P.bit_count() <= best_size:
            return
            
        # Pick a vertex v from P
        # Heuristic: Pick v with max degree in P
        # Iterate through bits in P
        # To speed up, we can pre-sort vertices by degree and pick the first one in P that has high degree?
        # Or just pick the first one?
        # Better: Pick the vertex in P with the maximum number of neighbors in P.
        
        # This is O(|P|^2) if done naively.
        # Let's try a simpler heuristic: Pick the vertex in P with the highest global degree.
        # We can pre-sort vertices by degree.
        
        # Let's just iterate through P and pick one.
        # To make it faster, we can use a random order or a fixed sorted order.
        # Let's use a fixed order: vertices sorted by degree descending.
        # We need a list of vertices sorted by degree.
        
        # Actually, let's just pick the lowest index in P? No, that's bad.
        # Let's pick the vertex in P that has the most neighbors in P.
        
        # Optimization: If this is too slow, we can just pick the first bit.
        
        # Let's implement a simple version first and check time.
        
        # Find v in P
        # v = (P & -P).bit_length() - 1  # lowest bit
        # This is fast.
        
        # Let's try picking the vertex with max degree in P.
        # We can approximate this by global degree.
        
        # Let's use a global sorted list of vertices by degree descending.
        # sorted_vertices = sorted(range(n), key=lambda x: adj_mask[x].bit_count(), reverse=True)
        
        # In expand, find the first v in sorted_vertices such that (P & (1<<v)) != 0.
        
        # This requires passing sorted_vertices.
        
        pass # Placeholder
        
    # This recursive approach might be too slow in Python if not careful.
    # Let's use an iterative or optimized recursive one.
    
    # Alternative: Use a library? No, only stdlib/numpy/scipy.
    # Numpy can help with bit operations?
    
    # Let's stick to a well-optimized B&B with bitmasks.
    
    # Pre-sort vertices by degree
    degs = [adj_mask[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda x: degs[x], reverse=True)
    
    # To allow fast "find first available in order", we can't easily do it with bitmasks and a static order list without scanning.
    # Scanning the order list (length 245) at each node is O(n).
    # Number of nodes might be large.
    
    # Let's try a different heuristic:
    # Color the graph to get an upper bound.
    # This is complex to implement correctly and quickly.
    
    # Let's try a simple Greedy + Local Search (Simulated Annealing) which is often faster to implement and robust.
    
    # State: Set of vertices (as a bitmask)
    # Objective: Maximize size, subject to independence.
    # We can define a penalty for violations.
    # Cost = -size + 1000 * violations
    
    def get_violations(mask):
        # Count edges in the conflict graph induced by mask
        # This is O(n^2) if done naively.
        # We can precompute for each vertex the conflict mask.
        # conflicts[i] is bitmask of vertices conflicting with i.
        # For a given mask, for each i in mask, count bits in (mask & conflicts[i]) >> 1?
        # Since conflict is symmetric, each edge is counted twice.
        # Violations = sum( (mask & conflicts[i]).bit_count() for i in bits(mask) ) // 2
        
        count = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            count += (mask & adj_conflict_mask[i]).bit_count()
            m ^= lsb
        return count // 2
        
    # Precompute adj_conflict_mask
    adj_conflict_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj_conflict[i]:
            mask |= (1 << j)
        adj_conflict_mask[i] = mask
            
    # Initial State: Greedy Independent Set
    # Pick vertices with low degree in conflict graph?
    # Or high degree in complement?
    # Let's use the greedy clique result from before as a valid independent set.
    # init_clique is a list of indices.
    init_mask = 0
    for v in init_clique:
        init_mask |= (1 << v)
        
    current_mask = init_mask
    current_size = init_mask.bit_count()
    current_viol = get_violations(current_mask) # Should be 0
    current_cost = -current_size + 1000 * current_viol
    
    best_mask = current_mask
    best_cost = current_cost
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    alpha = 0.99
    iterations = 100000
    
    for it in range(iterations):
        if time.time() > deadline:
            break
            
        if T < T_min:
            break
            
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        bit = 1 << v
        
        new_mask = current_mask ^ bit
        
        # Calculate change in size and violations
        # Size change: +1 if adding, -1 if removing
        if current_mask & bit:
            # Removing v
            size_delta = -1
            # Violations change:
            # v was connected to k neighbors in current_mask.
            # Those edges are removed.
            # New violations = old - k
            # But wait, if we remove v, we don't add any new violations.
            # So viol_delta = -k
            k = (current_mask & adj_conflict_mask[v]).bit_count()
            viol_delta = -k
        else:
            # Adding v
            size_delta = 1
            # v conflicts with k neighbors in current_mask.
            # Those edges are added.
            k = (current_mask & adj_conflict_mask[v]).bit_count()
            viol_delta = k
            
        new_size = current_size + size_delta
        new_viol = current_viol + viol_delta
        new_cost = -new_size + 1000 * new_viol
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_mask = new_mask
            current_size = new_size
            current_viol = new_viol
            current_cost = new_cost
            
            if new_viol == 0 and new_size > best_mask.bit_count():
                best_mask = new_mask
                best_cost = new_cost
                
        T *= alpha
        
    # Extract result from best_mask
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        result.append(vertices[i])
        m ^= lsb
        
    return result
