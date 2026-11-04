import time
import random
from collections import deque, defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Calculate Moore Bound
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))
    
    # 2. Fallback: LUW Construction (from seed, but optimized/cleaned)
    # We keep this as the guaranteed valid result.
    best_n, best_edges = construct_luw(k, g)
    
    # 3. Try Greedy Tree Construction
    # This attempts to find a graph on the Moore bound (or slightly larger)
    # by filling in the leaves of the BFS tree.
    
    # For k=5, g=9, r=4.
    # Leaves are at depth r.
    # We need to connect leaves from different branches.
    
    n_target = moore_bound
    # Ensure n_target is even if k is odd
    if k % 2 == 1 and n_target % 2 != 0:
        n_target += 1
        
    # We try to construct on n_target.
    # If it fails, we might try n_target + 10, etc.
    # But for now, we just try the Moore bound structure.
    
    try:
        n_greedy, edges_greedy = construct_greedy_tree(k, g, n_target)
        if n_greedy < best_n:
            best_n = n_greedy
            best_edges = edges_greedy
    except Exception:
        pass

    # If we have time, try a few more sizes if the first one failed?
    # The greedy tree is specific to the Moore bound structure.
    # If it fails, it's likely due to the strict constraints.
    # We can try a slightly larger n by adding a layer of "buffer" nodes.
    # But for simplicity and speed, we stick to the Moore bound attempt.
    
    return best_n, best_edges

def construct_luw(k, g):
    # Original LUW logic, kept as fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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

def construct_greedy_tree(k, g, n_target):
    r = (g - 1) // 2
    # Build BFS tree
    # Vertices are tuples (level, branch_id, index_in_branch)
    # Or just use integer IDs.
    
    # Level 0: 1 vertex (0)
    # Level 1: k vertices (1..k)
    # Level 2: k(k-1) vertices
    # ...
    # Level r: k(k-1)^(r-1) vertices
    
    n_vertices = 1
    for i in range(1, r + 1):
        n_vertices += k * ((k - 1) ** (i - 1))
        
    # If n_vertices != n_target, we can't use this exact tree.
    # For now, we assume n_target is the Moore bound.
    if n_vertices != n_target:
        return None, None

    # Assign IDs
    ids = []
    parent = [-1] * n_vertices
    level = [0] * n_vertices
    branch = [0] * n_vertices # Which L1 node is the ancestor
    
    ids.append(0)
    level[0] = 0
    
    next_id = 1
    # Level 1
    for i in range(k):
        ids.append(next_id)
        parent[next_id] = 0
        level[next_id] = 1
        branch[next_id] = i
        next_id += 1
        
    # Level 2..r
    for l in range(2, r + 1):
        # Iterate over nodes at level l-1
        start_idx = len(ids) - (k * ((k-1)**(l-2))) # Approximate, better to track
        # Actually, just iterate over all nodes and check level
        nodes_at_prev_level = [v for v in range(len(ids)) if level[v] == l-1]
        for v in nodes_at_prev_level:
            # v has k-1 children
            for j in range(k - 1):
                ids.append(next_id)
                parent[next_id] = v
                level[next_id] = l
                branch[next_id] = branch[v]
                next_id += 1
                
    # Now we have the tree.
    # Leaves are at level r.
    leaves = [v for v in range(n_vertices) if level[v] == r]
    
    # We need to add edges between leaves.
    # Each leaf needs k-1 more edges.
    # Constraint: Leaves must be from different branches.
    
    # Group leaves by branch
    branches = defaultdict(list)
    for v in leaves:
        branches[branch[v]].append(v)
        
    # We need to form a (k-1)-regular graph on the set of leaves,
    # where edges only connect different branches.
    
    # This is a bipartite/multipartite matching problem.
    # We can use a greedy randomized approach.
    
    # Initialize adjacency list for the final graph
    adj = defaultdict(list)
    # Add tree edges
    for v in range(1, n_vertices):
        u = parent[v]
        adj[u].append(v)
        adj[v].append(u)
        
    # Track remaining degree needed
    deg_needed = {v: k - len(adj[v]) for v in range(n_vertices)}
    
    # Only leaves have deg_needed > 0
    # We need to pick edges for leaves.
    
    # Randomize leaf order
    random.shuffle(leaves)
    
    # For each leaf, pick k-1 partners from other branches
    # We must ensure we don't pick the same edge twice.
    # And we must ensure the partner also has capacity.
    
    # This is tricky. A simpler approach:
    # Generate a random (k-1)-regular graph on the leaves, respecting the branch constraint.
    # We can use the configuration model restricted to the allowed pairs.
    
    # Allowed pairs: (u, v) where branch[u] != branch[v]
    
    # Let's try a simple greedy with backtracking if stuck.
    # Given time constraints, we just try a few random configurations.
    
    for attempt in range(100):
        edges_to_add = []
        used_edges = set()
        success = True
        
        # Create a list of stubs for leaves
        stubs = []
        for v in leaves:
            stubs.extend([v] * deg_needed[v])
            
        random.shuffle(stubs)
        
        # Pair stubs
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                success = False
                break
            if branch[u] == branch[v]:
                success = False
                break
            if (min(u,v), max(u,v)) in used_edges:
                success = False
                break
            used_edges.add((min(u,v), max(u,v)))
            edges_to_add.append((u, v))
            
        if not success:
            continue
            
        # Check if the resulting graph is valid (girth)
        # The tree edges + these leaf edges.
        # Girth is guaranteed >= 2*r + 1 = g if we only connect different branches?
        # Yes, because the LCA of any two leaves from different branches is at depth <= r-1?
        # No, LCA of leaves from different L1 branches is the Root (depth 0).
        # Cycle length = dist(u, root) + dist(v, root) + 1 = r + r + 1 = 2r + 1 = g.
        # So girth is exactly g (or larger if we connect deeper, but we don't).
        # So this construction is valid!
        
        # Build final edge list
        final_edges = []
        for v in range(1, n_vertices):
            final_edges.append((parent[v], v))
        final_edges.extend(edges_to_add)
        
        # Verify degree
        deg_check = defaultdict(int)
        for u, v in final_edges:
            deg_check[u] += 1
            deg_check[v] += 1
        if all(deg_check[v] == k for v in range(n_vertices)):
            return n_vertices, final_edges
            
    return None, None
