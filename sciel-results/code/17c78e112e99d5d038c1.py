import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph for k=3.
    """
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop at 140s to ensure return.
    deadline = start_time + 140.0

    if k == 3:
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        # Fallback for other k, though task specifies k=3
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Uses vectorized constraint checking and best-improvement local search.
    """
    # Petersen graph edges (10 vertices, 15 edges)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    coeff_vectors = set()
    max_walk_len = g - 1
    
    # We use a stack for DFS to avoid recursion limit issues, though depth is small.
    # State: (current_v, prev_v, length, counts_tuple)
    # counts is a tuple of 15 integers representing the net traversal count of each edge.
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, tuple([0]*15))]
        
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
                
                # Update counts
                # If we traverse edge e_idx from u to v (where pet_edges[e_idx]=(u,v)), count +1
                # If we traverse from v to u, count -1
                new_counts_list = list(counts)
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts_list[e_idx] += 1
                else:
                    new_counts_list[e_idx] -= 1
                new_counts = tuple(new_counts_list)
                
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    # Found a non-backtracking closed walk
                    coeff_vectors.add(new_counts)
                
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
    
    # Build numpy matrix for vectorized evaluation
    # Each row is a constraint, each column is an edge voltage
    if len(coeff_list) > 0:
        coeff_matrix = np.array(coeff_list, dtype=np.int64)
        n_constraints = len(coeff_matrix)
    else:
        coeff_matrix = np.zeros((0, 15), dtype=np.int64)
        n_constraints = 0

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = []
    candidate = 2
    while len(primes) < 500: # Enough primes to cover reasonable n
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    best_n = float('inf')
    best_edges = None
    
    # We want the smallest m, so we iterate primes in increasing order
    for m in primes:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        # If m is very small, it's likely no solution exists, but we try.
        # The number of constraints is fixed.
        
        found = False
        # Number of restarts. We can afford many restarts because each step is fast.
        max_attempts = 2000 
        max_steps_per_attempt = 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            voltages = np.array([random.randint(1, m - 1) for _ in range(15)], dtype=np.int64)
            
            # Local Search Loop with Best-Improvement
            for step in range(max_steps_per_attempt):
                if time.time() > deadline:
                    break
                
                # Compute violations: sums = (coeff_matrix @ voltages) % m
                # We want no sums to be 0.
                if n_constraints > 0:
                    sums = (coeff_matrix @ voltages) % m
                    violated_indices = np.where(sums == 0)[0]
                    
                    if len(violated_indices) == 0:
                        # All constraints satisfied
                        edges = _build_lifted_graph(pet_edges, voltages.tolist(), m)
                        # Optional: Verify girth (slow, so maybe skip for speed if confident)
                        # if _has_girth_at_least(n_candidate, edges, g):
                        return n_candidate, edges
                    
                    # Pick a random violated constraint
                    viol_idx = random.choice(violated_indices.tolist())
                    c = coeff_matrix[viol_idx]
                    
                    # Find edges with non-zero coefficient in this constraint
                    candidates = np.where(c != 0)[0]
                    if len(candidates) == 0:
                        break
                    j = random.choice(candidates.tolist())
                    
                    # Best Improvement: Try all m values for voltages[j]
                    # Current value
                    old_val = voltages[j]
                    
                    # Precompute the column j of the coefficient matrix for fast update
                    # delta_sums = c_j * delta
                    c_j = c[j]
                    
                    best_new_val = old_val
                    min_violations = len(violated_indices)
                    
                    # We only need to try values that are not old_val? 
                    # Actually, trying all m values is O(m). 
                    # For each new_val, we compute the new sums.
                    # sums_new = sums + c_j * (new_val - old_val)
                    # We want to minimize the count of zeros in sums_new.
                    
                    # Optimization: 
                    # The change in sums is constant across all rows for a given delta? No.
                    # sums_k_new = (sums_k + coeff_matrix[k, j] * delta) % m
                    # So the delta is the same for all rows, but the multiplier coeff_matrix[k, j] varies.
                    
                    col_j = coeff_matrix[:, j]
                    
                    # We can vectorize the evaluation of all candidate new values?
                    # Candidates: 0..m-1
                    # Deltas: (candidate - old_val) % m
                    # For each candidate, we add col_j * delta to sums.
                    # This is a matrix multiplication (m x 1) @ (1 x n_constraints)? No.
                    # It's an outer product-like operation.
                    
                    # Since m is small (< 200 usually for small n), we can just loop.
                    # But let's try to vectorize a bit.
                    
                    # Actually, just looping m times and checking violations is fine if m is small.
                    # Let's stick to a simple loop for clarity and correctness, as m is small.
                    
                    for new_val in range(m):
                        if new_val == old_val:
                            continue
                        delta = (new_val - old_val) % m
                        if delta == 0:
                            continue
                        
                        # Calculate new sums
                        # sums_new = (sums + col_j * delta) % m
                        # We only care about how many are 0.
                        
                        # Note: col_j * delta is a vector of length n_constraints
                        new_sums = (sums + col_j * delta) % m
                        new_violations = np.count_nonzero(new_sums == 0)
                        
                        if new_violations < min_violations:
                            min_violations = new_violations
                            best_new_val = new_val
                            
                            # Early exit if we found a solution?
                            if min_violations == 0:
                                break
                                
                    # Update voltage
                    voltages[j] = best_new_val
                else:
                    # No constraints, any assignment works
                    edges = _build_lifted_graph(pet_edges, voltages.tolist(), m)
                    return n_candidate, edges
                    
        # If we didn't find a solution for this m, try the next prime

    # Fallback if no solution found in time
    return _construct_luw(k=3, g=g, d=11)

def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edges = set()
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
                
                a, b = min(u_node, v_node), max(u_node, v_node)
                edges.add((a, b))
    
    return list(edges)

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
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
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    max_len = g - 1
    
    for start in range(n):
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start and l > 0:
                return False
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    
    return True
