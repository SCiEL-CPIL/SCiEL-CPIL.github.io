import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Primary strategy: Voltage lift of the Petersen graph (k=3, g=16) using
    simulated annealing to find a voltage assignment that kills all short
    non-backtracking closed walks. This targets n=10*m which is far smaller
    than LUW constructions (n ~ 2*3^d).
    Fallback: LUW construction.
    """
    start_time = time.time()
    # Evaluator kills at 150s. Keep a safety margin.
    deadline = start_time + 135.0

    if k == 3 and g == 16:
        # Try the voltage lift construction which can yield much smaller graphs
        result = _construct_voltage_lift(g, deadline)
        if result is not None:
            return result
        # Fall back to LUW if voltage lift fails or times out
        return _construct_luw(k, g, d=11)
    elif k == 3:
        # For other girths, try voltage lift with appropriate base if possible,
        # otherwise LUW.
        # The Petersen lift is specifically tuned for g=16 in this context.
        # For general g, we can still try, but LUW is safer for small g.
        if g >= 8:
             result = _construct_voltage_lift(g, deadline)
             if result is not None:
                 return result
        return _construct_luw(k, g, d=max(3, (g-1)//2))
    else:
        return _construct_luw(k, g, d=7)

def _construct_voltage_lift(g, deadline):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Uses simulated annealing + local search polish to find a voltage assignment
    satisfying non-backtracking closed walk constraints.
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
    
    adj = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    # We need to ensure that for any such walk, the sum of voltages (with signs) is not 0 mod m.
    max_walk_len = g - 1
    coeff_set = set()
    
    # DFS to find all non-backtracking closed walks of length 5 to max_walk_len
    for start_v in range(n_base):
        # Stack: (current, prev, length, counts_list)
        stack = [(start_v, -1, 0, [0] * 15)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in adj[v]:
                if next_v == prev_v:
                    continue
                
                e_idx = edge_index[(v, next_v)]
                u_edge, v_edge = pet_edges[e_idx]
                
                # Determine sign: +1 if traversing u->v, -1 if v->u
                if v == u_edge and next_v == v_edge:
                    sign = 1
                else:
                    sign = -1
                
                new_counts = counts.copy()
                new_counts[e_idx] += sign
                new_l = l + 1
                
                if next_v == start_v and new_l >= 5:
                    # Found a closed walk
                    # Normalize the coefficient vector to avoid duplicates
                    c_tuple = tuple(new_counts)
                    # Find first non-zero
                    first_nz = -1
                    for i in range(15):
                        if c_tuple[i] != 0:
                            first_nz = i
                            break
                    if first_nz != -1:
                        if c_tuple[first_nz] < 0:
                            c_tuple = tuple(-x for x in c_tuple)
                        coeff_set.add(c_tuple)
                
                stack.append((next_v, v, new_l, new_counts))
    
    if not coeff_set:
        return None
        
    coeff_list = list(coeff_set)
    coeff_matrix = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(coeff_matrix)
    
    # Precompute for each variable j, which constraints involve it and the coefficient
    var_to_constraints = [[] for _ in range(15)]
    for ci in range(n_constraints):
        for j in range(15):
            c = coeff_matrix[ci][j]
            if c != 0:
                var_to_constraints[j].append((ci, c))
    
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

    # Generate candidate moduli: all m from 52 upward, prioritizing composites
    # that give smaller n. We'll try in order of increasing n = 10*m.
    # Lower bound for n(3,16) is 512, so m >= 52.
    # Try m = 52, 53, 54, 55, ... up to some limit
    
    best_n = float('inf')
    best_edges = None
    
    # Try moduli in order of increasing n
    # m from 52 to about 120 should be sufficient
    candidate_ms = list(range(52, 150))
    
    for m in candidate_ms:
        if time.time() > deadline - 5:
            break
            
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
            
        # Try to find a valid voltage assignment for this m
        # Use Simulated Annealing + local search
        found = _solve_voltage_sa(coeff_matrix, var_to_constraints, m, deadline)
        
        if found is not None:
            # Build the graph
            edges = _build_lifted_graph(pet_edges, found, m)
            # Verify degree and simplicity
            if _is_valid_graph(n_candidate, edges, k=3):
                return n_candidate, edges
            else:
                pass

    return None

def _solve_voltage_sa(coeff_matrix, var_to_constraints, m, deadline):
    """
    Uses Simulated Annealing with incremental cost updates and a local search
    polish pass to find a voltage vector v in (Z_m\{0})^15
    such that coeff_matrix @ v != 0 (mod m) for all rows.
    """
    n_vars = 15
    n_constraints = len(coeff_matrix)
    
    # Precompute residues matrix for fast incremental updates
    # For each constraint ci and variable j, store coeff_matrix[ci][j] mod m
    # We'll maintain a running sum for each constraint
    
    def compute_initial_residues(v):
        """Compute all constraint residues given voltage vector v."""
        residues = np.zeros(n_constraints, dtype=np.int64)
        for ci in range(n_constraints):
            s = 0
            for j in range(n_vars):
                if coeff_matrix[ci][j] != 0:
                    s += coeff_matrix[ci][j] * v[j]
            residues[ci] = s % m
        return residues
    
    def count_violations(residues):
        return int(np.sum(residues == 0))
    
    def local_search_polish(v, residues, max_iters=50000):
        """
        Local search: try changing each variable to each possible non-zero value.
        Accept if it reduces cost. Continue until no improvement or max_iters.
        """
        current_cost = count_violations(residues)
        if current_cost == 0:
            return v, residues
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            improved = False
            # Shuffle variable order for diversity
            var_order = list(range(n_vars))
            random.shuffle(var_order)
            
            for j in var_order:
                old_val = v[j]
                old_residues = residues.copy()
                old_cost = current_cost
                
                # Try all non-zero values except old_val
                # For efficiency, sample a subset if m is large
                if m <= 60:
                    candidates = [x for x in range(1, m) if x != old_val]
                else:
                    # Sample 20 random candidates plus the old value neighbors
                    candidates = [random.randint(1, m-1) for _ in range(20)]
                    candidates = [x for x in candidates if x != old_val]
                
                best_new_val = old_val
                best_new_cost = old_cost
                best_new_residues = old_residues
                
                for new_val in candidates:
                    # Compute new residues by updating only affected constraints
                    new_residues = residues.copy()
                    for (ci, c) in var_to_constraints[j]:
                        # Update constraint ci: old contribution was c * old_val, new is c * new_val
                        new_residues[ci] = (new_residues[ci] + c * (new_val - old_val)) % m
                    
                    new_cost = count_violations(new_residues)
                    if new_cost < best_new_cost:
                        best_new_cost = new_cost
                        best_new_val = new_val
                        best_new_residues = new_residues
                
                if best_new_val != old_val:
                    v[j] = best_new_val
                    residues = best_new_residues
                    current_cost = best_new_cost
                    improved = True
                    if current_cost == 0:
                        return v, residues
            
            if not improved:
                break
        
        return v, residues
    
    # Multiple restarts with SA + local search
    best_v = None
    best_cost = float('inf')
    
    n_restarts = 20
    sa_time_per_restart = 3.0  # seconds per SA restart
    
    for restart in range(n_restarts):
        if time.time() > deadline - 3:
            break
        
        # Initial random assignment
        v = np.array([random.randint(1, m - 1) for _ in range(n_vars)], dtype=np.int64)
        residues = compute_initial_residues(v)
        current_cost = count_violations(residues)
        
        if current_cost == 0:
            return v
        
        # Simulated Annealing with incremental updates
        T = 1.0
        cooling = 0.998
        min_T = 0.05
        sa_start = time.time()
        
        while time.time() - sa_start < sa_time_per_restart:
            if time.time() > deadline - 2:
                break
            
            # Pick a random variable to change
            j = random.randint(0, n_vars - 1)
            old_val = v[j]
            
            # Pick a new random value (non-zero)
            new_val = random.randint(1, m - 1)
            if new_val == old_val:
                continue
            
            # Compute new residues incrementally
            new_residues = residues.copy()
            for (ci, c) in var_to_constraints[j]:
                new_residues[ci] = (new_residues[ci] + c * (new_val - old_val)) % m
            
            new_cost = count_violations(new_residues)
            delta = new_cost - current_cost
            
            if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                v[j] = new_val
                residues = new_residues
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_v = v.copy()
                    if best_cost == 0:
                        return best_v
            else:
                pass  # Don't update v or residues
            
            T *= cooling
        
        # Local search polish after SA
        v_polished, residues_polished = local_search_polish(v.copy(), residues.copy())
        polished_cost = count_violations(residues_polished)
        
        if polished_cost < best_cost:
            best_cost = polished_cost
            best_v = v_polished
            if best_cost == 0:
                return best_v
    
    if best_cost == 0 and best_v is not None:
        return best_v
    return None

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

def _is_valid_graph(n, edges, k):
    """
    Quick check for degree and simplicity.
    """
    if len(edges) != n * k // 2:
        return False
    
    degree = [0] * n
    edge_set = set()
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        if (min(u,v), max(u,v)) in edge_set:
            return False
        edge_set.add((min(u,v), max(u,v)))
        degree[u] += 1
        degree[v] += 1
        
    for d in degree:
        if d != k:
            return False
            
    return True

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
    Fallback method.
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
    Used for verification if needed, but primarily relying on constraint satisfaction.
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
