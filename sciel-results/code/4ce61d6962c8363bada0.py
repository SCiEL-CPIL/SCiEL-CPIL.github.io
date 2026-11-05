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
    Uses simulated annealing to find a voltage assignment satisfying
    non-backtracking closed walk constraints.
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
    # A closed walk corresponds to a vector c in Z^15 where c_i is the net number of times
    # edge i is traversed in the "forward" direction.
    # The condition is c . v != 0 (mod m).
    
    max_walk_len = g - 1
    coeff_set = set()
    
    # DFS to find all non-backtracking closed walks of length 5 to max_walk_len
    # Note: Petersen girth is 5, so we start checking from length 5.
    # Walks of length < 5 are impossible in Petersen (girth 5).
    # Actually, we need to check up to g-1.
    
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
                    # c and -c represent the same constraint (c.v=0 iff -c.v=0)
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
        # No constraints? Shouldn't happen for g > 5
        return None
        
    coeff_list = list(coeff_set)
    coeff_matrix = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(coeff_matrix)
    
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

    # We want to minimize n = 10 * m.
    # Lower bound for n(3,16) is 512. So m >= 51.2.
    # Smallest prime >= 52 is 53.
    # Let's try primes starting from 53 upwards.
    
    # Generate a list of candidate primes
    primes = []
    candidate = 53
    while len(primes) < 50:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
        
    best_n = float('inf')
    best_edges = None
    
    for m in primes:
        if time.time() > deadline - 5:
            break
            
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
            
        # Try to find a valid voltage assignment for this m
        # Use Simulated Annealing
        found = _solve_voltage_sa(coeff_matrix, m, deadline)
        
        if found is not None:
            # Build the graph
            edges = _build_lifted_graph(pet_edges, found, m)
            # Verify girth to be safe (though SA should have ensured it)
            # Verification is expensive for large n, so we trust the constraint check
            # but do a quick sanity check on degree and simplicity.
            if _is_valid_graph(n_candidate, edges, k=3):
                return n_candidate, edges
            else:
                # Should not happen, but if it does, continue
                pass

    return None

def _solve_voltage_sa(coeff_matrix, m, deadline):
    """
    Uses Simulated Annealing to find a voltage vector v in (Z_m\{0})^15
    such that coeff_matrix @ v != 0 (mod m) for all rows.
    """
    n_vars = 15
    # Initial random assignment (non-zero)
    v = np.array([random.randint(1, m - 1) for _ in range(n_vars)], dtype=np.int64)
    
    def cost(v):
        # Number of constraints violated
        if len(coeff_matrix) == 0:
            return 0
        residues = np.mod(coeff_matrix @ v, m)
        return np.sum(residues == 0)
        
    best_v = v.copy()
    best_cost = cost(v)
    
    if best_cost == 0:
        return best_v

    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    max_iters = 200000 # Cap iterations, but time is the main limit
    
    current_v = v.copy()
    current_cost = best_cost
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # Pick a random variable to change
        j = random.randint(0, n_vars - 1)
        old_val = current_v[j]
        
        # Pick a new random value (non-zero)
        new_val = random.randint(1, m - 1)
        if new_val == old_val:
            continue
            
        current_v[j] = new_val
        new_cost = cost(current_v)
        
        delta = new_cost - current_cost
        
        if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_v = current_v.copy()
                if best_cost == 0:
                    return best_v
        else:
            current_v[j] = old_val
            
        T *= cooling
        
    if best_cost == 0:
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
