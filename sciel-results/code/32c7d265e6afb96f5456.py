"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound_lower(k, g):
    """Return a safe lower bound on n(k,g)."""
    if g % 2 == 0:
        return 2 * sum((k-1)**i for i in range(g//2))
    else:
        return 1 + k * sum((k-1)**i for i in range((g-1)//2))

def _verify_girth(n, edges, k, g):
    """Verify the graph is k-regular, simple, and has girth >= g using non-backtracking walks."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, j in edges:
        if i == j:
            return False
        if i > j:
            i, j = j, i
        if (i, j) in edge_set:
            return False
        edge_set.add((i, j))
        adj[i].append(j)
        adj[j].append(i)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    for v in range(n):
        visited = {}
        queue = deque()
        for w in adj[v]:
            visited[(w, v)] = 1
            queue.append((w, v, 1))
        
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= g:
                break
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor == v:
                    if depth + 1 < g:
                        return False
                    continue
                state = (neighbor, node)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((neighbor, node, depth + 1))
    return True

def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction - the baseline."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def _construct_cage_voltage(k, g, deadline):
    """
    Attempt to construct the (4,10)-cage using voltage graph method.
    Base graph: Bipartite double cover of K5 (10 vertices, 4-regular).
    Voltage group: Z_32.
    Lift has 320 vertices.
    Uses local search to find valid voltage assignments.
    """
    if k != 4 or g != 10:
        return None

    n_base = 10
    n_lift = 320
    group_size = 32
    
    # Base graph edges: (i, 5+j) for i != j, i in 0..4, j in 0..4
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))
    
    num_edges = len(base_edges)  # 20 edges
    
    def build_edges(voltages):
        """Build lift edges from voltage assignment."""
        edges = []
        for idx, (u, v) in enumerate(base_edges):
            w = voltages[idx]
            for x in range(group_size):
                u_id = u * group_size + x
                v_id = v * group_size + (x + w) % group_size
                edges.append((u_id, v_id))
        return edges
    
    def count_short_cycles(voltages):
        """Fast girth check: returns number of short cycle violations.
        Uses early exit and sampled vertices for speed."""
        # Build adjacency efficiently
        adj = [[] for _ in range(n_lift)]
        edge_set = set()
        for idx, (u, v) in enumerate(base_edges):
            w = voltages[idx]
            for x in range(group_size):
                u_id = u * group_size + x
                v_id = v * group_size + (x + w) % group_size
                a, b = (u_id, v_id) if u_id < v_id else (v_id, u_id)
                if (a, b) in edge_set:
                    return float('inf')
                edge_set.add((a, b))
                adj[u_id].append(v_id)
                adj[v_id].append(u_id)
        
        # Non-backtracking walk check - sample vertices for speed
        violations = 0
        # Check all vertices but with early exit on first violation per vertex
        for v in range(n_lift):
            if violations > 10:  # Early exit if too many violations
                return violations
            visited = {}
            queue = deque()
            for w in adj[v]:
                visited[(w, v)] = 1
                queue.append((w, v, 1))
            
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= g:
                    break
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor == v:
                        if depth + 1 < g:
                            violations += 1
                        continue
                    state = (neighbor, node)
                    if state not in visited:
                        visited[state] = depth + 1
                        queue.append((neighbor, node, depth + 1))
        return violations
    
    rng = random.Random(42)
    
    # Try random initialization first
    best_violations = float('inf')
    best_voltages = None
    
    max_random = 30
    for attempt in range(max_random):
        if time.time() > deadline - 30:
            break
        voltages = [rng.randint(0, group_size - 1) for _ in range(num_edges)]
        viol = count_short_cycles(voltages)
        if viol == 0:
            edges = build_edges(voltages)
            if _verify_girth(n_lift, edges, k, g):
                return n_lift, edges
        if viol < best_violations:
            best_violations = viol
            best_voltages = voltages[:]
    
    # Local search from best random solution
    if best_voltages is not None:
        current = best_voltages[:]
        current_viol = best_violations
        temp = 5.0
        cooling = 0.95
        
        max_iterations = 3000
        for iteration in range(max_iterations):
            if time.time() > deadline - 15:
                break
            if current_viol == 0:
                edges = build_edges(current)
                if _verify_girth(n_lift, edges, k, g):
                    return n_lift, edges
                break
            
            # Pick a random edge to change
            idx = rng.randint(0, num_edges - 1)
            old_val = current[idx]
            current[idx] = rng.randint(0, group_size - 1)
            
            new_viol = count_short_cycles(current)
            
            # Simulated annealing acceptance
            if new_viol < current_viol:
                current_viol = new_viol
            else:
                delta = new_viol - current_viol
                if rng.random() < np.exp(-delta / max(temp, 0.01)):
                    current_viol = new_viol
                else:
                    current[idx] = old_val
            
            temp *= cooling
    
    # Final attempt: try a few more random restarts if time allows
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        voltages = [rng.randint(0, group_size - 1) for _ in range(num_edges)]
        viol = count_short_cycles(voltages)
        if viol == 0:
            edges = build_edges(voltages)
            if _verify_girth(n_lift, edges, k, g):
                return n_lift, edges
    
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Try the specific cage construction for (4,10)
    if k == 4 and g == 10:
        result = _construct_cage_voltage(k, g, deadline)
        if result is not None:
            return result
            
    # Fallback to LUW construction
    n, edges = _construct_luw(k, g)
    
    return n, edges

# EVOLVE-BLOCK-END
