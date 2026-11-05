import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Fallback for non-3-regular or if search fails
    def luw_construct(k, g):
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

    if k != 3:
        return luw_construct(k, g)
    
    # Petersen Graph Base
    # Vertices 0-4 (outer), 5-9 (inner)
    # Outer cycle: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    base_adj = [[] for _ in range(10)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    # Find all simple cycles in base graph up to length 15
    # Since base is small, we can do DFS from each vertex
    cycles = [] # List of (length, list_of_edges)
    visited_cycles = set()
    
    def find_cycles(start):
        stack = [(start, [start], [0], 0)] # node, path, edges, edge_count
        while stack:
            node, path, edges, e_count = stack.pop()
            if e_count > 0 and node == start:
                # Found a cycle
                # Canonical form to avoid duplicates
                min_idx = min(range(len(path)), key=lambda i: path[i])
                rotated_path = path[min_idx:] + path[:min_idx]
                # Reverse check
                rev_path = rotated_path[::-1]
                key = tuple(rotated_path) if rotated_path < rev_path else tuple(rev_path)
                if key not in visited_cycles:
                    visited_cycles.add(key)
                    if len(path) <= 15:
                        cycles.append((len(path), edges[:]))
                continue
            if e_count >= 15:
                continue
            for neighbor in base_adj[node]:
                if neighbor == start and e_count > 0:
                    stack.append((neighbor, path + [neighbor], edges + [e_count], e_count + 1))
                elif neighbor not in path:
                    stack.append((neighbor, path + [neighbor], edges + [e_count], e_count + 1))
    
    for i in range(10):
        find_cycles(i)
        
    # Map edge index to (u, v) for quick lookup in voltage sum calculation
    # We need to know the direction of the edge in the cycle to assign signs
    # But for voltage graphs, the sum is just sum of voltages with signs based on orientation.
    # Let's store cycles as list of (edge_idx, sign)
    
    edge_indices = {}
    for i, (u, v) in enumerate(base_edges):
        edge_indices[(u, v)] = i
        edge_indices[(v, u)] = i
        
    cycle_constraints = [] # (length, [(edge_idx, sign), ...])
    # Re-do cycle finding properly
    cycles = []
    visited_cycles = set()
    
    def find_cycles_v2(start):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > 1 and node == start:
                min_idx = min(range(len(path)), key=lambda i: path[i])
                rotated_path = path[min_idx:] + path[:min_idx]
                rev_path = rotated_path[::-1]
                key = tuple(rotated_path) if rotated_path < rev_path else tuple(rev_path)
                if key not in visited_cycles:
                    visited_cycles.add(key)
                    if len(path) <= 15:
                        # Convert to edge constraints
                        constraints = []
                        for i in range(len(path)-1):
                            u, v = path[i], path[i+1]
                            # Determine edge index
                            idx = edge_indices[(u, v)]
                            # Sign: +1 if u < v? No, orientation matters.
                            # Let's define voltage on edge (u,v) as v_uv.
                            # If we traverse u->v, we add v_uv.
                            # If we traverse v->u, we subtract v_uv (or add v_vu = -v_uv).
                            # Let's just store (u, v, idx).
                            constraints.append((u, v, idx))
                        cycles.append((len(path), constraints))
                continue
            if len(path) >= 15:
                continue
            for neighbor in base_adj[node]:
                if neighbor == start and len(path) > 1:
                    stack.append((neighbor, path + [neighbor]))
                elif neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
    
    for i in range(10):
        find_cycles_v2(i)
        
    # Precompute cycle data for fast evaluation
    # Each cycle: (length, [(u, v, idx), ...])
    # To evaluate, we need voltages v[idx].
    # Sum = sum( v[idx] if u->v else -v[idx] )
    
    def evaluate_assignment(volts, m):
        bad_count = 0
        for length, cons in cycles:
            s = 0
            for u, v, idx in cons:
                if u < v: # Arbitrary orientation definition for storage?
                    # We need to know if the traversal in the cycle was u->v or v->u
                    # The cons list has (u, v) in the order of the path.
                    # So if path is ...u, v..., we traverse u->v.
                    # We stored (u, v, idx).
                    # The voltage variable is associated with the edge idx.
                    # Let's say voltage is defined for (min(u,v), max(u,v)).
                    # If we traverse min->max, add vol. If max->min, sub vol.
                    a, b = sorted((u, v))
                    if u == a:
                        s += volts[idx]
                    else:
                        s -= volts[idx]
            s %= m
            g = math.gcd(s, m)
            lift_len = length * (m // g)
            if lift_len < g: # girth target g
                bad_count += 1
        return bad_count

    # Search for m
    # Moore bound n=512 -> 10m >= 512 -> m >= 52
    best_n = None
    best_edges = None
    
    for m in range(52, 100): # Try up to n=1000
        if time.time() > deadline:
            break
            
        # Simulated Annealing / Random Search
        best_cost = float('inf')
        best_volts = None
        
        # Multiple restarts
        for restart in range(50):
            if time.time() > deadline:
                break
                
            volts = [random.randint(0, m-1) for _ in range(15)]
            cost = evaluate_assignment(volts, m)
            
            if cost == 0:
                best_volts = volts
                best_cost = 0
                break
                
            # Local search
            temp = 1.0
            for step in range(200):
                if time.time() > deadline:
                    break
                # Pick random edge
                e_idx = random.randint(0, 14)
                old_val = volts[e_idx]
                new_val = random.randint(0, m-1)
                volts[e_idx] = new_val
                new_cost = evaluate_assignment(volts, m)
                
                if new_cost < cost or random.random() < math.exp((cost - new_cost) / temp):
                    cost = new_cost
                    if cost == 0:
                        break
                else:
                    volts[e_idx] = old_val
                    
                temp *= 0.95
                
            if cost < best_cost:
                best_cost = cost
                best_volts = volts[:]
                
        if best_cost == 0:
            # Construct graph
            # Vertices: (base_node, volt)
            # Edges: (base_u, vol_u) - (base_v, vol_v)
            # vol_v = vol_u + voltage(edge)
            
            n_vertices = 10 * m
            edges = []
            for i, (u, v) in enumerate(base_edges):
                vol = best_volts[i]
                for start_vol in range(m):
                    node_u = u * m + start_vol
                    node_v = v * m + (start_vol + vol) % m
                    # Ensure u < v in base to avoid duplicate edges?
                    # The base graph is simple. The lift is simple if voltages are consistent.
                    # We add edge (node_u, node_v)
                    edges.append((node_u, node_v))
                    
            # Verify girth
            # Simple BFS girth check
            adj = [[] for _ in range(n_vertices)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            girth = float('inf')
            for start in range(n_vertices):
                dist = [-1] * n_vertices
                parent = [-1] * n_vertices
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] * 2 + 1 >= girth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < girth:
                                girth = cycle_len
                                
            if girth >= g:
                return n_vertices, edges
            else:
                # Should not happen if evaluate_assignment was correct
                pass
                
    # Fallback to LUW if no small graph found
    return luw_construct(k, g)

# EVOLVE-BLOCK-END
