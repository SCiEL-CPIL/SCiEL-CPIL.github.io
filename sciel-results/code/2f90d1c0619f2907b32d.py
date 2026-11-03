import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 4 and g == 10:
        return _construct_4_10_heuristic(start_time, deadline)
    
    return _luw_construct(k, g)

def _construct_4_10_heuristic(start_time, deadline):
    # Base graph: K_{5,5} minus perfect matching
    # Vertices 0..4 (top), 5..9 (bottom)
    base_edges = []
    edge_index = {}
    idx = 0
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))
                edge_index[(i, 5 + j)] = idx
                edge_index[(5 + j, i)] = idx
                idx += 1
    
    num_edges = len(base_edges)
    
    # Enumerate cycles of length 4, 6, 8 in base graph
    # A cycle is a list of (u, v) pairs.
    # We need the signed sum of voltages.
    # For an undirected edge (u,v) with voltage w, if we traverse u->v, we add w. If v->u, we subtract w.
    # We can represent a cycle as a list of edge indices and signs.
    
    cycles = []
    
    # Helper to get neighbors
    adj = [[] for _ in range(10)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Find all simple cycles of length L
    def find_cycles(L):
        cycles_found = []
        # Start from each vertex
        for start in range(10):
            # DFS
            # Path: [start]
            # Visited: {start}
            # To avoid duplicates, we can impose an ordering? 
            # Or just find all and then deduplicate?
            # Given small graph, finding all is fine.
            
            def dfs(current, path, visited):
                if len(path) == L:
                    # Check if edge (current, start) exists
                    if start in adj[current]:
                        # Found a cycle
                        # Represent cycle
                        # Path: start -> ... -> current
                        # Edges: (start, path[1]), ..., (current, start)
                        cycle_edges = []
                        for i in range(len(path) - 1):
                            u = path[i]
                            v = path[i+1]
                            e_idx = edge_index[(u, v)]
                            # Sign: +1 if u < v in some canonical form? 
                            # Let's define voltage w_e for edge e.
                            # If we go u->v, and u is "smaller" in some sense, add w.
                            # Simpler: Just store the edge index and the direction relative to a fixed orientation.
                            # Let's fix orientation of each edge as (min(u,v), max(u,v)).
                            # If path goes min->max, sign is +1. Else -1.
                            if u < v:
                                cycle_edges.append((e_idx, 1))
                            else:
                                cycle_edges.append((e_idx, -1))
                        
                        # Last edge: current -> start
                        u = current
                        v = start
                        e_idx = edge_index[(u, v)]
                        if u < v:
                            cycle_edges.append((e_idx, 1))
                        else:
                            cycle_edges.append((e_idx, -1))
                            
                        cycles_found.append(cycle_edges)
                    return
                
                for neighbor in adj[current]:
                    if neighbor == start and len(path) < L:
                        continue
                    if neighbor not in visited:
                        # To avoid duplicates, we can enforce that start is the smallest vertex in the cycle
                        # If neighbor > start and we are at the end? No, just ensure start is min.
                        if start < min(path[1:]) if len(path) > 1 else True:
                             pass # This logic is tricky.
                        
                        # Simple deduplication: only start cycles where start is the smallest vertex
                        # We already iterate start from 0 to 9.
                        # We must ensure that no other vertex in the path is smaller than start.
                        # This guarantees each cycle is found exactly once (rooted at its min vertex).
                        if all(v > start for v in path[1:]):
                             pass
                        else:
                            continue # Skip if any previous vertex is smaller than start? 
                            # Wait, if start is the min, then all others must be > start.
                            # So we check if all vertices in path (excluding start) are > start.
                        
                        visited.add(neighbor)
                        dfs(neighbor, path + [neighbor], visited)
                        visited.remove(neighbor)
                        
            dfs(start, [start], {start})
        return cycles_found

    # Optimization: Only find cycles where start is the minimum vertex
    def find_cycles_opt(L):
        cycles_found = []
        for start in range(10):
            def dfs(current, path, visited):
                if len(path) == L:
                    if start in adj[current]:
                        cycle_edges = []
                        for i in range(len(path) - 1):
                            u = path[i]
                            v = path[i+1]
                            e_idx = edge_index[(u, v)]
                            if u < v:
                                cycle_edges.append((e_idx, 1))
                            else:
                                cycle_edges.append((e_idx, -1))
                        u = current
                        v = start
                        e_idx = edge_index[(u, v)]
                        if u < v:
                            cycle_edges.append((e_idx, 1))
                        else:
                            cycle_edges.append((e_idx, -1))
                        cycles_found.append(cycle_edges)
                    return
                
                for neighbor in adj[current]:
                    if neighbor == start:
                        continue
                    if neighbor not in visited:
                        # Ensure start is the unique minimum
                        if neighbor > start and all(v > start for v in path[1:]):
                            visited.add(neighbor)
                            dfs(neighbor, path + [neighbor], visited)
                            visited.remove(neighbor)
            dfs(start, [start], {start})
        return cycles_found

    cycles_4 = find_cycles_opt(4)
    cycles_6 = find_cycles_opt(6)
    cycles_8 = find_cycles_opt(8)
    
    all_cycles = cycles_4 + cycles_6 + cycles_8
    
    # Local Search
    # Initial random voltages
    voltages = [random.randint(0, 31) for _ in range(num_edges)]
    
    def count_bad_cycles(v):
        count = 0
        for cycle in all_cycles:
            s = 0
            for e_idx, sign in cycle:
                s += sign * v[e_idx]
            if s % 32 == 0:
                count += 1
        return count
        
    current_bad = count_bad_cycles(voltages)
    
    # Hill Climbing
    max_steps = 5000
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        if current_bad == 0:
            break
            
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        old_v = voltages[e_idx]
        
        # Try all 32 values
        best_v = old_v
        best_bad = current_bad
        
        # Precompute contribution of this edge to each cycle?
        # Just recompute is fast enough for small num_cycles
        
        for new_v in range(32):
            if new_v == old_v:
                continue
            voltages[e_idx] = new_v
            bad = count_bad_cycles(voltages)
            if bad < best_bad:
                best_bad = bad
                best_v = new_v
        
        voltages[e_idx] = best_v
        current_bad = best_bad
        
    if current_bad == 0:
        # Verify girth in the lift
        n_lift = 320
        edges = []
        for u, v_base in base_edges:
            i = u
            j = v_base - 5
            w = voltages[edge_index[(i, 5+j)]]
            for x in range(32):
                u_id = u * 32 + x
                v_id = v_base * 32 + (x + w) % 32
                edges.append((u_id, v_id))
                
        if _check_girth_fast(n_lift, edges, 10):
            return n_lift, edges
    
    # Fallback
    return _luw_construct(4, 10)

def _check_girth_fast(n, edges, min_girth):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = min_girth // 2 - 1
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            d = dist[u]
            if d >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    cycle_len = d + 1 + dist[v]
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g):
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
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

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
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
