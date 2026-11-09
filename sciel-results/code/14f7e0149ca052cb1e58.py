import time
import random
import sys
from collections import deque

def _has_girth_at_least(adj, n, target_girth):
    """Check if girth >= target_girth. Returns False early if a shorter cycle is found."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < target_girth:
                        return False
    return True

def _bfs_girth(adj, n):
    """Compute exact girth using BFS from each vertex."""
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth <= 2:
                        return min_girth
    return min_girth

def _construct_cage_3_16():
    """Attempt to find a 3-regular graph of girth 16 with 936 vertices using local search."""
    n = 936
    half = n // 2
    deadline = time.time() + 20.0  # Short time budget for local search
    
    # Generate a random 3-regular bipartite graph
    # Left: 0..467, Right: 468..935
    # We represent edges as a list of (u, v) where u in Left, v in Right
    # Or just adjacency lists.
    
    def generate_random_bipartite():
        # Use configuration model
        left_stubs = []
        right_stubs = []
        for i in range(half):
            left_stubs.extend([i, i, i])
            right_stubs.extend([i, i, i])
        
        random.shuffle(left_stubs)
        random.shuffle(right_stubs)
        
        adj = [set() for _ in range(n)]
        # Pair left stubs with right stubs
        # left_stubs[i] is a left vertex, right_stubs[i] is a right vertex
        for i in range(half * 3):
            u = left_stubs[i]
            v = right_stubs[i] + half
            if v in adj[u]:
                # Multi-edge, retry or fix
                # For simplicity, we'll just regenerate if we detect issues in a wrapper
                return None
            adj[u].add(v)
            adj[v].add(u)
        
        # Check degrees
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return [list(s) for s in adj]

    # Local search
    for attempt in range(100): # Max 100 restarts within time
        if time.time() > deadline:
            break
        
        adj = generate_random_bipartite()
        if adj is None:
            continue
            
        # Iterative improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find a short cycle
            # We use BFS from a random node up to depth 7
            start = random.randint(0, n - 1)
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            cycle_found = None
            
            while queue:
                v = queue.popleft()
                if dist[v] >= 7:
                    continue
                for u in adj[v]:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        parent[u] = v
                        queue.append(u)
                    elif u != parent[v] and v != parent[u]:
                        cycle_len = dist[v] + dist[u] + 1
                        if cycle_len < 16:
                            # Reconstruct cycle
                            # Path from start to v and start to u
                            path_v = []
                            curr = v
                            while curr != start:
                                path_v.append(curr)
                                curr = parent[curr]
                            path_v.append(start)
                            
                            path_u = []
                            curr = u
                            while curr != start:
                                path_u.append(curr)
                                curr = parent[curr]
                            path_u.append(start)
                            
                            # Combine paths to form cycle
                            # The cycle is path_v + reversed(path_u[1:])
                            # But we need to break it.
                            # A 2-switch involves swapping edges.
                            # For a cycle (v0, v1, ..., vk-1, v0), we can swap (v0, v1) and (vk-1, v0) with other edges?
                            # Simpler: Pick two non-adjacent edges in the cycle and swap their endpoints.
                            # Let cycle be c0, c1, ..., c_{L-1}.
                            # Edges are (c0,c1), (c1,c2), etc.
                            # Swap (c0,c1) and (c2,c3)? No, that creates two cycles.
                            # Standard 2-switch on bipartite:
                            # Find cycle. Pick two edges (u1, v1) and (u2, v2) in the cycle such that they are not adjacent.
                            # Replace with (u1, v2) and (u2, v1).
                            # This breaks the cycle.
                            
                            # Extract cycle vertices in order
                            # path_v is start..v, path_u is start..u
                            # The cycle is v .. start .. u .. v?
                            # Actually, the cycle is the path from v to start plus path from start to u.
                            # Let's just get the list of vertices in the cycle.
                            cycle_verts = path_v[:0:-1] + [start] + path_u[:0:-1]
                            # Wait, path_v is [start, ..., v]. Reversed is [v, ..., start].
                            # path_u is [start, ..., u].
                            # Cycle: v -> ... -> start -> ... -> u -> v?
                            # No, u is connected to v? No, u is a neighbor of v?
                            # In BFS, if u is already visited and u != parent[v], then (v,u) is a back edge.
                            # So the cycle is v-u-...-start-...-v.
                            # Vertices: v, u, parent[u], ..., start, parent[v], ..., v.
                            # Let's construct the list.
                            c1 = v
                            c2 = u
                            # Path from u to start
                            pu = []
                            curr = u
                            while curr != start:
                                pu.append(curr)
                                curr = parent[curr]
                            pu.append(start)
                            # Path from start to v
                            pv = []
                            curr = v
                            while curr != start:
                                pv.append(curr)
                                curr = parent[curr]
                            pv.append(start)
                            
                            # Cycle: v, u, pu[1:-1], start, pv[1:-1], v
                            # Actually, simpler:
                            # Cycle is v -> u -> ... -> start -> ... -> v
                            # Vertices: [v] + [u] + pu[1:-1] + pv[1:-1]
                            # Note: pu[0] is u, pv[0] is v.
                            # pu is [u, ..., start].
                            # pv is [v, ..., start].
                            # Cycle: v, u, pu[1:-1], pv[1:-1]
                            # Length: 1 + 1 + (len(pu)-2) + (len(pv)-2) = len(pu) + len(pv) - 2
                            # dist[v] + dist[u] + 1 = (len(pv)-1) + (len(pu)-1) + 1 = len(pu) + len(pv) - 1.
                            # Matches.
                            
                            cycle_verts = [v, u] + pu[1:-1] + pv[1:-1]
                            L = len(cycle_verts)
                            
                            # Pick two non-adjacent edges to swap
                            # Edges are (cycle_verts[i], cycle_verts[(i+1)%L])
                            # We need to pick i and j such that |i-j| > 1 and |i-j| < L-1
                            # And the swap must be valid (no multi-edges)
                            found_swap = False
                            for _ in range(10): # Try a few random swaps
                                i = random.randint(0, L - 1)
                                j = random.randint(0, L - 1)
                                if abs(i - j) <= 1 or abs(i - j) >= L - 1:
                                    continue
                                
                                u1, v1 = cycle_verts[i], cycle_verts[(i + 1) % L]
                                u2, v2 = cycle_verts[j], cycle_verts[(j + 1) % L]
                                
                                # Ensure bipartite consistency: u1, u2 in Left, v1, v2 in Right
                                # In a bipartite graph, cycle vertices alternate Left/Right.
                                # So if i is even, u1 is Left, v1 is Right.
                                # If j is even, u2 is Left, v2 is Right.
                                # Swap: (u1, v2) and (u2, v1)
                                
                                # Check if edges already exist
                                if v2 in adj[u1] or v1 in adj[u2]:
                                    continue
                                    
                                # Perform swap
                                adj[u1].remove(v1)
                                adj[u1].append(v2)
                                adj[v1].remove(u1)
                                adj[v1].append(u2)
                                
                                adj[u2].remove(v2)
                                adj[u2].append(v1)
                                adj[v2].remove(u2)
                                adj[v2].append(u1)
                                
                                found_swap = True
                                improved = True
                                break
                            if found_swap:
                                break # Restart cycle search
                            
            # After loop, check if girth is 16
            if _has_girth_at_least(adj, n, 16):
                edges = []
                for i in range(half):
                    for j in adj[i]:
                        if j > half:
                            edges.append((i, j))
                return n, edges

    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    
    def _build_luw(d):
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
        
        rule_list = [rule[c] for c in range(2, d + 1)]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for idx, c in enumerate(range(2, d + 1)):
                    a, b = rule_list[idx]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        i = 0
        while i < len(order):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        return len(order), edges

    deadline = time.time() + 100.0
    
    for d in range(5, 20):
        if time.time() > deadline:
            break
        
        n, edges = _build_luw(d)
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if _has_girth_at_least(adj, n, g):
            return n, edges
        
    d_orig = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _build_luw(d_orig)

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0
    
    if k == 3 and g == 16:
        result = _construct_cage_3_16()
        if result is not None:
            return result
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)
