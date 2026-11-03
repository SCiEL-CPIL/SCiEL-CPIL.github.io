import time
import sys
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    n = 320
    
    # Helper to generate a random 4-regular bipartite graph
    def gen_random_4reg_bipartite(n_half, seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Configuration model for bipartite graph
        # Left: 0..n_half-1, Right: n_half..n-1
        left_stubs = []
        right_stubs = []
        for i in range(n_half):
            left_stubs.extend([i] * 4)
            right_stubs.extend([n_half + i] * 4)
        
        random.shuffle(left_stubs)
        random.shuffle(right_stubs)
        
        edges = set()
        valid = True
        for l, r in zip(left_stubs, right_stubs):
            if l == r: # Should not happen in bipartite
                valid = False
                break
            if (l, r) in edges: # Multi-edge
                valid = False
                break
            edges.add((l, r))
            
        if not valid or len(edges) != n_half * 4:
            return None
        
        return sorted(edges)

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def count_short_cycles(adj, n, g):
        # Use matrix powers for speed
        A = np.zeros((n, n), dtype=int)
        for u, v in adj:
            A[u, v] = 1
            A[v, u] = 1
        
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A4 @ A2
        A8 = A4 @ A4
        
        # Trace(A^k) counts closed walks of length k.
        # For 4-cycles: Tr(A^4) = 16n + 8 * (# 4-cycles) * 2? 
        # Actually, Tr(A^4) = 16n + 8 * C4.
        # Tr(A^6) = 16n*3 + ... it's complex to extract exact cycle counts from trace without corrections.
        # But minimizing the trace is a good surrogate.
        # Tr(A^4) is dominated by 4-cycles if they exist.
        # Tr(A^6) is dominated by 6-cycles.
        # Tr(A^8) is dominated by 8-cycles.
        
        c4 = np.trace(A4) - 16 * n
        c6 = np.trace(A6) - 16 * n * 3 # Approximate baseline
        c8 = np.trace(A8) - 16 * n * 11 # Approximate baseline
        
        # The constants are tricky. Let's just use the raw traces as a heuristic.
        # Lower is better.
        obj = np.trace(A4) + 10 * np.trace(A6) + 100 * np.trace(A8)
        return obj, A

    def find_short_cycle(adj, n, max_len):
        """Find a cycle of length < max_len using BFS from each vertex."""
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = [start]
            visited[start] = 0
            parent[start] = -2
            
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if visited[u] >= max_len:
                    continue
                
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        # Reconstruct cycle
                        a, b = u, v
                        while a != b:
                            if visited[a] > visited[b]:
                                a = parent[a]
                            else:
                                b = parent[b]
                        lca = a
                        
                        path1 = []
                        a = u
                        while a != lca:
                            path1.append(a)
                            a = parent[a]
                        path1.append(lca)
                        
                        path2 = []
                        b = v
                        while b != lca:
                            path2.append(b)
                            b = parent[b]
                        
                        cycle = path1 + path2[::-1]
                        if len(cycle) < max_len:
                            return cycle
        return None

    def check_girth(adj, n, min_girth):
        """Verify girth >= min_girth"""
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = [start]
            visited[start] = 0
            parent[start] = -2
            
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if visited[u] >= min_girth:
                    continue
                
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = visited[u] + visited[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def try_edge_switch(adj, n, cycle):
        """Try to break the cycle via edge switching"""
        if len(cycle) < 4:
            return None
        
        # Get edges in the cycle
        cycle_edges = []
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            if u > v: u, v = v, u
            cycle_edges.append((u, v))
        
        # Pick two non-adjacent edges
        edges_set = set()
        for u, v in adj:
            if u > v: u, v = v, u
            edges_set.add((u, v))
            
        # We need to pick two edges (a,b) and (c,d) from the cycle such that they don't share vertices
        # And the new edges (a,c) and (b,d) are not in the graph.
        
        m = len(cycle_edges)
        for i in range(m):
            e1 = cycle_edges[i]
            for j in range(i+1, m):
                e2 = cycle_edges[j]
                
                # Check if non-adjacent
                if set(e1) & set(e2):
                    continue
                
                a, b = e1
                c, d = e2
                
                # Option 1: (a,c) and (b,d)
                new_e1 = (min(a,c), max(a,c))
                new_e2 = (min(b,d), max(b,d))
                
                if new_e1 not in edges_set and new_e2 not in edges_set and new_e1 != new_e2:
                    # Perform switch
                    new_adj = [list(lst) for lst in adj]
                    
                    # Remove e1 and e2
                    new_adj[a].remove(b)
                    new_adj[b].remove(a)
                    new_adj[c].remove(d)
                    new_adj[d].remove(c)
                    
                    # Add new_e1 and new_e2
                    new_adj[a].append(c)
                    new_adj[c].append(a)
                    new_adj[b].append(d)
                    new_adj[d].append(b)
                    
                    # Verify degrees (should be 4)
                    # And simplicity (no multi-edges)
                    # Check if (a,c) was already there? We checked edges_set.
                    # But we must ensure we didn't create a multi-edge.
                    # Since we removed (a,b) and (c,d), and added (a,c) and (b,d),
                    # and (a,c) was not in edges_set, it's safe.
                    
                    return new_adj
                
                # Option 2: (a,d) and (b,c)
                new_e1 = (min(a,d), max(a,d))
                new_e2 = (min(b,c), max(b,c))
                
                if new_e1 not in edges_set and new_e2 not in edges_set and new_e1 != new_e2:
                    new_adj = [list(lst) for lst in adj]
                    
                    new_adj[a].remove(b)
                    new_adj[b].remove(a)
                    new_adj[c].remove(d)
                    new_adj[d].remove(c)
                    
                    new_adj[a].append(d)
                    new_adj[d].append(a)
                    new_adj[b].append(c)
                    new_adj[c].append(b)
                    
                    return new_adj
        return None

    best_adj = None
    best_obj = float('inf')
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        edges = gen_random_4reg_bipartite(n // 2, seed=attempt)
        if edges is None:
            continue
        
        adj = build_adj(edges, n)
        
        # Initial objective
        obj, A = count_short_cycles(adj, n, g)
        
        if obj < best_obj:
            best_obj = obj
            best_adj = [list(lst) for lst in adj]
        
        # Local search
        for step in range(500):
            if time.time() > deadline:
                break
            
            cycle = find_short_cycle(adj, n, g)
            if cycle is None:
                # Check if girth is actually >= 10
                if check_girth(adj, n, g):
                    # Convert adj back to edges
                    final_edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                final_edges.append((u, v))
                    return n, final_edges
            
            new_adj = try_edge_switch(adj, n, cycle)
            if new_adj is not None:
                # Check if the new graph is better
                new_obj, _ = count_short_cycles(new_adj, n, g)
                if new_obj <= obj:
                    obj = new_obj
                    adj = new_adj
                    if obj < best_obj:
                        best_obj = obj
                        best_adj = [list(lst) for lst in adj]
                else:
                    # Simulated annealing style: accept with small probability
                    if random.random() < 0.01:
                        adj = new_adj
                        obj = new_obj

    # Return best result found
    if best_adj is not None:
        final_edges = []
        for u in range(n):
            for v in best_adj[u]:
                if u < v:
                    final_edges.append((u, v))
        return n, final_edges

    # Fallback
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Fallback LUW construction"""
    q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]
    
    def make_key(side, x):
        return (side, x)
    
    start_x = (None,) + (0,) * d
    start_key = make_key(0, start_x)
    
    index = {start_key: 0}
    order = [(0, start_x)]
    edges = []
    
    queue = [0]
    head = 0
    n_vertices = 1
    
    while head < len(order):
        v_idx = queue[head]
        head += 1
        side, x = order[v_idx]
        
        for first in range(q):
            y_list = [None] * (d + 1)
            y_list[0] = None
            y_list[1] = first
            for c in range(2, d + 1):
                a = rule_a[c]
                b = rule_b[c]
                if side == 0:
                    y_list[c] = gf_add(x[c], gf_mul(y_list[a], x[b]))
                else:
                    y_list[c] = gf_sub(x[c], gf_mul(x[a], y_list[b]))
            
            other_side = 1 - side
            other_x = tuple(y_list)
            other_key = make_key(other_side, other_x)
            
            if other_key not in index:
                index[other_key] = n_vertices
                order.append((other_side, other_x))
                queue.append(n_vertices)
                n_vertices += 1
            
            if side == 0:
                edges.append((v_idx, index[other_key]))
                
    return n_vertices, edges
# EVOLVE-BLOCK-END
