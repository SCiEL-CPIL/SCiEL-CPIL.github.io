import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # The task is for k=4, g=10.
    # We use a lifting approach from the base graph K_{4,4}.
    # Base graph B has 8 vertices.
    # Lifted graph size is 8 * m.
    # We search for a voltage assignment in Z_m that eliminates cycles of length < g.
    
    if k != 4 or g != 10:
        # Fallback to LUW for other parameters
        return _luw_fallback(k, g)

    # Base graph: K_{4,4}
    # Vertices 0-3 (left), 4-7 (right)
    # Edges: (i, j+4) for i in 0..3, j in 0..3
    
    # Precompute cycles of length 4, 6, 8 in K_{4,4}
    # A cycle is a list of edge indices.
    # Edges are indexed 0..15.
    # Edge (i, j+4) has index i*4 + j.
    
    cycles_4 = []
    cycles_6 = []
    cycles_8 = []
    
    # Generate all simple cycles of length 4, 6, 8 in K_{4,4}
    # Since K_{4,4} is small, we can enumerate them.
    
    # Helper to get edge index
    def edge_idx(u, v):
        if u < 4 and v >= 4:
            return u * 4 + (v - 4)
        elif v < 4 and u >= 4:
            return v * 4 + (u - 4)
        else:
            return -1
            
    # Generate cycles by DFS
    def find_cycles():
        adj = {}
        for u in range(8):
            adj[u] = []
        for i in range(4):
            for j in range(4):
                u = i
                v = j + 4
                idx = i * 4 + j
                adj[u].append((v, idx))
                adj[v].append((u, idx))
                
        visited = [False] * 8
        path = []
        edge_path = []
        
        def dfs(u, depth, max_depth):
            if depth == max_depth:
                # Check if cycle closes
                if u == path[0]:
                    # Normalize cycle: start with min vertex, min edge
                    # We just store the edge indices
                    c = tuple(sorted(edge_path))
                    if c not in seen_cycles:
                        seen_cycles.add(c)
                        if max_depth == 4: cycles_4.append(c)
                        elif max_depth == 6: cycles_6.append(c)
                        elif max_depth == 8: cycles_8.append(c)
                    return
                
            for v, idx in adj[u]:
                if v == path[0] and depth == max_depth - 1:
                    # Close the cycle
                    new_edge_path = edge_path + [idx]
                    c = tuple(sorted(new_edge_path))
                    if c not in seen_cycles:
                        seen_cycles.add(c)
                        if max_depth == 4: cycles_4.append(c)
                        elif max_depth == 6: cycles_6.append(c)
                        elif max_depth == 8: cycles_8.append(c)
                elif v not in path:
                    path.append(v)
                    edge_path.append(idx)
                    dfs(v, depth + 1, max_depth)
                    path.pop()
                    edge_path.pop()
                    
        seen_cycles = set()
        for start in range(4): # Start from left side to avoid duplicates?
            # Actually, just run from 0 and check all?
            # K_{4,4} is small.
            pass
        # Simpler: Just enumerate all combinations?
        # Let's just use a standard cycle enumeration for small graph.
        # Since K44 is bipartite, cycles have even length.
        
        # For efficiency, we can just hardcode the cycle structure or generate quickly.
        # Number of 4-cycles: 36.
        # Number of 6-cycles: 
        # Number of 8-cycles:
        
        # Let's just do a quick BFS/DFS for each length.
        for L in [4, 6, 8]:
            # Reset
            seen_cycles.clear()
            for start in range(8):
                path = [start]
                edge_path = []
                # Fix start to be 0 to reduce duplicates?
                # No, just enumerate all and deduplicate.
                pass
        
        # This DFS might be slow if not careful.
        # Let's use a simpler method:
        # For K_{4,4}, a cycle of length 2L is a sequence of L left nodes and L right nodes.
        # Left nodes: l1, l2, ..., lL (distinct)
        # Right nodes: r1, r2, ..., rL (distinct)
        # Edges: (l1,r1), (r1,l2), (l2,r2), ..., (lL,rL), (rL,l1).
        # This forms a cycle.
        
        import itertools
        
        for L in [2, 3, 4]: # Length 4, 6, 8
            for lefts in itertools.permutations(range(4), L):
                for rights in itertools.permutations(range(4), L):
                    # Check if this forms a simple cycle
                    # It is simple if all lefts and rights are distinct (guaranteed by permutation)
                    # and the edges exist (they do in K44)
                    # But we need to avoid counting the same cycle multiple times.
                    # A cycle is defined by the set of edges.
                    edges = []
                    for i in range(L):
                        u = lefts[i]
                        v = rights[i] + 4
                        idx1 = u * 4 + (v - 4)
                        edges.append(idx1)
                        # Next edge is (rights[i], lefts[i+1])
                        # Wait, the sequence is l1-r1-l2-r2-...-lL-rL-l1
                        # So edges are (l1,r1), (l2,r1)?? No.
                        # Path: l1 -> r1 -> l2 -> r2 -> ... -> lL -> rL -> l1
                        # Edges:
                        # (l1, r1)
                        # (l2, r1) -- No, r1 is connected to l1 and l2.
                        # So edge (l2, r1) is used.
                        # Edge (l2, r2) is used.
                        # So the edges are:
                        # (l1, r1), (l2, r1), (l2, r2), (l3, r2), ...
                        # This is not a simple cycle in terms of edges if we just list pairs.
                        # The edges in the cycle are:
                        # e1: (l1, r1)
                        # e2: (l2, r1)
                        # e3: (l2, r2)
                        # e4: (l3, r2)
                        # ...
                        # e_{2L-1}: (lL, rL)
                        # e_{2L}: (l1, rL)
                        
                        pass
                    # This logic is getting messy.
                    # Let's just use the DFS with deduplication.
        
        break # Abort this manual generation, use DFS
        
    # Let's implement the DFS properly.
    adj = [[] for _ in range(8)]
    for i in range(4):
        for j in range(4):
            u = i
            v = j + 4
            idx = i * 4 + j
            adj[u].append((v, idx))
            adj[v].append((u, idx))
            
    all_cycles = {4: [], 6: [], 8: []}
    seen = set()
    
    def dfs_cycles(start, u, path, edge_path, depth, max_depth):
        if depth == max_depth:
            if u == start:
                # Found a cycle
                # Normalize: sort edge indices
                key = tuple(sorted(edge_path))
                if key not in seen:
                    seen.add(key)
                    all_cycles[max_depth].append(key)
            return
        
        for v, idx in adj[u]:
            if v == start and depth == max_depth - 1:
                # Close cycle
                new_edge_path = edge_path + [idx]
                key = tuple(sorted(new_edge_path))
                if key not in seen:
                    seen.add(key)
                    all_cycles[max_depth].append(key)
            elif v not in path:
                path.add(v)
                dfs_cycles(start, v, path, edge_path + [idx], depth + 1, max_depth)
                path.remove(v)
                
    for L in [4, 6, 8]:
        seen.clear()
        # Start from 0 to cover all?
        # In a connected graph, starting from 0 finds all cycles?
        # No, only cycles containing 0.
        # So we must start from all vertices.
        for s in range(8):
            dfs_cycles(s, s, {s}, [], 0, L)
            
    # Now we have all cycles.
    # Search for voltages in Z_m.
    # We try m = 40, 48, 56, ... up to 64.
    
    best_n = 512
    best_edges = None
    
    for m in [40, 48, 56, 64]:
        if time.time() > deadline:
            break
            
        # Simulated Annealing
        # State: list of 16 voltages, each in [0, m-1]
        # Cost: number of cycles with sum 0
        
        def cost(voltages):
            c = 0
            for L in [4, 6, 8]:
                for cyc in all_cycles[L]:
                    s = 0
                    for e in cyc:
                        s = (s + voltages[e]) % m
                    if s == 0:
                        c += 1
            return c
            
        # Init
        v = [random.randint(0, m-1) for _ in range(16)]
        curr_cost = cost(v)
        
        if curr_cost == 0:
            # Found immediately
            break
            
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        
        while T > T_min:
            if time.time() > deadline - 5:
                break
                
            # Pick random edge
            e = random.randint(0, 15)
            old_v = v[e]
            new_v = random.randint(0, m-1)
            v[e] = new_v
            
            new_cost = cost(v)
            delta = new_cost - curr_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                curr_cost = new_cost
            else:
                v[e] = old_v
                
            T *= alpha
            
        if curr_cost == 0:
            # Construct graph
            n = 8 * m
            edges = []
            # Vertex (side, base_idx, volt)
            # side 0: left, side 1: right
            # Base left: 0-3, Base right: 4-7
            # Edge (i, j+4) in base has voltage v[idx]
            # Lifted edge:
            # From (0, i, w) to (1, j, w + v[idx])
            
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    volt = v[idx]
                    for w in range(m):
                        u = i * m + w
                        v_dest = j * m + ((w + volt) % m)
                        # u is in left part (0..3m-1)
                        # v_dest is in right part (4m..7m-1)?
                        # Let's map:
                        # Left vertices: 0 .. 4m-1. (i, w) -> i*m + w
                        # Right vertices: 4m .. 8m-1. (j, w) -> 4m + j*m + w
                        u = i * m + w
                        v_dest = 4 * m + j * m + ((w + volt) % m)
                        edges.append((u, v_dest))
                        
            best_n = n
            best_edges = edges
            break
            
    if best_edges is None:
        # Fallback to LUW
        return _luw_fallback(k, g)
        
    return best_n, best_edges

def _luw_fallback(k, g):
    # Original LUW code
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
