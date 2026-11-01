import time
import random
import numpy as np

def construct(k, g):
    # Target: 3-regular graph, girth >= 16
    # Strategy: Lift Heawood graph (girth 7, 14 vertices) by Z_L
    # Find smallest L such that there exists a voltage assignment with no non-backtracking
    # walk of length < 16 having voltage sum 0.
    
    deadline = time.time() + 110.0 # Safety margin
    
    # Heawood graph edges
    # Vertices 0..13
    # Heawood is the bipartite incidence graph of Fano plane
    # Let's define it explicitly
    heawood_edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (1, 6),
        (2, 4), (2, 7), (2, 8),
        (3, 5), (3, 7), (3, 9),
        (4, 10), (4, 11), (4, 12),
        (5, 10), (5, 13), (5, 9), # Wait, need to ensure 3-regular
        # Let's use a known list or generate it
    ]
    
    # Correct Heawood Graph Edges
    # It is the (3,7)-cage.
    # Vertices can be seen as points and lines of Fano plane.
    # Points: 0-6, Lines: 7-13.
    # Point i connected to Line j if i is on line j.
    # Fano plane lines:
    # 0: {0,1,2}
    # 1: {0,3,4}
    # 2: {0,5,6}
    # 3: {1,3,5}
    # 4: {1,4,6}
    # 5: {2,3,6}
    # 6: {2,4,5}
    
    edges = []
    for i in range(7):
        for j in range(7):
            # Check if point i is on line j
            # This logic is tedious, let's hardcode the adjacency
            pass
            
    # Hardcoded Heawood adjacency (3-regular, 14 vertices)
    adj = {
        0: [1, 2, 3],
        1: [0, 4, 5],
        2: [0, 6, 7],
        3: [0, 8, 9],
        4: [1, 10, 11],
        5: [1, 12, 13],
        6: [2, 10, 12],
        7: [2, 11, 13],
        8: [3, 10, 13],
        9: [3, 11, 12],
        10: [4, 6, 8],
        11: [4, 7, 9],
        12: [5, 6, 9],
        13: [5, 7, 8]
    }
    
    # Verify it's 3-regular and connected
    # It is.
    
    # Map edges to indices
    edge_list = []
    edge_index = {}
    for u in range(14):
        for v in adj[u]:
            if u < v:
                edge_list.append((u, v))
                edge_index[(u, v)] = len(edge_list) - 1
                edge_index[(v, u)] = len(edge_list) - 1
    
    m_edges = len(edge_list) # Should be 21
    
    # Precompute all non-backtracking walks of length 7 to 15
    # A walk is a sequence of vertices (v0, v1, ..., vk)
    # We store the sequence of edge indices.
    walks_by_length = {l: [] for l in range(7, 16)}
    
    # Start from each vertex
    for start in range(14):
        # Current path: list of (vertex, prev_vertex)
        # We need to track the last vertex to avoid backtracking
        # Actually, just track the sequence of vertices.
        
        # Recursive generation or iterative BFS
        # Since max length is 15, recursion depth is small.
        
        def gen_walks(current, prev, length, edge_seq):
            if length >= 7:
                if length < 16:
                    walks_by_length[length].append(tuple(edge_seq))
            if length >= 15:
                return
            for nxt in adj[current]:
                if nxt != prev:
                    # Add edge (current, nxt)
                    e_idx = edge_index[(current, nxt)]
                    gen_walks(nxt, current, length + 1, edge_seq + [e_idx])
        
        # Start with length 0? 
        # A walk of length 1 is just an edge.
        # We want walks of length L.
        # Let's initialize for length 1.
        for nxt in adj[start]:
            e_idx = edge_index[(start, nxt)]
            gen_walks(nxt, start, 1, [e_idx])
            
    # Total walks count
    total_walks = sum(len(w) for w in walks_by_length.values())
    
    # Try different L values
    # Start from small L to minimize n = 14 * L
    # Moore bound is 512. 14 * 37 = 518. 
    # But can we satisfy girth 16 with L=37?
    # Number of walks ~ 900k. 900k / 37 ~ 24k bad walks.
    # It might be possible, but hard.
    # Let's try L from 67 down? Or up?
    # Smaller L is better.
    # Let's try L in range [67, 60, 53, ...]
    # Actually, let's just iterate L from 67 downwards to 37.
    
    best_n = None
    best_edges = None
    
    # We need a fast check and repair.
    # State: array of voltages for m_edges edges.
    # Bad walks: list of edge indices.
    
    # To speed up, we can use a dictionary of bad walks?
    # No, just check all walks each time? 900k checks * 15 ops = 13.5M ops.
    # In Python, this is too slow per iteration.
    
    # Optimization:
    # Maintain a count of "active" bad walks?
    # Or, only check walks that were previously bad or affected by the change.
    # When we change voltage of edge e, only walks containing e can change status.
    # Let's precompute which walks contain which edge.
    
    edge_to_walks = {i: [] for i in range(m_edges)}
    for l in range(7, 16):
        for w in walks_by_length[l]:
            for e in w:
                edge_to_walks[e].append(w)
                
    # This might be large. 21 edges.
    # Average walks per edge: 900k / 21 ~ 43k.
    # So checking affected walks is ~43k checks. Much faster.
    
    # Initialize
    for L in range(67, 36, -1):
        if time.time() > deadline:
            break
            
        # Random initialization
        voltages = np.random.randint(0, L, m_edges)
        
        # Calculate initial bad walks
        # We need a data structure to quickly find if a walk is bad.
        # Let's just recompute sums for all walks initially? 
        # 900k walks * 15 edges = 13.5M additions. ~0.1s in NumPy?
        # We can vectorize this.
        
        # Represent walks as a matrix of shape (N_walks, 15) with -1 for padding?
        # Or just loop.
        
        # Let's use a list of bad walks.
        bad_walks = set()
        
        # To speed up initial check, we can use the fact that we just started.
        # But 900k is small.
        
        # Let's define a function to check a walk
        def is_bad(walk, vols):
            s = 0
            for e in walk:
                s += vols[e]
            return s % L == 0
            
        # Initial check
        for l in range(7, 16):
            for w in walks_by_length[l]:
                if is_bad(w, voltages):
                    bad_walks.add(w)
                    
        if not bad_walks:
            # Found a solution!
            # Construct the graph
            n = 14 * L
            graph_edges = []
            # Vertices are (v, x) where v in 0..13, x in 0..L-1
            # Edge (u, v) in base with voltage vol connects (u, x) to (v, x+vol)
            for u, v in edge_list:
                e_idx = edge_index[(u, v)]
                vol = voltages[e_idx]
                for x in range(L):
                    u_lift = u * L + x
                    v_lift = v * L + (x + vol) % L
                    # Ensure u_lift < v_lift for unique edge?
                    # The graph is simple.
                    graph_edges.append((u_lift, v_lift))
                    
            return n, graph_edges
            
        # Local Search
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if not bad_walks:
                # Success
                n = 14 * L
                graph_edges = []
                for u, v in edge_list:
                    e_idx = edge_index[(u, v)]
                    vol = voltages[e_idx]
                    for x in range(L):
                        u_lift = u * L + x
                        v_lift = v * L + (x + vol) % L
                        graph_edges.append((u_lift, v_lift))
                return n, graph_edges
                
            # Pick a random bad walk
            w = random.choice(list(bad_walks))
            
            # Pick a random edge in the walk
            e = random.choice(w)
            
            # Change voltage of e
            old_vol = voltages[e]
            new_vol = np.random.randint(0, L)
            if new_vol == old_vol:
                continue
                
            delta = (new_vol - old_vol) % L
            voltages[e] = new_vol
            
            # Update bad_walks
            # Only walks containing e are affected
            # For each walk in edge_to_walks[e]:
            #   Check if it was bad before, and if it is bad now.
            
            # We need to know the sum of the walk BEFORE the change.
            # This is expensive to recompute for every affected walk if we don't store sums.
            # Let's store sums? 
            # There are 900k walks. Storing sums is fine.
            # But updating them?
            
            # Alternative: 
            # Remove all walks in edge_to_walks[e] from bad_walks.
            # Then re-check them and add back if bad.
            
            affected = edge_to_walks[e]
            for w_aff in affected:
                bad_walks.discard(w_aff)
                
            for w_aff in affected:
                if is_bad(w_aff, voltages):
                    bad_walks.add(w_aff)
                    
    # If no solution found, return the best one found or a fallback
    # Fallback: Return the LUW graph from the seed? 
    # Or just return a random 3-regular graph? No, must be valid.
    # The seed program returns a valid graph.
    # Let's replicate the seed's logic as a fallback if search fails.
    
    # Fallback: Lazebnik-Ustimenko-Woldar
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
