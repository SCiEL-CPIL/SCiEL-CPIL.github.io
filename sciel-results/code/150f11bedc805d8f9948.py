import time
import random
import itertools

# Precompute all labeled Q3 bitmasks on vertices 0..7
# Q3 edges: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
# Total 12 edges.
# Vertices 0..7.
# We generate all 8! permutations of [0..7] and create the bitmask for each.
# Since Aut(Q3) has size 48, we will get 40320 / 48 = 840 unique masks.

def get_q3_masks():
    base_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    masks = set()
    for p in itertools.permutations(range(8)):
        # p[i] is the vertex in the permutation corresponding to the original vertex i?
        # Let's say original vertex u maps to new vertex p[u].
        # Edge (u,v) in original becomes (p[u], p[v]) in new.
        mask = 0
        for u, v in base_edges:
            nu = p[u]
            nv = p[v]
            if nu > nv:
                nu, nv = nv, nu
            # Bit position for edge (nu, nv) in 8x8 matrix
            # We use a 64-bit int for the 8x8 adjacency matrix (upper triangle or full?)
            # Let's use full 8x8 = 64 bits.
            # Bit index = nu * 8 + nv and nv * 8 + nu
            mask |= (1 << (nu * 8 + nv))
            mask |= (1 << (nv * 8 + nu))
        masks.add(mask)
    return list(masks)

Q3_MASKS = get_q3_masks()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_edges = 0
    best_graph = None
    
    # Graph representation: list of 16 ints, each 16 bits.
    # graph[i] has bit j set if edge (i,j) exists.
    
    def get_edge_count(graph):
        count = 0
        for i in range(16):
            count += bin(graph[i]).count('1')
        return count // 2

    def is_q3_free(graph):
        # Check all 8-vertex subsets
        for subset in itertools.combinations(range(16), 8):
            # Build 8x8 mask for this subset
            # Map subset[i] to i
            mask = 0
            # Optimize: if edges < 12, skip
            # Count edges in induced subgraph
            e_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    u = subset[i]
                    v = subset[j]
                    if (graph[u] >> v) & 1:
                        e_count += 1
                        mask |= (1 << (i * 8 + j))
                        mask |= (1 << (j * 8 + i))
                    else:
                        # If we are building mask incrementally, we need to be careful.
                        # Let's just build the full mask first.
                        pass
            
            if e_count < 12:
                continue
                
            # Re-build mask properly if e_count >= 12
            # Actually, let's just build the mask in the loop above and check count at the end?
            # The loop above is inefficient.
            
            # Better:
            mask = 0
            e_count = 0
            for i in range(8):
                u = subset[i]
                for j in range(i+1, 8):
                    v = subset[j]
                    if (graph[u] >> v) & 1:
                        e_count += 1
                        mask |= (1 << (i * 8 + j))
                        mask |= (1 << (j * 8 + i))
            
            if e_count < 12:
                continue
                
            # Check if mask is superset of any Q3 mask
            for q3_mask in Q3_MASKS:
                if (mask & q3_mask) == q3_mask:
                    return False, subset, q3_mask
        return True, None, None

    def try_add_edge(graph, u, v):
        if (graph[u] >> v) & 1:
            return True
        graph[u] |= (1 << v)
        graph[v] |= (1 << u)
        ok, _, _ = is_q3_free(graph)
        if not ok:
            graph[u] &= ~(1 << v)
            graph[v] &= ~(1 << u)
            return False
        return True

    # Main Loop: Multiple restarts
    while time.time() < deadline:
        # Start with a random dense graph
        p = 0.8
        graph = [0] * 16
        for u in range(16):
            for v in range(u+1, 16):
                if random.random() < p:
                    graph[u] |= (1 << v)
                    graph[v] |= (1 << u)
        
        # Repair phase: Remove edges to make it Q3-free
        while True:
            if time.time() > deadline:
                break
            ok, subset, q3_mask = is_q3_free(graph)
            if ok:
                break
            # Remove a random edge from the found Q3
            # We need to find which edges of the Q3 are present.
            # q3_mask is in local coordinates 0..7.
            # subset[i] is the global vertex for local i.
            edges_to_remove = []
            for i in range(8):
                for j in range(i+1, 8):
                    if (q3_mask >> (i * 8 + j)) & 1:
                        edges_to_remove.append((subset[i], subset[j]))
            
            # Pick a random edge from the Q3 to remove
            u, v = random.choice(edges_to_remove)
            graph[u] &= ~(1 << v)
            graph[v] &= ~(1 << u)
        
        if time.time() > deadline:
            break
            
        # Maximize phase: Add edges greedily
        # Shuffle edges to avoid bias
        all_edges = [(u, v) for u in range(16) for v in range(u+1, 16)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            try_add_edge(graph, u, v)
        
        # Record best
        current_edges = get_edge_count(graph)
        if current_edges > best_edges:
            best_edges = current_edges
            best_graph = [g for g in graph]
            
    # Return list of edges
    if best_graph is None:
        # Fallback to baseline if search failed
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            if (best_graph[u] >> v) & 1:
                edges.append([u, v])
    return edges
