import time
import random

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    
    Returns:
        list of edges [u, v]
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    best_edges = None
    best_score = -1
    
    # Construct the Clebsch graph
    # Vertices are 0..15, represented as 5-bit strings with MSB 0.
    # u, v are adjacent if Hamming distance of their 5-bit reps is 1 or 4.
    
    def get_5bit_rep(i):
        # i is 0..15. Represent as 5 bits: 0 b3 b2 b1 b0
        # So the bits are [0, b3, b2, b1, b0]
        # We can just use i as the lower 4 bits and prepend 0.
        # But for Hamming distance, we need the full 5-bit string.
        # Let's create a list of 5-bit strings for 0..15.
        pass

    # Precompute 5-bit representations
    reps = []
    for i in range(n):
        # i is 4-bit. We prepend 0.
        # So the 5-bit integer is just i.
        # Wait, if i=1 (0001), 5-bit is 00001.
        # If i=15 (1111), 5-bit is 01111.
        # This works.
        reps.append(i)
        
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            # Calculate Hamming distance between u and v as 5-bit numbers
            # Since u, v < 16, they fit in 5 bits.
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist == 1 or dist == 4:
                edges.append([u, v])
                
    # Verify it's triangle free and not 3-colorable
    # For speed, we assume it is correct based on theory.
    # But let's do a quick check to be safe.
    
    def is_triangle_free(edges_list):
        adj = [[0]*n for _ in range(n)]
        for u, v in edges_list:
            adj[u][v] = 1
            adj[v][u] = 1
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True

    def is_3_colorable_bitmask(adj_masks):
        colors = [-1] * n
        order = sorted(range(n), key=lambda v: bin(adj_masks[v]).count('1'), reverse=True)
        
        def bt(pos):
            if pos == n:
                return True
            v = order[pos]
            used = 0
            mask = adj_masks[v]
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if colors[u] != -1:
                    used |= (1 << colors[u])
                m ^= lsb
                
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if bt(pos + 1):
                        return True
                    colors[v] = -1
            return False
            
        return bt(0)

    def edges_to_masks(edges):
        masks = [0] * n
        for u, v in edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks

    if not is_triangle_free(edges):
        # Fallback to Grötzsch if something went wrong (should not happen)
        groetzsch_edges = [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
            [5, 1], [5, 4],
            [6, 0], [6, 2],
            [7, 1], [7, 3],
            [8, 2], [8, 4],
            [9, 3], [9, 0],
            [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]
        ]
        return groetzsch_edges

    masks = edges_to_masks(edges)
    if is_3_colorable_bitmask(masks):
        # If it is 3-colorable, it's invalid.
        # Clebsch is 4-chromatic, so this should be False.
        # Fallback to Grötzsch
        groetzsch_edges = [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
            [5, 1], [5, 4],
            [6, 0], [6, 2],
            [7, 1], [7, 3],
            [8, 2], [8, 4],
            [9, 3], [9, 0],
            [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]
        ]
        return groetzsch_edges

    best_edges = edges
    best_score = len(edges)
    
    # Local search to see if we can add more edges
    # Clebsch is maximal triangle-free, so this should not improve.
    # But we run it just in case.
    
    def local_search(masks, edges_list):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = []
            for i in range(n):
                for j in range(i+1, n):
                    if not (masks[i] & (1 << j)):
                        candidates.append((i, j))
            
            if not candidates:
                break
                
            random.shuffle(candidates)
            
            for u, v in candidates:
                if masks[u] & (1 << v):
                    continue
                
                if masks[u] & masks[v]:
                    continue
                
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
                if not is_3_colorable_bitmask(masks):
                    edges_list.append([u, v])
                    improved = True
                else:
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    
        return masks, edges_list, improved

    current_masks, current_edges, improved = local_search(masks, edges[:])
    
    if len(current_edges) > best_score:
        best_score = len(current_edges)
        best_edges = current_edges[:]

    return best_edges
