import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Construct initial 45-edge graph
    edges = []
    # Grötzsch graph edges (0-10)
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    for i in range(5):
        edges.append([i, 5 + (i + 1) % 5])
        edges.append([i, 5 + (i - 1) % 5])
    for i in range(5):
        edges.append([10, 5 + i])
        
    # Max independent set of Grötzsch is {5,6,7,8,9} (the outer cycle vertices)
    # Wait, is {5..9} independent?
    # 5 connected to 4,0,10.
    # 6 connected to 0,1,10.
    # 5 and 6 are not connected.
    # Yes, {5,6,7,8,9} is independent.
    max_iset = [5, 6, 7, 8, 9]
    
    for new_v in range(11, 16):
        for v_i in max_iset:
            edges.append([new_v, v_i])

    n = 16
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute masks for fast 3-color check
    adj_masks = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_masks[u] = mask

    # 2. Local Search
    # List of all possible non-edges
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to avoid bias? Or just iterate.
        # We need to re-calculate all_pairs after each addition? 
        # Or just iterate through the original list and check if edge exists.
        # Better: maintain a list of candidates.
        
        # Simple approach: iterate through all pairs, if not edge, try to add.
        # To be efficient, we can just loop through the static list of non-edges from the start.
        # But if we add an edge, it's no longer a non-edge.
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if v in adj[u]:
                continue
                
            # Check triangle
            creates_triangle = False
            # Common neighbors
            # Use masks for speed
            common = adj_masks[u] & adj_masks[v]
            if common:
                creates_triangle = True
            
            if creates_triangle:
                continue
                
            # Tentatively add
            adj[u].add(v)
            adj[v].add(u)
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
            # Check 3-colorability
            if is_3_colorable_fast(adj_masks, n):
                # Remove
                adj[u].remove(v)
                adj[v].remove(u)
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
            else:
                # Keep
                improved = True
                # Optional: break and restart? Or continue?
                # Continuing is fine.
                
    # Return edges
    final_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                final_edges.append([u, v])
    return final_edges

def is_3_colorable_fast(adj_masks, n):
    # Order by degree
    deg = [bin(m).count('1') for m in adj_masks]
    order = sorted(range(n), key=lambda x: -deg[x])
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
        
    # Earlier masks
    earlier = [0] * n
    for v in range(n):
        m = 0
        mv = adj_masks[v]
        while mv:
            lsb = mv & (-mv)
            w = lsb.bit_length() - 1
            if pos[w] < pos[v]:
                m |= (1 << w)
            mv ^= lsb
        earlier[v] = m
        
    color = [-1] * n
    
    def backtrack(i):
        if i == n:
            return True
        v = order[i]
        forbidden = 0
        m = earlier[v]
        while m:
            lsb = m & (-m)
            w = lsb.bit_length() - 1
            if color[w] != -1:
                forbidden |= (1 << color[w])
            m ^= lsb
            
        for c in range(3):
            if not (forbidden & (1 << c)):
                color[v] = c
                if backtrack(i + 1):
                    return True
                color[v] = -1
        return False
        
    return backtrack(0)
