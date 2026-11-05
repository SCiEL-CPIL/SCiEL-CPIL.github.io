import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Fast 3-colorability check using backtracking and bitmasks
    # adj is a list of bitmasks
    def is_3_colorable(adj):
        # Order vertices by degree descending for better pruning
        # We need to map original indices to sorted order
        # But for n=16, simple recursion on 0..15 with bitmask checks is fast enough
        # Let's use a recursive function with memoization or just simple backtracking
        # Simple backtracking:
        
        # colors[i] = -1 (uncolored), 0, 1, 2
        colors = [-1] * n
        
        # Heuristic: color vertices in order of decreasing degree
        # But we need to map back to original indices for adjacency checks
        # Let's just use original indices but pick the next vertex with the most constrained color?
        # For n=16, even simple 0..15 order is likely fast enough if we use bitmasks for neighbor conflicts.
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = idx
            # Determine available colors
            # A color c is invalid if any neighbor has color c
            # We can precompute for each vertex and each color, a bitmask of neighbors?
            # Or just check neighbors.
            
            # Optimization: Check if any neighbor is colored with c
            # neighbors of v are bits set in adj[v]
            
            # Try color 0
            # Check if any neighbor has color 0
            # We can maintain a bitmask for each color?
            # Let's stick to simple check first, n is small.
            
            for c in range(3):
                conflict = False
                # Check neighbors of v
                # Iterate through bits in adj[v]
                mask = adj[v]
                while mask:
                    # Get lowest set bit
                    lsb = mask & -mask
                    u = lsb.bit_length() - 1
                    if colors[u] == c:
                        conflict = True
                        break
                    mask -= lsb
                
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            
            return False
        
        return backtrack(0)

    # Randomized Greedy Construction
    # We run multiple trials
    # Each trial:
    # 1. Shuffle edges
    # 2. Greedily add edges if no triangle is formed
    # 3. Check 3-colorability
    # 4. Update best if valid and denser
    
    # To improve efficiency, we can use a slightly smarter greedy:
    # Prioritize edges that connect vertices with high degrees?
    # But random shuffle is simple and effective for finding diverse maximal graphs.
    
    while time.time() < deadline:
        # Reset adjacency
        adj = [0] * n
        edge_count = 0
        
        # Shuffle edges for this trial
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if edge (u, v) can be added
            # 1. Not already present (implied by greedy construction from scratch)
            # 2. No common neighbor (triangle-free check)
            # Common neighbor exists if (adj[u] & adj[v]) != 0
            if (adj[u] & adj[v]) == 0:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        # Check if this maximal triangle-free graph is 3-colorable
        # If it is 3-colorable, it is NOT a valid candidate (we need chi >= 4)
        # If it is NOT 3-colorable, it is a valid candidate.
        
        # Optimization: Before running full backtracking, we can check for simple properties.
        # But backtracking on 16 nodes is very fast.
        
        if not is_3_colorable(adj):
            if edge_count > best_count:
                best_count = edge_count
                # Reconstruct edge list from adj
                best_edges = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            best_edges.append([u, v])
                        mask -= lsb

    # Fallback: If no 4-chromatic graph was found (unlikely, but possible if all random maximal graphs were 3-colorable),
    # return the Clebsch graph (40 edges) as a safe valid solution.
    if best_count == 0:
        # Construct Clebsch Graph
        clebsch_edges = []
        for u in range(n):
            for bit_idx in range(4):
                v = u ^ (1 << bit_idx)
                if u < v:
                    clebsch_edges.append([u, v])
            v = u ^ 15
            if u < v:
                clebsch_edges.append([u, v])
        best_edges = clebsch_edges

    return best_edges
