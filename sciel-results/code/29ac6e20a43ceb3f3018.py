import time
import random

def _has_c5(n, adj):
    """Check if graph has a 5-cycle using adjacency list."""
    # For each vertex v1, enumerate paths v1-v2-v3-v4-v5 and check if v5 adjacent to v1
    # with all distinct vertices.
    for v1 in range(n):
        for v2 in adj[v1]:
            for v3 in adj[v2]:
                if v3 == v1:
                    continue
                for v4 in adj[v3]:
                    if v4 == v1 or v4 == v2:
                        continue
                    for v5 in adj[v4]:
                        if v5 == v1 or v5 == v2 or v5 == v3:
                            continue
                        if v5 in adj[v1]:
                            return True
    return False

def _alpha_ge_9(n, adj):
    """Check if independence number >= 9 using branch and bound with bitmasks."""
    # Build adjacency masks
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    def _branch(candidates, size):
        if size >= 9:
            return True
        if candidates == 0:
            return False
        # Upper bound
        if size + bin(candidates).count('1') < 9:
            return False
        
        # Pick vertex with highest degree in candidate subgraph for better pruning
        best_v = -1
        best_deg = -1
        temp = candidates
        while temp:
            v_bit = temp & (-temp)
            v_idx = v_bit.bit_length() - 1
            deg = bin(candidates & adj_mask[v_idx]).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v_idx
            temp ^= v_bit
        
        if best_v < 0:
            return False
            
        v_bit = 1 << best_v
        
        # Option 1: include best_v
        new_candidates = candidates & ~adj_mask[best_v] & ~v_bit
        if _branch(new_candidates, size + 1):
            return True
        
        # Option 2: exclude best_v
        if _branch(candidates & ~v_bit, size):
            return False
        
        return False
    
    all_mask = (1 << n) - 1
    return _branch(all_mask, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s/150s limit
    
    # Fallback: 8 disjoint K4s = 32 vertices
    n_fallback = 32
    edges_fallback = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges_fallback.append([base + i, base + j])
    
    # Try to find a 33-vertex solution
    # Strategy: start with 8 K4s, add vertex 32, add edges to reduce alpha below 9
    # while preserving C5-freeness.
    
    random.seed(42)
    
    # Precompute base edges (8 K4s)
    base_edges = set()
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                base_edges.add((base + i, base + j))
    
    # Key insight: To get alpha <= 8 with 33 vertices, we need to carefully
    # add vertex 32 and some inter-K4 edges.
    
    # Approach 1: Try adding vertex 32 connected to specific vertices in each K4,
    # plus some inter-K4 edges to break independent sets of size 9.
    
    # The 8 K4s have independent sets of size 8 (one from each K4).
    # Adding vertex 32: if 32 is not connected to any vertex, alpha becomes 9.
    # If 32 is connected to at least one vertex in each K4, then any independent set
    # containing 32 can have at most one vertex from each K4 (avoiding 32's neighbors).
    # So alpha with 32 included is at most 1 + 8 = 9 (still too high).
    # We need to further reduce by adding inter-K4 edges.
    
    # Let's try: connect 32 to exactly one vertex in each K4 (say vertex 0 of each K4).
    # Then add inter-K4 edges to ensure no independent set of size 9 exists.
    
    # For a C5-free graph, inter-K4 edges must be carefully placed.
    
    # Let's try a systematic search over connection patterns for vertex 32
    # and small sets of inter-K4 edges.
    
    k4_verts = [[4*b + i for i in range(4)] for b in range(8)]
    
    # Try different connection patterns for vertex 32
    # Pattern: for each K4 b, connect 32 to vertex k4_verts[b][pattern[b]]
    
    found = False
    best_edges = None
    
    # First, try simple patterns where 32 connects to one vertex per K4
    # and we add a few inter-K4 edges to break alpha=9
    
    num_patterns = 4 ** 8  # too many, so sample
    random_patterns = []
    for _ in range(5000):
        pattern = [random.randint(0, 3) for _ in range(8)]
        random_patterns.append(pattern)
    
    for pattern in random_patterns:
        if time.time() > deadline - 5.0:
            break
        
        # Build edge set: base K4s + 32 connected to k4_verts[b][pattern[b]] for each b
        edge_set = set(base_edges)
        for b in range(8):
            edge_set.add((32, k4_verts[b][pattern[b]]))
        
        # Check C5-free
        # Build adjacency
        n = 33
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        if _has_c5(n, adj):
            continue
        
        # Check alpha
        if _alpha_ge_9(n, adj):
            # Need to add more edges to reduce alpha
            # Try adding inter-K4 edges
            # An independent set of size 9 would be: 32 + one vertex from each K4 (not the neighbor of 32)
            # So we need to add edges between K4s to prevent this.
            
            # Find an independent set of size 9 to guide edge additions
            # For now, try adding random inter-K4 edges
            
            added = False
            for _ in range(100):
                if time.time() > deadline - 5.0:
                    break
                
                # Pick two random K4s and add an edge between them
                b1 = random.randint(0, 7)
                b2 = random.randint(0, 7)
                if b1 == b2:
                    continue
                v1 = random.choice(k4_verts[b1])
                v2 = random.choice(k4_verts[b2])
                
                if (v1, v2) in edge_set or (v2, v1) in edge_set:
                    continue
                
                edge_set.add((v1, v2))
                
                # Rebuild adj
                adj = [set() for _ in range(n)]
                for u, v in edge_set:
                    adj[u].add(v)
                    adj[v].add(u)
                
                if _has_c5(n, adj):
                    edge_set.discard((v1, v2))
                    continue
                
                if not _alpha_ge_9(n, adj):
                    found = True
                    best_edges = [[min(u,v), max(u,v)] for u, v in edge_set]
                    break
            
            if found:
                break
    
    if found and best_edges is not None:
        return {'n': 33, 'edges': best_edges}
    
    # Fallback to 32 vertices
    return {'n': n_fallback, 'edges': edges_fallback}
