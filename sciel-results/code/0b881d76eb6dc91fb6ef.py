import time
import itertools
import numpy as np

def construct():
    """
    Constructs a 2-coloring of K_n without a monochromatic K_{2,2,2}.
    Aims for n=30, which is the maximum known (R(K_{2,2,2})=31).
    Uses the Paley graph of order 31 minus one vertex as the primary candidate.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # 1. Generate Paley graph coloring for n=31
    q = 31
    # Quadratic residues mod 31
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    # Map vertices 0..30 to 0..30
    # We will remove vertex 0, so we keep vertices 1..30
    # Relabel vertices 1..30 to 0..29 for the output
    n_target = 30
    vertices = list(range(1, q)) # [1, 2, ..., 30]
    
    colors = []
    for i in range(n_target):
        u = vertices[i]
        for j in range(i + 1, n_target):
            v = vertices[j]
            diff = (v - u) % q
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
    
    # 2. Verify the coloring
    def is_valid(n, colors):
        # Build adjacency bitmasks for both colors
        # adj[0] is list of bitmasks for color 0
        # adj[1] is list of bitmasks for color 1
        adj = [[0] * n for _ in range(2)]
        
        # Fill adjacency matrices
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        # Check for K_{2,2,2} in both colors
        for c in range(2):
            A = adj[c]
            # Iterate over all pairs (u, v) to form the first part of the tripartition
            for u in range(n):
                for v in range(u + 1, n):
                    # S is the set of vertices w such that w is connected to both u and v
                    # w must be distinct from u and v
                    s_mask = (A[u] & A[v]) & ~(1 << u) & ~(1 << v)
                    
                    # If |S| < 4, cannot form two more pairs
                    if s_mask.bit_count() < 4:
                        continue
                    
                    # Extract vertices in S
                    # We need to check if the induced subgraph on S contains a K_{2,2}
                    # K_{2,2} in S means there exist a,b in S such that |N(a) \cap N(b) \cap S| >= 2
                    
                    # Iterate over pairs (a, b) in S
                    # We can do this by iterating over all pairs in S
                    # To optimize, we can just check all pairs
                    s_vertices = []
                    mask = s_mask
                    while mask:
                        # Get lowest set bit
                        lsb = mask & -mask
                        idx_v = lsb.bit_length() - 1
                        s_vertices.append(idx_v)
                        mask ^= lsb
                    
                    # Check for K_{2,2} in s_vertices
                    # For each pair (a, b) in s_vertices, check if they share at least 2 neighbors in s_vertices
                    len_s = len(s_vertices)
                    found_k22 = False
                    for i_s in range(len_s):
                        a = s_vertices[i_s]
                        for j_s in range(i_s + 1, len_s):
                            b = s_vertices[j_s]
                            # Common neighbors of a and b in S
                            common = (A[a] & A[b]) & s_mask
                            if common.bit_count() >= 2:
                                found_k22 = True
                                break
                        if found_k22:
                            break
                    if found_k22:
                        return False
        return True

    if is_valid(n_target, colors):
        return {'n': n_target, 'colors': colors}
    
    # 3. Fallback: If n=30 fails (unlikely), try local search or return n=17
    # Given the strong evidence that Paley(31)-minus-vertex is valid, this block is a safety net.
    # We will try to improve a random n=30 coloring or use n=17 Paley.
    
    # Try to repair n=30 via simulated annealing if time permits
    if time.time() < deadline - 20:
        # Simple local search: flip edges to reduce number of K_{2,2,2}
        # Count K_{2,2,2} is expensive, so we use a heuristic: minimize "cliques" or use the same check
        # However, implementing a full SA for this complex constraint is risky in remaining time.
        # Let's try removing a different vertex from Paley(31)
        for remove_v in range(1, q):
            if remove_v == 0:
                continue
            vertices = [v for v in range(q) if v != remove_v]
            colors_try = []
            for i in range(n_target):
                u = vertices[i]
                for j in range(i + 1, n_target):
                    v = vertices[j]
                    diff = (v - u) % q
                    if diff in residues:
                        colors_try.append(0)
                    else:
                        colors_try.append(1)
            
            if is_valid(n_target, colors_try):
                return {'n': n_target, 'colors': colors_try}
            
            if time.time() > deadline - 5:
                break
    
    # Fallback to known valid n=17
    n_fallback = 17
    residues_17 = set()
    for x in range(1, 17):
        residues_17.add((x * x) % 17)
    
    colors_fallback = []
    for i in range(n_fallback):
        for j in range(i + 1, n_fallback):
            diff = (j - i) % 17
            if diff in residues_17:
                colors_fallback.append(0)
            else:
                colors_fallback.append(1)
                
    return {'n': n_fallback, 'colors': colors_fallback}
