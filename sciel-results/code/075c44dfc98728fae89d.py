import time
import random
import sys

def _validate(n, colors):
    """
    Check if the coloring is valid (no monochromatic C4).
    colors is a list of length n*(n-1)//2, color of edge (i, j) with i < j.
    """
    if len(colors) != n * (n - 1) // 2:
        return False
    
    # Build adjacency bitmasks for each color
    # adj[c][i] is a bitmask of neighbors of i in color c
    adj = [[0] * n for _ in range(5)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 4:
                return False
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    # Check for C4
    # A C4 in color c exists if there are i, j, k, l distinct such that
    # i~j, j~k, k~l, l~i in color c.
    # Equivalent: two vertices i, j have at least 2 common neighbors in color c.
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                # Common neighbors of i and j in color c
                common = adj[c][i] & adj[c][j]
                if common != 0:
                    # Check if there are at least 2 bits set
                    if common & (common - 1) != 0:
                        return False
    return True

def construct():
    start_time = time.time()
    limit = 115.0  # 150s limit, keep 35s margin
    
    # Start with known valid n=11
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find larger n
    # We will attempt to build colorings for n = 12, 13, ...
    # Using a randomized greedy approach with backtracking (limited depth)
    
    for target_n in range(12, 27):
        if time.time() - start_time > limit:
            break
            
        found = False
        # Try a few random restarts for this target_n
        for restart in range(20):
            if time.time() - start_time > limit:
                break
                
            # Initialize colors with -1
            colors = [-1] * (target_n * (target_n - 1) // 2)
            
            # Order of edges to process
            # Randomize order to escape local minima
            edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
            random.shuffle(edges)
            
            # Greedy assignment with consistency check
            # We maintain the adjacency bitmasks
            adj = [[0] * target_n for _ in range(5)]
            
            valid_assignment = True
            for i, j in edges:
                # Try each color
                random.shuffle(range(5))
                assigned = False
                for c in range(5):
                    # Check if adding (i, j) with color c creates a C4
                    # Common neighbors of i and j in color c
                    common = adj[c][i] & adj[c][j]
                    # If common has >= 2 bits, adding edge (i,j) might not create C4 immediately?
                    # Wait. A C4 is a cycle of length 4.
                    # If i and j already have 2 common neighbors u, v in color c,
                    # then i-u-j-v-i is a C4? No, that requires edges (i,u), (u,j), (j,v), (v,i).
                    # We are adding (i,j).
                    # If we add (i,j), we create a C4 if there exist u, v such that
                    # (i,u), (u,v), (v,j) are in color c? No.
                    # The condition for NO C4 is: for any pair (i,j), the number of common neighbors
                    # in color c must be < 2?
                    # No. If i and j have 2 common neighbors u, v, then we have a K_{2,2} subgraph.
                    # The edges are (i,u), (i,v), (j,u), (j,v).
                    # This forms a C4: i-u-j-v-i.
                    # So if we add edge (i,j), it doesn't directly create a C4 with u,v
                    # UNLESS the edges (i,u), (i,v), (j,u), (j,v) are ALREADY present.
                    # But we are building the graph.
                    # If (i,u), (i,v), (j,u), (j,v) are already colored c, then we ALREADY have a C4.
                    # So we must ensure that BEFORE adding (i,j), i and j do NOT have 2 common neighbors in color c.
                    # Wait, if they DO have 2 common neighbors, the C4 already exists.
                    # So we must reject color c if common has >= 2 bits.
                    
                    if common & (common - 1) != 0:
                        continue
                        
                    # Also, adding (i,j) might create a new C4 involving (i,j) itself?
                    # A C4 involving (i,j) would be i-j-k-l-i.
                    # This requires (j,k), (k,l), (l,i) in color c.
                    # This is equivalent to saying j and i have a common neighbor k? No.
                    # It means there is a path of length 3 from i to j in color c?
                    # No, a C4 is a cycle.
                    # If we add (i,j), we close a cycle if there is a path of length 3 from i to j.
                    # Path i-u-v-j. Edges (i,u), (u,v), (v,j).
                    # This requires u to be a neighbor of i, v to be a neighbor of j, and (u,v) in color c.
                    # This is harder to check incrementally.
                    
                    # However, the condition "no two vertices have 2 common neighbors" is NECESSARY and SUFFICIENT
                    # for the graph to be C4-free?
                    # Yes. A graph is C4-free iff no two vertices have 2 common neighbors.
                    # So we just need to ensure that for every pair (i,j), the number of common neighbors in color c is < 2.
                    # This must hold for the FINAL graph.
                    # During construction, if we ensure that at every step, no pair has 2 common neighbors,
                    # then the final graph will be C4-free.
                    
                    # So the check `if common & (common - 1) != 0: continue` is sufficient.
                    
                    # Assign color c
                    colors[edges.index((i, j))] = c  # This is slow, use index mapping
                    # Better: keep a mapping
                    # For now, let's just use the index in the edges list
                    # We need to map (i,j) to index in `colors`
                    # idx = i * (target_n - 1) - i * (i + 1) // 2 + j - i - 1
                    idx = i * (target_n - 1) - i * (i + 1) // 2 + j - i - 1
                    colors[idx] = c
                    
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    
                    assigned = True
                    break
                    
                if not assigned:
                    valid_assignment = False
                    break
            
            if valid_assignment:
                # Verify
                if _validate(target_n, colors):
                    best_n = target_n
                    best_colors = colors
                    found = True
                    break # Move to next n
        
        # If we found a valid coloring for this n, we can try to go higher
        # If not, we might stop or continue trying higher n with more restarts?
        # Usually if n fails, n+1 might also fail. But we can try a few more.
        if not found and target_n > 13:
            # If we failed on 14, 15, maybe 16 is too hard.
            # But let's try up to 16.
            pass
            
    return {'n': best_n, 'colors': best_colors}
