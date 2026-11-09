import time
import random
import sys

def _is_valid_coloring(n, colors):
    """
    Verify if the coloring is valid.
    colors is a list of length n*(n-1)/2.
    """
    if n < 2:
        return True
    
    # Build adjacency bitsets for each color
    # adj[c][u] is an integer bitmask of neighbors of u in color c
    adj = [[0] * n for _ in range(5)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            # Set bit j in adj[c][i] and bit i in adj[c][j]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    
    # Check for C4: for each color, no two vertices should have 2+ common neighbors
    for c in range(5):
        for u in range(n):
            mask_u = adj[c][u]
            # Iterate over set bits in mask_u
            # For each pair of neighbors v, w of u, check if they are connected?
            # No, C4 is u-v-w-x-u.
            # Equivalent: for any u, v, the number of common neighbors < 2.
            # Common neighbors of u and v: adj[c][u] & adj[c][v]
            # We need popcount(adj[c][u] & adj[c][v]) <= 1 for all u != v.
            
            # Optimization: 
            # For each u, consider pairs of neighbors.
            # If two neighbors v, w of u are connected to each other? No.
            # If v and w have a common neighbor x != u, then u-v-x-w-u is a C4?
            # No, u-v, v-x, x-w, w-u. Yes.
            # So for each u, and each pair of neighbors (v, w) in color c,
            # we must ensure that (adj[c][v] & adj[c][w]) & ~(1 << u) has popcount 0.
            # i.e., v and w have no common neighbor other than u.
            
            # This check is O(n^3) or O(n^2 * 2^d).
            # With bitsets, intersection is fast.
            
            # Let's iterate u, then v, then w?
            # Better: Iterate u, v.
            # common = adj[c][u] & adj[c][v]
            # if popcount(common) > 1: return False
            
            for v in range(n):
                if u == v: continue
                common = adj[c][u] & adj[c][v]
                # Check if popcount > 1
                # popcount in python: bin(x).count('1')
                # Or use a lookup table for 16-bit chunks?
                # n <= 29, so 29 bits.
                # bin(common).count('1') is fast enough for n=29?
                # 29*29*5 = 4205 intersections. Each intersection is 1 op.
                # Counting bits is fast.
                if bin(common).count('1') > 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    best_n = 2
    best_colors = [0] # n=2 is always valid (1 edge, 1 color)
    
    # We will try to build up to n=29.
    # Start with a known good small n or build from scratch.
    # Let's try to reach n=26.
    
    def try_build(n_target):
        """
        Attempts to build a valid coloring for K_n_target.
        Uses greedy with random restarts and local search for the last few vertices.
        """
        # We represent the coloring as a list of lists or a flat list.
        # For speed in search, we maintain adj bitsets.
        
        # State:
        # adj[c][u] = bitmask of neighbors
        # colors[i][j] = color
        
        # We build vertex by vertex.
        # To make it faster, we can fix the first 20 vertices using a known pattern?
        # Or just use a strong heuristic greedy.
        
        # Heuristic:
        # When adding vertex v, assign colors to (v, u) for u < v.
        # We want to minimize the number of "forbidden pairs" (pairs with 1 common neighbor).
        
        # Let's implement a recursive backtracking for the last few vertices
        # and greedy for the first ones.
        
        # Actually, a pure greedy with restarts often gets stuck around n=20-22.
        # To reach 26, we need a better method.
        
        # Method:
        # 1. Greedily build up to n=20.
        # 2. Use local search to add vertices 20..25.
        
        # Let's try a simpler approach: 
        # Use the algebraic construction for n=25 if we can hardcode it.
        # Since I can't, I will use a fast randomized search.
        
        # To improve success, we will use a "conflict-directed" backtracking for the last 5 vertices.
        
        pass

    # Since implementing a full SAT solver or complex backtracker is risky for time,
    # I will use a robust greedy with many restarts, aiming for n=26.
    # If it fails, it will return the best found.
    
    # We know n=26 is possible.
    # I will implement a specific heuristic: 
    # Order vertices 0..25.
    # For each new vertex, try all color permutations? No.
    
    # Let's try to construct n=25 using the finite field method properly.
    # Vectors in Z_5^2.
    # We need 5 sets S_0..S_4 of size 4 (or similar) that are Sidon.
    # Total 24 non-zero vectors.
    # If we can partition 24 vectors into 5 sets of 4, we are good.
    # 24 / 5 is not integer.
    # So one set must have size 5? Or some sets have different sizes?
    # If a set has size 5, it might not be Sidon.
    
    # Actually, the construction for R_5(C_4) > 25 uses 5 graphs.
    # I will use a pre-computed valid coloring for n=25 found in literature or generate one.
    # To be safe, I will generate a valid coloring for n=25 using a fast solver.
    
    # Let's use a simple backtracking for n=25.
    # It should be fast enough.
    
    def solve_n(n):
        # Returns colors list or None
        # Use backtracking with MRV heuristic
        # This might be too slow for n=25 in Python if not optimized.
        # However, n=25 is small.
        
        # We will use a greedy approach with randomization and restarts.
        # If it fails for n=25, we try n=24, etc.
        
        for attempt in range(100):
            if time.time() > deadline:
                return None
            
            adj = [[0] * n for _ in range(5)]
            colors = [[-1] * n for _ in range(n)]
            
            # Order of adding vertices
            # We add vertices 1 to n-1.
            
            success = True
            for v in range(1, n):
                # Determine colors for (v, u) for u in 0..v-1
                # We do this in random order
                us = list(range(v))
                random.shuffle(us)
                
                # For each u, we pick a color
                # We maintain a list of available colors?
                # No, we pick the color that creates the least conflicts.
                
                for u in us:
                    best_c = -1
                    best_score = float('inf')
                    
                    for c in range(5):
                        # Check if color c is valid for (v, u)
                        # Conflict if:
                        # 1. There exists w < v such that (v, w) is c and (u, w) is c?
                        #    No, that would mean v and u have common neighbor w in color c.
                        #    If they already have one, adding another creates C4?
                        #    Wait. If (v, w) is c and (u, w) is c, then w is a common neighbor.
                        #    If there is another common neighbor x, then C4.
                        #    So we must ensure that the number of common neighbors of v and u in color c is < 1?
                        #    No, it must be < 2.
                        #    Currently, common neighbors are adj[c][v] & adj[c][u].
                        #    If we add edge (v, u) with color c, does it change the common neighbors of other pairs?
                        #    It makes v and u adjacent.
                        #    For any pair (x, y), if v is in adj[c][x] and u is in adj[c][y]...
                        
                        # Let's re-evaluate the conflict condition.
                        # Adding edge (v, u) with color c:
                        # 1. v and u become neighbors in color c.
                        # 2. For any w, if w is neighbor of v in c, then (u, w) must not be a neighbor in c?
                        #    No. If w is neighbor of v in c, and we add (v, u) in c,
                        #    then u and w share neighbor v in color c.
                        #    So u and w must not have any OTHER common neighbor in color c.
                        #    i.e., (adj[c][u] & adj[c][w]) must be 0.
                        #    Because if there was x in adj[c][u] & adj[c][w], then u-x-w-v-u is a C4.
                        
                        # So the condition is:
                        # For all w in adj[c][v] (neighbors of v in c),
                        #   adj[c][u] & adj[c][w] must be 0.
                        # AND
                        # For all w in adj[c][u] (neighbors of u in c),
                        #   adj[c][v] & adj[c][w] must be 0.
                        
                        # Note: adj[c][v] only contains vertices < v (since we are building sequentially).
                        # adj[c][u] contains vertices < u.
                        
                        valid = True
                        # Check neighbors of v
                        mask_v = adj[c][v]
                        # Iterate bits in mask_v
                        m = mask_v
                        while m:
                            lsb = m & -m
                            w = lsb.bit_length() - 1
                            # Check if u and w have common neighbors in c
                            # Common neighbors of u and w: adj[c][u] & adj[c][w]
                            # Must be 0
                            if adj[c][u] & adj[c][w]:
                                valid = False
                                break
                            m ^= lsb
                            
                        if not valid:
                            continue
                        
                        # Check neighbors of u
                        mask_u = adj[c][u]
                        m = mask_u
                        while m:
                            lsb = m & -m
                            w = lsb.bit_length() - 1
                            # Check if v and w have common neighbors in c
                            # Common neighbors of v and w: adj[c][v] & adj[c][w]
                            # Must be 0
                            if adj[c][v] & adj[c][w]:
                                valid = False
                                break
                            m ^= lsb
                            
                        if valid:
                            # Score: number of "forbidden pairs" created
                            # A pair (x, y) is forbidden if they have exactly 1 common neighbor.
                            # Adding (v, u) in c creates new common neighbor relationships:
                            # For w in adj[c][v], (u, w) now has common neighbor v.
                            # If (u, w) had 0 common neighbors before, it now has 1.
                            # This is "bad" (forbidden pair).
                            # We want to minimize the number of such pairs.
                            
                            score = 0
                            # For w in adj[c][v]:
                            #   if (adj[c][u] & adj[c][w]) == 0 (which is true if valid),
                            #   then (u, w) becomes a forbidden pair.
                            #   So score += popcount(adj[c][v])?
                            #   Wait, adj[c][v] & adj[c][w] is 0.
                            #   So yes, every neighbor w of v in c creates a forbidden pair (u, w).
                            #   Similarly, every neighbor w of u in c creates a forbidden pair (v, w).
                            #   But we must avoid double counting?
                            #   (u, w) and (v, w) are different pairs.
                            
                            score = bin(adj[c][v]).count('1') + bin(adj[c][u]).count('1')
                            
                            if score < best_score or (score == best_score and random.random() < 0.5):
                                best_score = score
                                best_c = c
                                
                    if best_c == -1:
                        success = False
                        break
                    
                    # Assign color
                    colors[v][u] = best_c
                    colors[u][v] = best_c
                    adj[best_c][v] |= (1 << u)
                    adj[best_c][u] |= (1 << v)
                
                if not success:
                    break
            
            if success:
                # Flatten colors
                res = []
                for i in range(n):
                    for j in range(i + 1, n):
                        res.append(colors[i][j])
                return res
        
        return None

    # Try to find a valid coloring for n=26
    for n in range(26, 1, -1):
        if time.time() > deadline:
            break
        
        # Try multiple times
        found = None
        for _ in range(50):
            if time.time() > deadline:
                break
            found = solve_n(n)
            if found:
                if _is_valid_coloring(n, found):
                    best_n = n
                    best_colors = found
                    break
        
        if found:
            break

    if best_n < 2:
        best_n = 2
        best_colors = [0]

    return {'n': best_n, 'colors': best_colors}
