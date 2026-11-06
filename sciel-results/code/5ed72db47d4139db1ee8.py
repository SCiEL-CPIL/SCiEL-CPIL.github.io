import time
import random
import itertools

def _has_k222_fast(adj, n):
    """
    Check if graph with adjacency bitmasks adj (list of ints) on n vertices 
    contains K2,2,2 as a subgraph.
    K2,2,2 exists if there are 4 distinct vertices a,b,c,d such that 
    their common neighborhood has size >= 2.
    """
    # Optimization: Precompute pair intersections? 
    # For n=30, n^4 is small enough to do on the fly or with small precomputation.
    # Let's just do the 4-nested loop with bitmask ANDs.
    
    # To speed up, we can iterate a < b < c < d
    for a in range(n):
        ma = adj[a]
        for b in range(a + 1, n):
            mab = ma & adj[b]
            for c in range(b + 1, n):
                mabc = mab & adj[c]
                for d in range(c + 1, n):
                    mabcd = mabc & adj[d]
                    if mabcd.bit_count() >= 2:
                        return True
    return False

def _is_valid_fast(colors, n):
    """
    Check if coloring is valid (no K2,2,2 in either color class).
    colors: list of 0/1 of length n*(n-1)//2
    """
    if len(colors) != n * (n - 1) // 2:
        return False
    
    adj0 = [0] * n
    adj1 = [0] * n
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
            
    if _has_k222_fast(adj0, n):
        return False
    if _has_k222_fast(adj1, n):
        return False
    return True

def _paley_colors(n, p):
    """
    Generate Paley coloring for n vertices using prime p.
    Returns list of 0/1.
    """
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            colors.append(0 if diff in squares else 1)
    return colors

def _local_search_30(start_colors, n, time_limit):
    """
    Attempt to fix a coloring of K_n to be valid.
    Uses a simple hill-climbing / random restart approach.
    """
    start_time = time.time()
    colors = start_colors[:]
    
    # Initial check
    if _is_valid_fast(colors, n):
        return colors
    
    # Count violations? 
    # A "violation" is a K2,2,2. 
    # We can define a penalty function: number of K2,2,2 subgraphs in both colors.
    # But counting is expensive. 
    # Instead, we can pick a random "bad" 4-tuple (a,b,c,d) that causes a violation 
    # (i.e., common neighbors >= 2) and flip an edge involved in that structure.
    
    # Simpler: Random flips.
    # Given the small search space of edges (435 for n=30), random flips might work if we guide them.
    
    # Let's try a guided local search:
    # 1. Find all "bad" 4-tuples for color 0 and color 1.
    # 2. For each bad tuple, the common neighbors e,f form the third part.
    #    The edges to flip could be any edge incident to a,b,c,d or e,f.
    #    Flipping an edge in the "cross" set will break the K2,2,2.
    
    for it in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # Check validity
        if _is_valid_fast(colors, n):
            return colors
        
        # Find a violation to repair
        # We need a function to find one bad 4-tuple quickly.
        # Re-use the logic from _has_k222_fast but return the tuple.
        bad_found = False
        # Check color 0
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        
        # Find bad 4-tuple in color 0
        found_tuple = None
        for a in range(n):
            ma = adj0[a]
            for b in range(a + 1, n):
                mab = ma & adj0[b]
                for c in range(b + 1, n):
                    mabc = mab & adj0[c]
                    for d in range(c + 1, n):
                        mabcd = mabc & adj0[d]
                        if mabcd.bit_count() >= 2:
                            found_tuple = (a, b, c, d, mabcd, 0)
                            break
                    if found_tuple: break
                if found_tuple: break
            if found_tuple: break
        
        if not found_tuple:
            # Check color 1
            for a in range(n):
                ma = adj1[a]
                for b in range(a + 1, n):
                    mab = ma & adj1[b]
                    for c in range(b + 1, n):
                        mabc = mab & adj1[c]
                        for d in range(c + 1, n):
                            mabcd = mabc & adj1[d]
                            if mabcd.bit_count() >= 2:
                                found_tuple = (a, b, c, d, mabcd, 1)
                                break
                        if found_tuple: break
                    if found_tuple: break
                if found_tuple: break
            
        if not found_tuple:
            # Should not happen if _is_valid_fast returned False, but just in case
            return colors
        
        a, b, c, d, common_mask, color = found_tuple
        
        # Get two vertices e, f from common_mask
        # Pick random e, f from the set
        verts = []
        mask = common_mask
        while mask:
            lsb = mask & -mask
            idx_v = lsb.bit_length() - 1
            verts.append(idx_v)
            mask ^= lsb
        
        if len(verts) < 2:
            # Fallback random flip
            idx_flip = random.randint(0, len(colors) - 1)
            colors[idx_flip] = 1 - colors[idx_flip]
            continue
            
        e, f = random.sample(verts, 2)
        
        # The K2,2,2 involves vertices {a,b}, {c,d}, {e,f}.
        # Edges that are part of the subgraph (color == 'color'):
        # a-c, a-d, b-c, b-d, a-e, a-f, b-e, b-f, c-e, c-f, d-e, d-f
        # We should flip one of these edges to break the subgraph.
        # Flipping an edge changes its color, so it's no longer present in 'color'.
        
        # Candidate edges to flip (indices in the flat list)
        # Helper to get index
        def get_idx(u, v):
            if u > v: u, v = v, u
            return u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)
            # Formula: sum_{k=0}^{u-1} (n-1-k) + (v-u-1)
            # Let's use a simpler mapping or precomputed list.
            # For n=30, 435 edges. Precomputation is cheap.
        
        # Precompute edge indices for speed
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
                
        candidates = [
            (a, c), (a, d), (b, c), (b, d),
            (a, e), (a, f), (b, e), (b, f),
            (c, e), (c, f), (d, e), (d, f)
        ]
        
        # Pick a random candidate edge to flip
        # Ensure the edge currently has the 'color' value (it should, by construction of the violation)
        valid_candidates = []
        for u, v in candidates:
            if u > v: u, v = v, u
            if (u, v) in edge_map:
                idx_e = edge_map[(u, v)]
                if colors[idx_e] == color:
                    valid_candidates.append(idx_e)
        
        if valid_candidates:
            idx_flip = random.choice(valid_candidates)
            colors[idx_flip] = 1 - colors[idx_flip]
        else:
            # Fallback
            idx_flip = random.randint(0, len(colors) - 1)
            colors[idx_flip] = 1 - colors[idx_flip]

    return colors if _is_valid_fast(colors, n) else None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Try to find n=30
    # Strategy A: Local search on random n=30
    # Strategy B: Local search on extended Paley 29 (add vertex 29, connect to 0..28)
    
    best_n = 0
    best_colors = None
    
    # First, secure n=29 with Paley
    p = 29
    colors_29 = _paley_colors(29, 29)
    if colors_29 and _is_valid_fast(colors_29, 29):
        best_n = 29
        best_colors = colors_29[:]
    
    # Attempt n=30
    # Use the saved time for search
    time_spent = time.time() - start_time
    remaining = deadline - time.time()
    
    if remaining > 5.0:
        # Try a few random restarts for n=30
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize
            if attempt % 2 == 0:
                # Random
                init_colors = [random.randint(0, 1) for _ in range(30 * 29 // 2)]
            else:
                # Based on Paley 29
                # Take Paley 29 colors, add vertex 29.
                # Connect 29 to others based on Paley rule if we had p=31? No, p=29 doesn't cover 29.
                # Just randomize edges incident to 29.
                init_colors = colors_29[:]
                # Edges involving 29: (29, 0), (29, 1), ..., (29, 28)
                # Indices: 
                # (29, j) is the last group?
                # Lexicographic order: (0,1)..(0,28), (1,2)..(1,28), ..., (28,29)
                # Total edges: 435.
                # Edges with 29:
                # (0,29) is at index: sum_{i=0}^{28} (29-i) - 1? 
                # Let's just append 30 random bits? No, we need to insert them in the correct lexicographic position.
                # Easier: build the matrix, then flatten.
                mat = [[0]*30 for _ in range(30)]
                idx = 0
                for i in range(29):
                    for j in range(i+1, 29):
                        mat[i][j] = colors_29[idx]
                        mat[j][i] = colors_29[idx]
                        idx += 1
                for j in range(29):
                    mat[29][j] = random.randint(0, 1)
                    mat[j][29] = mat[29][j]
                
                init_colors = []
                for i in range(30):
                    for j in range(i+1, 30):
                        init_colors.append(mat[i][j])
            
            # Run local search with limited time
            time_left_for_search = max(1.0, min(5.0, deadline - time.time() - 1.0))
            result = _local_search_30(init_colors, 30, time_left_for_search)
            
            if result:
                return {'n': 30, 'colors': result}
    
    # If n=30 not found, return best known (29)
    if best_colors is None:
        # Fallback to n=17 if 29 failed for some reason
        p = 17
        best_n = 17
        best_colors = _paley_colors(17, 17)
        
    return {'n': best_n, 'colors': best_colors}
