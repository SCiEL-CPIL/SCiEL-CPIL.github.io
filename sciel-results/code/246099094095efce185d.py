import time
import numpy as np

def _paley_color(n, squares):
    """Generate Paley coloring for n vertices."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            colors.append(0 if d in squares else 1)
    return colors

def _has_k222(color, adj):
    """
    Check if color class contains K2,2,2.
    adj: list of bitmasks for color class.
    We need to find 6 vertices partitioned into 3 pairs
    such that all cross-edges between different pairs exist.
    
    Efficient check: for each pair (a,b) in color,
    for each pair (c,d) in color with c>a,
    check if there's a pair (e,f) in color with e>c
    such that e and f are both connected to a,b,c,d.
    
    Actually, let's use a more direct approach.
    K2,2,2 means: vertices a,b | c,d | e,f
    All edges between {a,b} and {c,d} exist
    All edges between {a,b} and {e,f} exist
    All edges between {c,d} and {e,f} exist
    
    For n=30, we can enumerate pairs.
    """
    n = len(adj)
    # Precompute all pairs in this color class
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                pairs.append((i, j))
    
    if len(pairs) < 3:
        return False
    
    # For each pair (a,b), compute common neighbors in the color class
    # that are not a or b
    # Then for each such common neighbor pair, etc.
    
    # More efficient: for each pair p1=(a,b), p2=(c,d) with a<b, c<d, c>a
    # Compute the intersection of neighbors of a,b,c,d (excluding a,b,c,d themselves)
    # If this intersection has at least 2 vertices, we found K2,2,2
    
    # Actually, we need e,f both connected to a,b AND both connected to c,d
    # So e,f must be in N(a,b) ∩ N(c,d) where N(x,y) = {z : x~z and y~z}
    
    # Let's precompute for each pair, the set of common neighbors
    pair_common = {}
    for (a, b) in pairs:
        common = adj[a] & adj[b]
        common &= ~(1 << a)
        common &= ~(1 << b)
        pair_common[(a, b)] = common
    
    # Now for each pair of pairs
    for idx1, (p1, p2) in enumerate([(pairs[i], pairs[j]) 
                                       for i in range(len(pairs)) 
                                       for j in range(i + 1, len(pairs))]):
        a, b = p1
        c, d = p2
        if c <= b:  # Ensure ordering to avoid duplicates
            continue
        
        # Common neighbors of a,b,c,d
        common_all = pair_common[p1] & pair_common[p2]
        common_all &= ~(1 << c)
        common_all &= ~(1 << d)
        
        # Need at least 2 vertices in common_all
        if common_all & (common_all - 1):  # At least 2 bits set
            return True
    
    return False

def _check_valid(n, colors):
    """Check if coloring is valid (no K2,2,2 in either color)."""
    # Build adjacency bitmasks for each color
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
    
    if _has_k222(0, adj0):
        return False
    if _has_k222(1, adj1):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s kill

    # Start with Paley(29)
    n = 29
    q = n
    squares = {x * x % q for x in range(1, q)}
    colors = _paley_color(n, squares)
    
    # Verify baseline
    if not _check_valid(n, colors):
        # Should not happen, but fallback
        raise ValueError("Paley(29) invalid!")
    
    best_n = n
    best_colors = colors
    
    # Try to extend to n=30 using local search
    # Strategy: add vertex 29, assign colors to its 29 edges,
    # then try to fix violations by flipping some edges.
    
    # Build adjacency for current 29-vertex solution
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
    
    # Try multiple random initial assignments for vertex 29
    max_attempts = 20
    rng = np.random.RandomState(42)
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        # Random initial coloring for edges from vertex 29
        new_colors = colors[:] + [int(rng.randint(2)) for _ in range(n)]
        new_n = n + 1
        
        # Build new adjacency
        new_adj0 = adj0[:] + [0]
        new_adj1 = adj1[:] + [0]
        
        idx29 = new_n - 1  # vertex 29
        for i in range(n):
            if new_colors[n*(n-1)//2 + i] == 0:
                new_adj0[i] |= (1 << idx29)
                new_adj0[idx29] |= (1 << i)
            else:
                new_adj1[i] |= (1 << idx29)
                new_adj1[idx29] |= (1 << i)
        
        # Local search: flip edges to eliminate K2,2,2
        # Simple approach: repeatedly find a violation and flip an edge in it
        improved = True
        max_iters = 500
        iters = 0
        
        while improved and iters < max_iters:
            if time.time() > deadline - 3:
                break
            improved = False
            iters += 1
            
            # Check for violations
            has0 = _has_k222(0, new_adj0)
            has1 = _has_k222(1, new_adj1)
            
            if not has0 and not has1:
                # Valid! Update best
                if new_n > best_n:
                    best_n = new_n
                    best_colors = new_colors[:]
                break
            
            # Find a violating K2,2,2 and flip a random edge in it
            # For simplicity, just flip a random edge connected to vertex 29
            # This is a crude heuristic but fast
            if has0:
                # Try flipping an edge in color 0 involving vertex 29
                flipped = False
                for i in range(n):
                    if new_adj0[idx29] & (1 << i):
                        # Flip edge (i, idx29) from 0 to 1
                        new_adj0[idx29] &= ~(1 << i)
                        new_adj0[i] &= ~(1 << idx29)
                        new_adj1[idx29] |= (1 << i)
                        new_adj1[i] |= (1 << idx29)
                        new_colors[n*(n-1)//2 + i] = 1
                        flipped = True
                        improved = True
                        break
                if not flipped:
                    # Flip a random internal edge
                    if n > 2:
                        i, j = int(rng.randint(n)), int(rng.randint(n))
                        if i == j:
                            continue
                        if i > j:
                            i, j = j, i
                        eidx = i * (n - 1) - i * (i + 1) // 2 + j - i - 1
                        # This index calculation is complex; skip for now
                        pass
            else:
                # Similar for color 1
                flipped = False
                for i in range(n):
                    if new_adj1[idx29] & (1 << i):
                        new_adj1[idx29] &= ~(1 << i)
                        new_adj1[i] &= ~(1 << idx29)
                        new_adj0[idx29] |= (1 << i)
                        new_adj0[i] |= (1 << idx29)
                        new_colors[n*(n-1)//2 + i] = 0
                        flipped = True
                        improved = True
                        break
                if not flipped:
                    pass
        
        # Check final validity of this attempt
        if _check_valid(new_n, new_colors):
            if new_n > best_n:
                best_n = new_n
                best_colors = new_colors[:]
                break
    
    return {'n': best_n, 'colors': best_colors}
