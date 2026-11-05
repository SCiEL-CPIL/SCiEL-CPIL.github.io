import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # Known good construction: n=9 from the seed
    # Try to improve using a smarter search
    
    # For the C6-free 4-coloring problem, the key constraint is that
    # each color class must be C6-free. A graph is C6-free if it has no
    # cycle of length exactly 6.
    
    # Strategy: Use a greedy + local search approach
    # 1. Start with a random or structured coloring
    # 2. Check for C6 in each color class
    # 3. If found, recolor edges to break the cycle
    # 4. Repeat until no C6 or time limit
    
    # First, let's implement efficient C6 detection using bitmasks
    
    def find_c6(adj, n):
        """Find a C6 in graph adj (list of bitmasks) on n vertices.
        Returns tuple (v0,v1,v2,v3,v4,v5) or None."""
        for v0 in range(n):
            for v1 in range(n):
                if v1 == v0 or not (adj[v0] >> v1) & 1:
                    continue
                for v2 in range(n):
                    if v2 in (v0, v1) or not (adj[v1] >> v2) & 1:
                        continue
                    for v3 in range(n):
                        if v3 in (v0, v1, v2) or not (adj[v2] >> v3) & 1:
                            continue
                        for v4 in range(n):
                            if v4 in (v0, v1, v2, v3) or not (adj[v3] >> v4) & 1:
                                continue
                            # v5 adjacent to v4 and v0, not in {v0..v4}
                            cands = adj[v4] & adj[v0]
                            mask = (1 << n) - 1
                            for v in (v0, v1, v2, v3, v4):
                                mask &= ~(1 << v)
                            if cands & mask:
                                v5 = (cands & mask) & (-(cands & mask))
                                v5 = v5.bit_length() - 1
                                return (v0, v1, v2, v3, v4, v5)
        return None

    def count_c6s(adj, n, limit=10):
        """Count C6s up to limit for pruning."""
        count = 0
        for v0 in range(n):
            for v1 in range(n):
                if v1 == v0 or not (adj[v0] >> v1) & 1:
                    continue
                for v2 in range(n):
                    if v2 in (v0, v1) or not (adj[v1] >> v2) & 1:
                        continue
                    for v3 in range(n):
                        if v3 in (v0, v1, v2) or not (adj[v2] >> v3) & 1:
                            continue
                        for v4 in range(n):
                            if v4 in (v0, v1, v2, v3) or not (adj[v3] >> v4) & 1:
                                continue
                            cands = adj[v4] & adj[v0]
                            mask = (1 << n) - 1
                            for v in (v0, v1, v2, v3, v4):
                                mask &= ~(1 << v)
                            if cands & mask:
                                count += 1
                                if count >= limit:
                                    return count
        return count

    def edge_to_idx(i, j, n):
        """Convert (i,j) with i<j to linear index."""
        # Pairs before row i: sum_{k=0}^{i-1}(n-1-k) = i*(n-1) - i*(i-1)//2
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def idx_to_edge(idx, n):
        """Convert linear index to (i,j)."""
        cum = 0
        for i in range(n - 1):
            row_len = n - 1 - i
            if idx < cum + row_len:
                j = i + 1 + (idx - cum)
                return (i, j)
            cum += row_len
        return (n - 2, n - 1)

    def check_coloring(colors, n):
        """Check if coloring has no monochromatic C6. Return True if valid."""
        for c in range(4):
            adj = [0] * n
            for idx in range(len(colors)):
                if colors[idx] == c:
                    i, j = idx_to_edge(idx, n)
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            if find_c6(adj, n) is not None:
                return False
        return True

    def local_search(colors, n, max_iters=5000):
        """Local search: try to eliminate C6s by recoloring edges."""
        colors = colors[:]
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find a C6 in some color class
            found = False
            for c in range(4):
                adj = [0] * n
                for idx in range(len(colors)):
                    if colors[idx] == c:
                        i, j = idx_to_edge(idx, n)
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                
                cyc = find_c6(adj, n)
                if cyc is not None:
                    # Found C6: v0-v1-v2-v3-v4-v5-v0 in color c
                    # Try recoloring one of the 6 edges to a different color
                    edges_in_cycle = [(cyc[k], cyc[(k+1)%6]) for k in range(6)]
                    # Sort so i<j
                    edges_in_cycle = [tuple(sorted(e)) for e in edges_in_cycle]
                    
                    success = False
                    for (i, j) in edges_in_cycle:
                        idx = edge_to_idx(i, j, n)
                        old_c = colors[idx]
                        for new_c in range(4):
                            if new_c == old_c:
                                continue
                            # Try this recoloring
                            colors[idx] = new_c
                            # Check if this creates a C6 in new_c
                            adj_new = [0] * n
                            for idx2 in range(len(colors)):
                                if colors[idx2] == new_c:
                                    i2, j2 = idx_to_edge(idx2, n)
                                    adj_new[i2] |= (1 << j2)
                                    adj_new[j2] |= (1 << i2)
                            if find_c6(adj_new, n) is None:
                                # Also check that removing from old_c doesn't create issues
                                # (removing edges can't create C6s)
                                success = True
                                break
                            colors[idx] = old_c
                        if success:
                            break
                    if success:
                        found = True
                        break
            
            if not found:
                # No C6 found - valid!
                return colors
        
        return None

    # Start with structured colorings for various n
    best_n = 0
    best_colors = None

    # Try n from 17 down
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
        
        total_pairs = n * (n - 1) // 2
        
        # Try multiple initial colorings
        found = False
        
        # Strategy 1: Random initial coloring
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            colors = [random.randint(0, 3) for _ in range(total_pairs)]
            
            # Run local search
            result = local_search(colors, n, max_iters=2000)
            if result is not None:
                # Verify
                if check_coloring(result, n):
                    best_n = n
                    best_colors = result
                    found = True
                    break
        
        if found:
            break
        
        # Strategy 2: Structured coloring based on distances
        if n <= 17 and not found:
            # Use modular arithmetic coloring
            # Color edge (i,j) by (j-i) mod something
            for mod in range(4, 12):
                if time.time() > deadline:
                    break
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = j - i
                        colors.append(d % mod % 4)
                
                result = local_search(colors, n, max_iters=2000)
                if result is not None:
                    if check_coloring(result, n):
                        best_n = n
                        best_colors = result
                        found = True
                        break
        
        if found:
            break

    # Fallback to known n=9
    if best_n < 9:
        n = 9
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        best_n = n

    return {'n': best_n, 'colors': best_colors}
