import time
import random

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    def validate(n, colors):
        """Check if coloring is valid (no C4 in any color class)."""
        m = n * (n - 1) // 2
        if len(colors) != m:
            return False
        for c in range(5):
            adj = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj[i].add(j)
                        adj[j].add(i)
                    idx += 1
            for i in range(n):
                for j in range(i + 1, n):
                    if len(adj[i] & adj[j]) >= 2:
                        return False
        return True
    
    def count_violations(n, colors):
        """Count number of violating pairs (i,j) that have >=2 common neighbors in some color."""
        total = 0
        for c in range(5):
            adj = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj[i].add(j)
                        adj[j].add(i)
                    idx += 1
            for i in range(n):
                for j in range(i + 1, n):
                    if len(adj[i] & adj[j]) >= 2:
                        total += 1
        return total
    
    def incremental_violation_change(n, colors, edge_idx, new_color):
        """
        Compute the change in total violations if we change edge edge_idx to new_color.
        Returns (delta, old_color) where delta is new_violations - old_violations.
        Uses incremental approach: only recompute affected pairs.
        """
        # Decode edge_idx to (i, j)
        # edge_idx = i*(2n-1-i)//2 + (j-i-1)
        # Solve for i: i*(2n-1-i)//2 <= edge_idx
        # Try binary search or direct formula
        # i*(2n-1-i) <= 2*edge_idx
        # 2ni - i^2 - i <= 2*edge_idx
        # i^2 - (2n-1)i + 2*edge_idx >= 0
        # Use the quadratic formula or just iterate
        # For small n this is fast enough
        i = 0
        while i < n:
            start = i * (2 * n - 1 - i) // 2
            end = start + (n - 1 - i)
            if start <= edge_idx < end:
                break
            i += 1
        else:
            i = n - 2
        j = i + 1 + (edge_idx - i * (2 * n - 1 - i) // 2)
        
        old_color = colors[edge_idx]
        if old_color == new_color:
            return 0, old_color
        
        # We need to compute how many C4s are created/destroyed
        # A C4 in color c involves 4 vertices a,b,c,d where all 4 edges are color c
        # and they form a cycle (not just a complete graph)
        # Actually, the condition is: no two vertices have two common neighbors in the same color
        # So a violation is a pair (u,v) such that |N_c(u) ∩ N_c(v)| >= 2
        
        # When we change edge (i,j) from old_color to new_color:
        # 1. In old_color: edge (i,j) is removed
        #    - This may reduce common neighbors for pairs involving i or j
        # 2. In new_color: edge (i,j) is added
        #    - This may increase common neighbors for pairs involving i or j
        
        # Build adjacency sets for old_color and new_color
        adj_old = [set() for _ in range(n)]
        adj_new = [set() for _ in range(n)]
        idx = 0
        for a in range(n):
            for b in range(a + 1, n):
                c = colors[idx]
                if c == old_color:
                    adj_old[a].add(b)
                    adj_old[b].add(a)
                elif c == new_color:
                    adj_new[a].add(b)
                    adj_new[b].add(a)
                idx += 1
        
        # Remove (i,j) from adj_old, add to adj_new
        adj_old[i].discard(j)
        adj_old[j].discard(i)
        adj_new[i].add(j)
        adj_new[j].add(i)
        
        # Count violations before and after for affected pairs
        # Affected pairs are those involving i or j (since only their neighborhoods change)
        # Plus pairs (x,y) where x or y is a neighbor of i or j in old or new color
        
        # Actually, the violation count for pair (u,v) depends on |N(u) ∩ N(v)|
        # When we add/remove edge (i,j), the neighborhoods of i and j change
        # So pairs (i,v) and (j,v) for all v are affected
        # Also, pairs (u,v) where u or v is in N(i) or N(j) might be affected?
        # No - the neighborhood of u only changes if u is i or j
        # So only pairs involving i or j have their common-neighbor counts changed
        
        delta = 0
        
        # For old_color: removing edge (i,j)
        # Pairs (i, v) for v != j: |N_old(i) ∩ N_old(v)| changes
        # Before: included j in N_old(i), so if j was also in N_old(v), it contributed
        # After: j is not in N_old(i), so that contribution is gone
        # Similarly for pairs (j, v)
        
        # Let's just count violations for pairs involving i or j in old and new colors
        # Before change:
        old_viol = 0
        for v in range(n):
            if v == i or v == j:
                continue
            # Pair (i, v) in old_color
            # N_old(i) includes j, N_old(v) may or may not include j
            common_before = len(adj_old[i] | {j}) & (adj_old[v])  # adj_old[i] doesn't have j yet in our modified version
            # Wait, we already removed j from adj_old[i] and adj_old[j]
            # So adj_old[i] is the "after" state
            # Let me re-think
            
        # This is getting complicated. Let me just count all violations before and after.
        # For n up to 29, this is O(n^2 * 5) which is fast enough.
        
        # Count violations in original coloring
        def count_viol_fast(n, colors):
            total = 0
            for c in range(5):
                adj = [set() for _ in range(n)]
                idx = 0
                for a in range(n):
                    for b in range(a + 1, n):
                        if colors[idx] == c:
                            adj[a].add(b)
                            adj[b].add(a)
                        idx += 1
                for a in range(n):
                    for b in range(a + 1, n):
                        if len(adj[a] & adj[b]) >= 2:
                            total += 1
            return total
        
        # This is still O(n^2 * 5) per call. For n=29, that's ~2000 * 5 = 10000 operations.
        # With 120s budget, we can do many iterations.
        
        # Let me use a different approach: precompute adjacency bitmasks for speed
        
        return 0, old_color  # Not using incremental, will use full recount
    
    def local_search(n, colors, max_iter=5000, seed=None):
        """Local search to minimize violations."""
        rng = random.Random(seed)
        colors = list(colors)
        m = n * (n - 1) // 2
        best = colors[:]
        best_viol = count_violations(n, best)
        
        if best_viol == 0:
            return best, 0
        
        # Precompute edge endpoints for fast lookup
        edge_endpoints = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_endpoints.append((i, j))
                idx += 1
        
        for it in range(max_iter):
            if best_viol == 0:
                break
            if time.time() > deadline - 5:
                break
            
            idx = rng.randint(0, m - 1)
            old = colors[idx]
            new = (old + rng.randint(1, 4)) % 5
            colors[idx] = new
            v = count_violations(n, colors)
            if v <= best_viol:
                best = colors[:]
                best_viol = v
            else:
                colors[idx] = old
        
        return best, best_viol
    
    # Start with a known good construction for smaller n
    # Use the Lazebnik-Woldar construction idea: based on finite fields
    # For n = q^2 + q + 1 with q prime power, we can construct a coloring
    # For q=5: n = 31 (too large, max is 29)
    # For q=4: n = 21
    # For q=3: n = 13
    # For q=2: n = 7
    
    # Let's start with a random coloring and use local search
    # But first, try to get a good starting point
    
    # Strategy: 
    # 1. For small n, we can use known constructions
    # 2. For larger n, use local search from random or structured starts
    
    # Known: n=7 is possible (Fano plane related)
    # n=13 is possible (projective plane of order 3)
    # n=21 is possible (projective plane of order 4)
    
    # Let's implement the projective plane construction for q being a prime power
    def projective_plane_coloring(q):
        """
        Construct a 5-coloring of K_{q^2+q+1} using the projective plane of order q.
        Vertices are the q^2+q+1 points of the projective plane.
        Edges are colored based on the line containing them.
        Each line has q+1 points, so q+1 edges per line.
        We need q+1 <= 5, so q <= 4.
        For q=4: 5 colors, n=21
        For q=3: 4 colors (use 3 of the 5), n=13
        For q=2: 3 colors, n=7
        """
        if q > 4:
            return None
        
        # Build projective plane of order q
        # Points: (x, y) for x,y in GF(q), plus (a, 1) for a in GF(q), plus (1, 0)
        # Total: q^2 + q + 1 points
        
        # For simplicity, use a different representation
        # Points are indexed 0 to q^2+q
        # Lines: each line has q+1 points
        
        # This is complex to implement correctly. Let me use a simpler approach.
        # For now, just return None and rely on local search.
        return None
    
    # Start with best known
    best_n = 7
    best_colors = None
    
    # Try n from 8 upward
    for target_n in range(8, 30):
        if time.time() > deadline - 20:
            break
        
        found = False
        remaining = deadline - time.time()
        
        # Number of attempts based on remaining time
        # Each attempt takes roughly: count_violations * max_iter
        # count_violations for n=29 is ~29*28/2 * 5 = 2030 set intersections
        # Each set intersection is O(degree)
        # Let's estimate ~0.01s per count_violations for n=29
        # So ~1000 iterations per second
        # With 20s remaining, ~20000 iterations total
        # Split among attempts
        
        num_attempts = max(3, int(remaining / 10))
        max_iter_per_attempt = max(500, int(remaining * 50 / num_attempts))
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(attempt * 7919 + target_n * 104729)
            colors = [rng.randint(0, 4) for _ in range(target_n * (target_n - 1) // 2)]
            
            colors, viol = local_search(target_n, colors, max_iter=max_iter_per_attempt, seed=attempt * 31 + target_n)
            
            if viol == 0:
                best_n = target_n
                best_colors = colors
                found = True
                break
        
        if not found and target_n >= 25:
            # If we can't find a solution for large n, stop
            # But keep trying with more iterations if time permits
            if time.time() > deadline - 10:
                break
    
    # Ensure we have a valid result
    if best_colors is None or not validate(best_n, best_colors):
        # Fallback to a small known-good coloring
        best_n = 7
        # Fano plane coloring
        # Points: 0-6
        # Lines: {0,1,2}, {0,3,4}, {0,5,6}, {1,3,5}, {1,4,6}, {2,3,6}, {2,4,5}
        # Color each line with a distinct color
        best_colors = [0] * (7 * 6 // 2)
        lines = [
            (0, 1, 2),
            (0, 3, 4),
            (0, 5, 6),
            (1, 3, 5),
            (1, 4, 6),
            (2, 3, 6),
            (2, 4, 5)
        ]
        idx = 0
        for i in range(7):
            for j in range(i + 1, 7):
                for c, line in enumerate(lines):
                    if i in line and j in line:
                        best_colors[idx] = c
                        break
                idx += 1
    
    return {'n': best_n, 'colors': best_colors}
