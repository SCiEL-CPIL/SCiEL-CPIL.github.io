import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Known result: R(3,3,3,3) > 50 (Chung's 50-vertex coloring)
    # We'll use a greedy vertex-by-vertex construction followed by local search.
    # The key insight is to build the coloring incrementally, adding one vertex at a time,
    # and use local search to resolve conflicts.

    # Helper: edge index for (i, j) with i < j in a K_n graph
    def edge_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Helper: count monochromatic triangles in a coloring
    def count_mono_triangles(colors, n):
        """Count total number of monochromatic triangles."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_idx(i, j, n)]
                for k in range(j + 1, n):
                    if colors[edge_idx(i, k, n)] == c and colors[edge_idx(j, k, n)] == c:
                        count += 1
        return count

    # Helper: find a monochromatic triangle, return (i, j, k) or None
    def find_mono_triangle(colors, n):
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_idx(i, j, n)]
                for k in range(j + 1, n):
                    if colors[edge_idx(i, k, n)] == c and colors[edge_idx(j, k, n)] == c:
                        return (i, j, k)
        return None

    # Greedy initialization: build K_n by adding vertices one at a time
    def greedy_init(n):
        """Build a 4-coloring of K_n with few monochromatic triangles using greedy approach."""
        colors = [0] * (n * (n - 1) // 2)
        
        # Start with K_1 (no edges)
        # For each new vertex v, assign colors to edges (v, u) for u < v
        for v in range(1, n):
            for u in range(v):
                idx = edge_idx(u, v, n)
                # Choose color that minimizes new monochromatic triangles
                best_color = 0
                best_count = float('inf')
                for c in range(4):
                    # If we assign color c to edge (u, v), how many new mono triangles?
                    # A new triangle (u, v, w) with w < u or w < v is formed if
                    # colors[(u,w)] == c and colors[(v,w)] == c
                    # But w < u < v, so we need to check all w < u
                    # Also triangles where u is the middle: (w, u, v) with w < u
                    # And triangles where v is the middle: (u, w, v) - but w > u > v? No.
                    # Actually, for edge (u, v) with u < v, new triangles are (w, u, v) for w < u
                    # where colors[(w,u)] == c and colors[(w,v)] == c
                    # But colors[(w,v)] hasn't been assigned yet!
                    # 
                    # Let me reconsider. When adding vertex v, we assign colors to (u, v) for all u < v.
                    # A monochromatic triangle involving v and two existing vertices u, w (u < w < v)
                    # is formed if colors[(u,v)] == colors[(w,v)] == colors[(u,w)].
                    # So when choosing color for (u, v), we need to check all w < u:
                    # if colors[(w, v)] == c and colors[(u, w)] == c, then triangle (w, u, v) is mono.
                    # But colors[(w, v)] may not be assigned yet if w is processed after u.
                    #
                    # To handle this, let's process u in order and use a running count.
                    # Actually, let's just do a simpler greedy: for each (u, v), pick the color
                    # that creates the fewest mono triangles with already-assigned edges.
                    
                    new_tris = 0
                    for w in range(u):
                        if colors[edge_idx(w, v, n)] == c and colors[edge_idx(u, w, n)] == c:
                            new_tris += 1
                    if new_tris < best_count:
                        best_count = new_tris
                        best_color = c
                colors[idx] = best_color
        
        return colors

    # Local search: minimize monochromatic triangles by flipping edge colors
    def local_search(colors, n, max_iters=50000, time_limit=10.0):
        """Minimize monochromatic triangles via local search."""
        current = colors[:]
        current_count = count_mono_triangles(current, n)
        
        if current_count == 0:
            return current, 0
        
        start_ls = time.time()
        for it in range(max_iters):
            if time.time() - start_ls > time_limit:
                break
            if current_count == 0:
                break
            
            # Pick a random edge
            e = random.randint(0, len(current) - 1)
            # Decode e to (i, j)
            # e = i*(n-1) - i*(i-1)//2 + (j-i-1)
            # Find i: i*(n-1) - i*(i-1)//2 <= e
            # Approx: i ~ n-1 - sqrt((n-1)^2 - 2*e)
            i = int((n - 1) - ((n - 1)**2 - 2 * e) ** 0.5)
            # Adjust
            while i < n - 1 and edge_idx(i, i + 1, n) <= e:
                i += 1
            i -= 1
            # Now find j
            rem = e - (i * (n - 1) - i * (i - 1) // 2)
            j = i + 1 + rem
            
            old_c = current[e]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Compute delta in mono triangle count
            # For each k not in {i, j}, check if changing (i,j) affects triangle (i,j,k)
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                ik_idx = edge_idx(i, k, n)
                jk_idx = edge_idx(j, k, n)
                c_ik = current[ik_idx]
                c_jk = current[jk_idx]
                
                # Old state: edge (i,j) has color old_c
                # Triangle (i,j,k) is mono if old_c == c_ik == c_jk
                was_mono = (old_c == c_ik and old_c == c_jk)
                # New state: edge (i,j) has color new_c
                is_mono = (new_c == c_ik and new_c == c_jk)
                
                if was_mono and not is_mono:
                    delta -= 1
                elif not was_mono and is_mono:
                    delta += 1
            
            # Accept if delta < 0 or with some probability (simulated annealing-like)
            if delta <= 0:
                current[e] = new_c
                current_count += delta
            else:
                # Simple hill climbing: only accept if it helps
                # Or use a small random acceptance
                if random.random() < 0.05:  # 5% chance to accept worse moves
                    current[e] = new_c
                    current_count += delta
        
        return current, current_count

    # Main construction: build up to n=50
    def try_build(target_n):
        """Try to build a valid 4-coloring of K_target_n."""
        # Start with greedy initialization
        colors = greedy_init(target_n)
        
        # Run local search to eliminate mono triangles
        colors, count = local_search(colors, target_n, max_iters=100000, time_limit=12.0)
        
        if count == 0:
            return colors
        return None

    # First, try to get a good K_32 coloring as a base
    # Then try to extend to larger n
    
    best_n = 0
    best_colors = None
    
    # Try direct construction for various n
    # Start with smaller n to establish a base, then try larger
    
    # Try n=32 first (known to be possible)
    if time.time() < deadline - 30:
        colors32 = greedy_init(32)
        colors32, count32 = local_search(colors32, 32, max_iters=50000, time_limit=8.0)
        if count32 == 0:
            best_n = 32
            best_colors = colors32[:]
    
    # Now try to extend to larger n
    # Try n=36, 40, 44, 48, 50
    target_ns = [36, 40, 44, 48, 50]
    
    for target in target_ns:
        if time.time() > deadline - 15:
            break
        if target <= best_n:
            continue
        
        # Try multiple times
        success = False
        attempts = 3 if target <= 44 else 2
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Greedy init + local search
            colors = greedy_init(target)
            colors, count = local_search(colors, target, max_iters=80000, time_limit=15.0)
            
            if count == 0:
                best_n = target
                best_colors = colors[:]
                success = True
                break
        
        if success:
            continue
        else:
            # If we can't reach this target, stop trying larger ones
            # But maybe we can reach something in between?
            # For now, just break
            break
    
    # If we haven't achieved n=50, try a more aggressive search for n=50
    if best_n < 50 and time.time() < deadline - 10:
        # Try more aggressive local search for n=50
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            colors = greedy_init(50)
            # More aggressive local search
            colors, count = local_search(colors, 50, max_iters=200000, time_limit=12.0)
            if count == 0:
                best_n = 50
                best_colors = colors[:]
                break
    
    # Final fallback: if we have nothing, return a small valid coloring
    if best_n == 0 or best_colors is None:
        # Return a trivial K_1
        best_n = 1
        best_colors = []
    
    return {'n': best_n, 'colors': best_colors}
