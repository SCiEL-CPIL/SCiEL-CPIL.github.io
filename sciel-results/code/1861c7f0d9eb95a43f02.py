import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Fallback: n=11 cyclic coloring
    def get_fallback():
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}

    # Try to find a valid coloring for n
    def find_coloring(n, time_limit):
        # Initialize random coloring
        colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
        
        # Helper to get/set edge color
        # Index: i < j. idx = i*n - i*(i+1)//2 + j - i - 1 ? 
        # Simpler: use a 2D list or a dict. 
        # For n=26, n*n=676, a list of lists is fine.
        edge_color = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * (n - 1) - i * (i + 1) // 2 + j - i - 1
                c = colors[idx]
                edge_color[i][j] = c
                edge_color[j][i] = c

        # Data structure for common neighbors
        # common[c][u][v] = number of w such that edge(u,w) and edge(v,w) are color c
        common = [[[0] * n for _ in range(n)] for _ in range(5)]
        
        # Initialize common counts
        for u in range(n):
            for v in range(u + 1, n):
                for c in range(5):
                    count = 0
                    for w in range(n):
                        if w != u and w != v:
                            if edge_color[u][w] == c and edge_color[v][w] == c:
                                count += 1
                    common[c][u][v] = count
                    common[c][v][u] = count

        # Check if valid
        def is_valid():
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        if common[c][u][v] >= 2:
                            return False
            return True

        # Get a random conflict (u, v, c) where common[c][u][v] >= 2
        def get_conflict():
            conflicts = []
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        if common[c][u][v] >= 2:
                            conflicts.append((u, v, c))
            if not conflicts:
                return None
            return random.choice(conflicts)

        # Flip edge (u, v) from color c_old to c_new
        def flip_edge(u, v, c_new):
            c_old = edge_color[u][v]
            if c_old == c_new:
                return
            
            # Update common counts for c_old
            # Neighbors of u in c_old: Nu
            # Neighbors of v in c_old: Nv
            # For each w in Nu, common[c_old][v][w] decreases by 1?
            # No, w is a common neighbor of u and v in c_old.
            # So for each w in Nu, the pair (v, w) had u as a common neighbor in c_old.
            # So common[c_old][v][w] -= 1.
            # And for each w in Nv, the pair (u, w) had v as a common neighbor in c_old.
            # So common[c_old][u][w] -= 1.
            
            # But we also need to consider pairs (x, y) where x is neighbor of u in c_old and y is neighbor of v in c_old?
            # No, the definition is: common[c][a][b] is count of z such that edge(a,z)=c and edge(b,z)=c.
            # If we change edge(u,v) from c_old to c_new:
            # 1. For c_old:
            #    u and v are no longer connected by c_old.
            #    This affects common counts for pairs involving u or v?
            #    Actually, it affects the "common neighbors" of other pairs.
            #    If w is a neighbor of u in c_old, then w and v share u as a common neighbor in c_old.
            #    So common[c_old][w][v] decreases by 1.
            #    If w is a neighbor of v in c_old, then w and u share v as a common neighbor in c c_old.
            #    So common[c_old][w][u] decreases by 1.
            #    Also, u and v themselves? common[c_old][u][v] is the count of common neighbors.
            #    Changing edge(u,v) does NOT change common[c_old][u][v] directly, 
            #    because u and v are not neighbors of themselves.
            #    Wait, the edge (u,v) is not used in the count of common neighbors of u and v.
            #    The count is over w != u, v.
            #    So common[c_old][u][v] is unchanged by flipping edge(u,v)?
            #    YES. The edge (u,v) is not part of the path u-w-v.
            #    So only pairs (w, v) and (w, u) are affected.
            
            # 2. For c_new:
            #    Similarly, common[c_new][w][v] increases by 1 for w in Nu_new?
            #    No, w must be a neighbor of u in c_new.
            #    But we are CHANGING u-v to c_new.
            #    So u and v are now connected by c_new.
            #    This means for any w, if edge(u,w)=c_new and edge(v,w)=c_new, 
            #    then u and v share w.
            #    So common[c_new][u][v] increases by 1?
            #    No, common[c_new][u][v] counts w such that edge(u,w)=c_new and edge(v,w)=c_new.
            #    The edge (u,v) itself is not w.
            #    So common[c_new][u][v] is unchanged.
            #    But pairs (w, u) and (w, v) are affected.
            #    If w is neighbor of v in c_new, then w and u share v as common neighbor in c_new.
            #    So common[c_new][w][u] increases by 1.
            
            # Summary of updates for flipping (u,v) from c_old to c_new:
            
            # Remove c_old:
            # For each w in range(n):
            #   if w != u and w != v:
            #     if edge_color[u][w] == c_old:
            #       common[c_old][w][v] -= 1
            #       common[c_old][v][w] -= 1
            #     if edge_color[v][w] == c_old:
            #       common[c_old][w][u] -= 1
            #       common[c_old][u][w] -= 1
            
            # Add c_new:
            # For each w in range(n):
            #   if w != u and w != v:
            #     if edge_color[u][w] == c_new:
            #       common[c_new][w][v] += 1
            #       common[c_new][v][w] += 1
            #     if edge_color[v][w] == c_new:
            #       common[c_new][w][u] += 1
            #       common[c_new][u][w] += 1
            
            # Update edge_color
            edge_color[u][v] = c_new
            edge_color[v][u] = c_new
            
            # Perform updates
            for w in range(n):
                if w == u or w == v:
                    continue
                if edge_color[u][w] == c_old:
                    common[c_old][w][v] -= 1
                    common[c_old][v][w] -= 1
                if edge_color[v][w] == c_old:
                    common[c_old][w][u] -= 1
                    common[c_old][u][w] -= 1
                
                if edge_color[u][w] == c_new:
                    common[c_new][w][v] += 1
                    common[c_new][v][w] += 1
                if edge_color[v][w] == c_new:
                    common[c_new][w][u] += 1
                    common[c_new][u][w] += 1

        # Local Search Loop
        max_iter = 100000
        for _ in range(max_iter):
            if time.time() > time_limit:
                break
            if is_valid():
                # Extract colors
                res_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        res_colors.append(edge_color[i][j])
                return res_colors
            
            conflict = get_conflict()
            if conflict is None:
                # Should be valid, but is_valid might be slow?
                # If no conflict, return
                res_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        res_colors.append(edge_color[i][j])
                return res_colors
            
            u, v, c = conflict
            # Change color of (u, v) to a random other color
            new_c = random.choice([x for x in range(5) if x != c])
            flip_edge(u, v, new_c)
            
        return None

    # Try n from 26 down to 11
    for n in range(26, 10, -1):
        if time.time() > deadline:
            break
        # Use a fraction of the time for each n
        # Give more time to larger n? Or just try quickly?
        # Let's try each n with a short timeout, e.g., 5s
        t_start = time.time()
        t_end = min(deadline, t_start + 5.0)
        
        # If n is small, it's easier.
        result = find_coloring(n, t_end)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback
    return get_fallback()
