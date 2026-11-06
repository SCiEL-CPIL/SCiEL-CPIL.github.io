import time
import random
import math

def construct():
    deadline = time.time() + 135
    
    # Greenwood-Gleason n=32 construction
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    def colors_to_adj(colors, n):
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                adj[i][j] = adj[j][i] = colors[idx]
                idx += 1
        return adj
    
    def adj_to_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(adj[i][j])
        return colors
        
    def count_mono_triangles(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                for k in range(j+1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        count += 1
        return count

    def backtrack_add_vertex(adj, n, time_limit=30.0):
        """
        Backtracking with forward checking to add vertex n to existing n-vertex coloring.
        assign[v] = color for edge (n, v), v in 0..n-1
        Constraint: for any pair (u,v) with u<v, if assign[u]==assign[v]==c, then adj[u][v] != c
        """
        start_time = time.time()
        assign = [-1] * n
        
        # For each color c, maintain which vertices are "forbidden" to have color c
        # A vertex v cannot have color c if there exists u < v with assign[u] == c and adj[u][v] == c
        # But we also need: if assign[u] == assign[v] == c, then adj[u][v] != c
        
        def is_valid_assign(v, c, assign):
            """Check if assigning color c to vertex v is valid given current partial assignment"""
            for u in range(v):
                if assign[u] == c and adj[u][v] == c:
                    return False
            return True
        
        def get_available_colors(v, assign):
            """Get list of available colors for vertex v"""
            avail = []
            for c in range(4):
                valid = True
                for u in range(v):
                    if assign[u] == c and adj[u][v] == c:
                        valid = False
                        break
                if valid:
                    avail.append(c)
            return avail
        
        # MRV: pick vertex with fewest available colors
        # But we must assign in order 0, 1, ..., n-1 for the backtracking to work properly
        # Actually, we can reorder: process vertices in an order that minimizes branching
        
        # Let's use a different approach: assign in order 0 to n-1
        # Use forward checking: after assigning vertex v, check if any future vertex has no available colors
        
        def forward_check(v, c, assign):
            """After assigning color c to vertex v, check if all future vertices have at least one available color"""
            for w in range(v + 1, n):
                has_color = False
                for cw in range(4):
                    valid = True
                    for u in range(w):
                        if u == v:
                            # assign[v] is now c
                            if c == cw and adj[v][w] == cw:
                                valid = False
                                break
                        elif assign[u] == cw and adj[u][w] == cw:
                            valid = False
                            break
                    if valid:
                        has_color = True
                        break
                if not has_color:
                    return False
            return True
        
        # Backtracking
        def bt(v):
            if time.time() - start_time > time_limit:
                return False
            if v == n:
                return True
            
            # Try colors in order of least constraining value
            colors_to_try = [0, 1, 2, 3]
            # Sort by how many future vertices they eliminate
            scored_colors = []
            for c in colors_to_try:
                if not is_valid_assign(v, c, assign):
                    continue
                # Count how many future vertices this color assignment would block
                block_count = 0
                for w in range(v + 1, n):
                    # If assign[v] = c, then w cannot have color c if adj[v][w] == c
                    # But w might still have other colors
                    # Count if this makes w have no available colors
                    w_has_other = False
                    for cw in range(4):
                        if cw == c:
                            continue
                        valid = True
                        for u in range(w):
                            if u == v:
                                if c == cw:  # This can't happen since we skip c
                                    pass
                            elif assign[u] == cw and adj[u][w] == cw:
                                valid = False
                                break
                        if valid:
                            w_has_other = True
                            break
                    if not w_has_other and (adj[v][w] == c):
                        block_count += 1
                
                scored_colors.append((block_count, c))
            
            scored_colors.sort()
            
            for _, c in scored_colors:
                if not is_valid_assign(v, c, assign):
                    continue
                assign[v] = c
                if forward_check(v, c, assign):
                    if bt(v + 1):
                        return True
                assign[v] = -1
            
            return False
        
        # Try multiple random orderings by shuffling initial color preferences
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            random.seed(42 + n + attempt)
            assign = [-1] * n
            if bt(0):
                return assign
        
        return None

    adj = colors_to_adj(colors_32, 32)
    best_n = 32
    best_colors = colors_32
    
    # Try to expand
    for target_n in range(33, 51):
        if time.time() > deadline - 15:
            break
            
        n = len(adj)
        
        found = False
        # Try backtracking with different time limits
        time_limit = min(20.0, max(5.0, (deadline - time.time()) - 15))
        assignment = backtrack_add_vertex(adj, n, time_limit=time_limit)
        
        if assignment is not None:
            # Success! Add vertex
            new_n = n + 1
            new_adj = [row[:] for row in adj]
            for _ in range(new_n):
                new_adj.append([-1] * new_n)
            for i in range(n):
                new_adj[i][n] = new_adj[n][i] = assignment[i]
            new_adj[n][n] = -1
            
            adj = new_adj
            best_n = new_n
            best_colors = adj_to_colors(adj, best_n)
            found = True
        else:
            # Failed to add vertex, try a different approach
            # Maybe we can modify the existing coloring slightly to make extension possible
            # For now, break
            break
            
    # Final verification
    adj_final = colors_to_adj(best_colors, best_n)
    if count_mono_triangles(adj_final, best_n) > 0:
        return {'n': 32, 'colors': colors_32}
    
    return {'n': best_n, 'colors': best_colors}
