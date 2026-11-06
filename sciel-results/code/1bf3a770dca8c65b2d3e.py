import time
import random

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def build_greenwood_gleason(n=32):
        """
        Builds the 32-vertex Greenwood-Gleason coloring.
        Vertices are 5-bit vectors (0..31).
        Color is index of lowest set bit of u^v, mapped to 0,1,2,3.
        Bits 0,1,2 -> Colors 0,1,2.
        Bits 3,4 -> Color 3.
        """
        color = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = i ^ j
                if d & 1:
                    c = 0
                elif d & 2:
                    c = 1
                elif d & 4:
                    c = 2
                else:
                    c = 3
                color[i][j] = c
                color[j][i] = c
        return color

    def count_mono_triangles(color, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = color[i][j]
                for k in range(j + 1, n):
                    if color[i][k] == c and color[j][k] == c:
                        count += 1
        return count

    def get_delta(color, n, u, v, new_c):
        old_c = color[u][v]
        if old_c == new_c:
            return 0
        delta = 0
        for k in range(n):
            if k == u or k == v:
                continue
            c_uk = color[u][k]
            c_vk = color[v][k]
            was_mono = (c_uk == old_c and c_vk == old_c)
            is_mono = (c_uk == new_c and c_vk == new_c)
            if is_mono and not was_mono:
                delta += 1
            elif was_mono and not is_mono:
                delta -= 1
        return delta

    def local_search(n, time_limit):
        """
        Attempts to find a valid 4-coloring of K_n.
        Returns (color, M) where M is the number of mono triangles.
        """
        start_time = time.time()
        best_color = None
        best_M = float('inf')
        
        # Try multiple random restarts
        while time.time() < start_time + time_limit:
            # Initialize random coloring
            color = [[random.randint(0, 3) for _ in range(n)] for _ in range(n)]
            # Make symmetric
            for i in range(n):
                for j in range(i + 1, n):
                    color[j][i] = color[i][j]
            
            M = count_mono_triangles(color, n)
            
            if M < best_M:
                best_M = M
                best_color = [row[:] for row in color]
                if M == 0:
                    return best_color, 0
            
            # Local search
            stuck_count = 0
            while time.time() < start_time + time_limit - 5:
                if M == 0:
                    return color, 0
                
                # Find best move
                best_delta = 0
                best_move = None
                # Full scan of all moves
                for u in range(n):
                    for v in range(u + 1, n):
                        old_c = color[u][v]
                        for new_c in range(4):
                            if new_c == old_c:
                                continue
                            d = get_delta(color, n, u, v, new_c)
                            if d < best_delta:
                                best_delta = d
                                best_move = (u, v, new_c)
                
                if best_move is not None:
                    u, v, new_c = best_move
                    color[u][v] = new_c
                    color[v][u] = new_c
                    M += best_delta
                    stuck_count = 0
                else:
                    # Stuck in local minimum
                    stuck_count += 1
                    if stuck_count > 20:
                        break
                    # Random kick to escape
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    new_c = random.randint(0, 3)
                    if new_c != color[u][v]:
                        d = get_delta(color, n, u, v, new_c)
                        color[u][v] = new_c
                        color[v][u] = new_c
                        M += d

        return best_color, best_M

    # Known valid solution for n=32
    best_n = 32
    best_valid_color = build_greenwood_gleason(32)
    
    # Try to find larger n
    # We prioritize higher n.
    for n in range(50, 31, -1):
        if time.time() > deadline - 10:
            break
        
        # Allocate time for each n
        # Higher n gets more time
        time_budget = 20 if n >= 48 else 15
        
        color, M = local_search(n, time_budget)
        
        if M == 0:
            best_n = n
            best_valid_color = color
            break

    # Extract colors in lexicographic order
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(best_valid_color[i][j])
    
    return {'n': best_n, 'colors': colors}
