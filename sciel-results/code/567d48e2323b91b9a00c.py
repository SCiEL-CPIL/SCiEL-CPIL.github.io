import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def count_and_find_triangle(N, n):
        """
        Counts monochromatic triangles and returns one if found.
        N is a list of 4 lists of bitmasks. N[c][i] is bitmask of neighbors of i with color c.
        Returns (count, (c, i, j, k) or None)
        """
        total = 0
        found = None
        for c in range(4):
            Nc = N[c]
            for i in range(n):
                Ni = Nc[i]
                for j in range(i + 1, n):
                    inter = Ni & Nc[j]
                    if inter:
                        total += inter.bit_count()
                        if found is None:
                            lsb = inter & -inter
                            k = lsb.bit_length() - 1
                            found = (c, i, j, k)
        return total // 3, found

    def find_one_triangle(N, n):
        """Quickly find one monochromatic triangle, or None if none exists."""
        for c in range(4):
            Nc = N[c]
            for i in range(n):
                Ni = Nc[i]
                for j in range(i + 1, n):
                    inter = Ni & Nc[j]
                    if inter:
                        lsb = inter & -inter
                        k = lsb.bit_length() - 1
                        return (c, i, j, k)
        return None

    def solve(n, time_limit, max_restarts=100):
        """
        Local search with restarts.
        Returns N if triangle-free coloring found, else None.
        """
        best_N = None
        best_count = float('inf')
        
        for restart in range(max_restarts):
            if time.time() >= time_limit:
                break
            
            # Initialize random coloring
            N = [[0] * n for _ in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = random.randint(0, 3)
                    N[c][i] |= (1 << j)
                    N[c][j] |= (1 << i)
            
            cur_count, _ = count_and_find_triangle(N, n)
            if cur_count == 0:
                return N
            
            if cur_count < best_count:
                best_count = cur_count
                best_N = [row[:] for row in N]
            
            # Local search phase with limited steps
            no_improve_steps = 0
            max_no_improve = 5000
            local_time_limit = min(time_limit, time.time() + 2.0)  # 2s per restart
            
            while time.time() < local_time_limit:
                tri = find_one_triangle(N, n)
                if tri is None:
                    return N
                
                c, i, j, k = tri
                # Try all 3 edges x 3 new colors = 9 possible moves
                # Pick the one that reduces triangle count most (or random if tied)
                best_move = None
                best_delta = 0
                
                edges = [(i, j), (j, k), (i, k)]
                for u, v in edges:
                    for new_c in range(4):
                        if new_c == c:
                            continue
                        # Simulate: what's the change in triangle count?
                        # This is complex; for simplicity, just do random move
                        # But let's at least try to be smart:
                        # Changing edge (u,v) from c to new_c:
                        # Removes triangles involving (u,v) in color c
                        # Adds triangles involving (u,v) in color new_c
                        # For now, use random selection
                        pass
                
                # Simple: random edge, random new color
                edge = random.choice(edges)
                u, v = edge
                new_c = random.choice([x for x in range(4) if x != c])
                
                # Apply move
                N[c][u] &= ~(1 << v)
                N[c][v] &= ~(1 << u)
                N[new_c][u] |= (1 << v)
                N[new_c][v] |= (1 << u)
                
                # Check improvement (sampled)
                if no_improve_steps % 100 == 0:
                    cur_count, _ = count_and_find_triangle(N, n)
                    if cur_count == 0:
                        return N
                    if cur_count < best_count:
                        best_count = cur_count
                        best_N = [row[:] for row in N]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 100
                else:
                    no_improve_steps += 1
                
                if no_improve_steps >= max_no_improve:
                    break
        
        # Return best found if it's triangle-free, else None
        if best_N is not None:
            cnt, _ = count_and_find_triangle(best_N, n)
            if cnt == 0:
                return best_N
        return None

    def N_to_colors(N, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                for c in range(4):
                    if (N[c][i] >> j) & 1:
                        colors.append(c)
                        break
        return colors

    # Try 50 with multiple restarts
    time_for_50 = min(60.0, deadline - 10.0)
    result_50 = solve(50, start_time + time_for_50)
    if result_50 is not None:
        cnt, _ = count_and_find_triangle(result_50, 50)
        if cnt == 0:
            return {'n': 50, 'colors': N_to_colors(result_50, 50)}

    # Try 40 with remaining time
    if time.time() < deadline - 5.0:
        result_40 = solve(40, deadline - 5.0)
        if result_40 is not None:
            cnt, _ = count_and_find_triangle(result_40, 40)
            if cnt == 0:
                return {'n': 40, 'colors': N_to_colors(result_40, 40)}

    # Fallback to seed 32
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
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
