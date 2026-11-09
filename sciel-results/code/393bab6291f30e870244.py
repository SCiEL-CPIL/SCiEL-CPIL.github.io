import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def is_in_c6(n, colors, u, v, c):
        """
        Checks if edge (u, v) in color c is part of a C6.
        """
        # Build adj for color c only? No, we need adj for all colors to build masks?
        # We only need adj for color c.
        # But we call this for specific c.
        # Let's precompute adj for all colors outside the loop?
        # No, colors change.
        
        # Build adj for color c
        adj_c = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * n - i * (i + 1) // 2 + (j - i - 1)
                if colors[idx] == c:
                    adj_c[i] |= (1 << j)
                    adj_c[j] |= (1 << i)
                    
        # Check if (u, v) is in a C6
        # Path of length 5 from u to v in G_c
        def has_path_length_5(start, end):
            stack = [(start, 0, 1 << start)]
            while stack:
                node, d, visited = stack.pop()
                if d == 5:
                    if node == end:
                        return True
                    continue
                neighbors = adj_c[node]
                if d == 4:
                    if neighbors & (1 << end) and not (visited & (1 << end)):
                        return True
                    continue
                next_mask = neighbors & ~visited
                while next_mask:
                    next_node = (next_mask & -next_mask).bit_length() - 1
                    next_mask &= next_mask - 1
                    new_visited = visited | (1 << next_node)
                    stack.append((next_node, d + 1, new_visited))
            return False

        return has_path_length_5(u, v)

    def get_idx(i, j, n):
        if i > j: i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def try_find_valid_coloring(n, max_iters=5000):
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Pick a random edge
            idx = random.randint(0, num_edges - 1)
            # Recover u, v from idx
            # This is slow. Let's pick u, v randomly.
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            c = colors[get_idx(u, v, n)]
            
            if is_in_c6(n, colors, u, v, c):
                # Recolor
                new_c = (c + random.randint(1, 3)) % 4
                colors[get_idx(u, v, n)] = new_c
            else:
                # If we pick many edges that are not in C6, we might miss that there are no C6s.
                # We need a global check occasionally.
                # Let's do a global check every 100 iterations.
                if _ % 100 == 0:
                    # Check all edges
                    found_c6 = False
                    for i in range(n):
                        for j in range(i + 1, n):
                            c_ij = colors[get_idx(i, j, n)]
                            if is_in_c6(n, colors, i, j, c_ij):
                                found_c6 = True
                                break
                        if found_c6: break
                    if not found_c6:
                        return colors
        
        return None

    for n in range(12, 9, -1):
        if time.time() > deadline:
            break
        for restart in range(20):
            if time.time() > deadline:
                break
            colors = try_find_valid_coloring(n, max_iters=1000)
            if colors is not None:
                return {'n': n, 'colors': colors}
    
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
