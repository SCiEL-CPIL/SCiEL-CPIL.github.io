import time
import random

N = 16
DEADLINE = 55.0  # safety margin before 60s kill

def construct():
    """
    Constructs a Gallai coloring of K_16 using randomized recursive splitting,
    then attempts to increase the number of distinct colors via local search
    (edge recoloring) while maintaining the no-rainbow-C4 constraint.
    Multiple random restarts are used to explore different structures.
    """
    start_time = time.time()
    
    # Precompute all 4-vertex subsets and their 3 four-cycles
    all_quads = []
    edge_to_cycles = { (i, j): [] for i in range(N) for j in range(i+1, N) }
    
    quad_index = 0
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    cycles = [
                        [(a, b), (b, c), (c, d), (d, a)],
                        [(a, b), (b, d), (d, c), (c, a)],
                        [(a, c), (c, b), (b, d), (d, a)],
                    ]
                    all_quads.append(cycles)
                    for cyc in cycles:
                        for (u, v) in cyc:
                            e_key = (u, v) if u < v else (v, u)
                            edge_to_cycles[e_key].append(quad_index)
                    quad_index += 1

    def count_colors(colors):
        seen = set()
        for i in range(N):
            for j in range(i + 1, N):
                seen.add(colors[i][j])
        return len(seen)

    def is_valid_local(colors, edge, new_color):
        """Check if recoloring `edge` to `new_color` maintains validity.
        Modifies colors in place if valid. Reverts if invalid."""
        u, v = edge
        old_color = colors[u][v]
        if old_color == new_color:
            return False
        
        e_key = (u, v) if u < v else (v, u)
        cycle_indices = edge_to_cycles[e_key]
        
        colors[u][v] = new_color
        colors[v][u] = new_color
        
        for q_idx in cycle_indices:
            cycles = all_quads[q_idx]
            for cyc in cycles:
                cs = set()
                for (x, y) in cyc:
                    cs.add(colors[x][y])
                    if len(cs) == 4:
                        colors[u][v] = old_color
                        colors[v][u] = old_color
                        return False
        return True

    def generate_random_gallai(rng):
        """Generates a random Gallai coloring using recursive random splits."""
        colors = [[0] * N for _ in range(N)]
        next_color = [0]

        def assign_colors(vertices):
            n = len(vertices)
            if n <= 1:
                return next_color[0]
            if n == 2:
                u, v = vertices
                c = next_color[0]
                colors[u][v] = c
                colors[v][u] = c
                next_color[0] += 1
                return next_color[0]
            if n == 3:
                u, v, w = vertices
                c1 = next_color[0]
                c2 = next_color[0] + 1
                c3 = next_color[0] + 2
                colors[u][v] = c1; colors[v][u] = c1
                colors[v][w] = c2; colors[w][v] = c2
                colors[u][w] = c3; colors[w][u] = c3
                next_color[0] += 3
                return next_color[0]
            
            # Random split
            min_k = 2
            max_k = n - 2
            if min_k > max_k:
                min_k = 1
                max_k = n - 1
            
            k = rng.randint(min_k, max_k)
            
            shuffled_verts = vertices[:]
            rng.shuffle(shuffled_verts)
            
            v1 = shuffled_verts[:k]
            v2 = shuffled_verts[k:]
            
            cross_color = next_color[0]
            for u in v1:
                for v in v2:
                    colors[u][v] = cross_color
                    colors[v][u] = cross_color
            next_color[0] += 1
            
            assign_colors(v1)
            assign_colors(v2)
            return next_color[0]

        assign_colors(list(range(N)))
        return colors

    def local_search(colors, rng, time_limit):
        """Performs local search to maximize color count within time_limit."""
        best_colors = [row[:] for row in colors]
        best_count = count_colors(best_colors)
        current_count = best_count
        
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        no_improve_streak = 0
        max_no_improve = 2000
        
        while time.time() - start_time < time_limit:
            if no_improve_streak > max_no_improve:
                break
                
            improved = False
            current_colors_set = set()
            for i in range(N):
                for j in range(i+1, N):
                    current_colors_set.add(colors[i][j])
            
            unused_colors = [c for c in range(120) if c not in current_colors_set]
            
            edges_to_try = all_edges[:]
            rng.shuffle(edges_to_try)
            
            for edge in edges_to_try:
                if time.time() - start_time > time_limit:
                    return best_colors
                
                if unused_colors:
                    rng.shuffle(unused_colors)
                    for new_color in unused_colors:
                        if is_valid_local(colors, edge, new_color):
                            current_count += 1
                            if current_count > best_count:
                                best_count = current_count
                                best_colors = [row[:] for row in colors]
                            improved = True
                            no_improve_streak = 0
                            break
                    if improved:
                        break
                
            if improved:
                continue
            else:
                # Try neutral move
                e_idx = rng.randint(0, len(all_edges)-1)
                edge = all_edges[e_idx]
                u, v = edge
                old_color = colors[u][v]
                
                found_move = False
                for _ in range(5):
                    new_color = rng.randint(0, 119)
                    if new_color == old_color:
                        continue
                    if is_valid_local(colors, edge, new_color):
                        new_count = count_colors(colors)
                        if new_count >= current_count:
                            current_count = new_count
                            if new_count > best_count:
                                best_count = new_count
                                best_colors = [row[:] for row in colors]
                            no_improve_streak = 0
                            found_move = True
                            break
                        else:
                            colors[u][v] = old_color
                            colors[v][u] = old_color
                    # else is_valid_local reverted it
                
                if not found_move:
                    no_improve_streak += 1

        return best_colors

    best_global = None
    best_global_count = 0
    
    # Allocate time for multiple restarts
    # Each restart gets a portion of the time
    # Let's do dynamic allocation: keep trying as long as time permits
    # Estimate time per restart ~ 1-2s
    
    while time.time() - start_time < DEADLINE:
        rng = random.Random()
        colors = generate_random_gallai(rng)
        
        # Give this search up to 5 seconds, or remaining time if less
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 1.0:
            break
            
        search_time_limit = min(5.0, remaining)
        
        result = local_search(colors, rng, start_time + search_time_limit)
        
        result_count = count_colors(result)
        if result_count > best_global_count:
            best_global_count = result_count
            best_global = result

    if best_global is None:
        # Fallback: simple deterministic
        colors = generate_random_gallai(random.Random(42))
        return [colors[i][j] for i in range(N) for j in range(i+1, N)]
        
    return [best_global[i][j] for i in range(N) for j in range(i+1, N)]

if __name__ == "__main__":
    # Simple self-check
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Number of colors used: {len(set(res))}")
    print(f"First 10 colors: {res[:10]}")
    print(f"Time: {t1 - t0:.2f}s")
