import time
import random

def construct():
    deadline = time.time() + 115
    N = 50
    num_edges = N * (N - 1) // 2
    
    # Edge mapping
    edges = []
    edge_to_idx = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(edges)
            edges.append((i, j))
            
    # Precompute neighbors for each edge: list of (w, idx_uw, idx_vw)
    edge_neighbors = [[] for _ in range(num_edges)]
    for idx, (u, v) in enumerate(edges):
        for w in range(N):
            if w == u or w == v:
                continue
            if u < w:
                idx_uw = edge_to_idx[(u, w)]
            else:
                idx_uw = edge_to_idx[(w, u)]
                
            if v < w:
                idx_vw = edge_to_idx[(v, w)]
            else:
                idx_vw = edge_to_idx[(w, v)]
                
            edge_neighbors[idx].append((w, idx_uw, idx_vw))

    def count_mono_triangles(colors):
        count = 0
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                c_ij = colors[edge_to_idx[(i, j)]]
                for k in range(j + 1, N):
                    if colors[edge_to_idx[(j, k)]] == c_ij and colors[edge_to_idx[(i, k)]] == c_ij:
                        count += 1
        return count

    def get_mono_triangles(colors):
        tri_list = []
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                c_ij = colors[edge_to_idx[(i, j)]]
                for k in range(j + 1, N):
                    if colors[edge_to_idx[(j, k)]] == c_ij and colors[edge_to_idx[(i, k)]] == c_ij:
                        tri_list.append((i, j, k))
        return tri_list

    best_n = 32
    # Fallback coloring for n=32 (Greenwood-Gleason)
    # We'll generate a valid one quickly if we fail n=50
    def generate_fallback():
        # Standard GG coloring for n=16, doubled for n=32
        # Actually, let's just return a simple valid n=16 or n=32 if we fail
        # The prompt says "Maximize n". If we can't do 50, we should return the best we have.
        # But we know 50 is possible. We must find it.
        pass

    attempts = 0
    max_attempts = 10000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        mono_count = count_mono_triangles(colors)
        
        # Greedy Repair
        temperature = 1.0
        min_temp = 0.1
        cooling = 0.999
        steps = 0
        max_steps = 50000
        no_improve = 0
        no_improve_limit = 1000
        
        while mono_count > 0 and steps < max_steps and time.time() < deadline:
            steps += 1
            # Get a random mono triangle
            # For efficiency, we can maintain a list, but rebuilding is O(N^3) = 125k ops.
            # Doing it every step is too slow. 
            # Instead, pick random edges and check if they form a mono triangle with a random 3rd vertex.
            # Or, maintain a list of mono triangles and update it.
            
            # Let's use a simpler approach: Pick random edge, if it's in a mono triangle, fix it.
            # If not, pick random edge again.
            
            # To be more directed, let's maintain the list of mono triangles.
            # Updating the list is tricky.
            
            # Alternative: Simulated Annealing on the energy (number of mono triangles).
            # Pick a random edge. Calculate delta for each new color.
            
            edge_idx = random.randint(0, num_edges - 1)
            u, v = edges[edge_idx]
            old_c = colors[edge_idx]
            
            best_delta = float('inf')
            best_new_c = -1
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                
                created = 0
                destroyed = 0
                
                for w, idx_uw, idx_vw in edge_neighbors[edge_idx]:
                    c_uw = colors[idx_uw]
                    c_vw = colors[idx_vw]
                    
                    if c_uw == c_vw == old_c:
                        destroyed += 1
                    if c_uw == c_vw == new_c:
                        created += 1
                        
                delta = created - destroyed
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
                    
            if best_new_c == -1:
                continue
                
            # Acceptance criterion
            accept = False
            if best_delta < 0:
                accept = True
            elif best_delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                # Simulated annealing
                if random.random() < pow(2.71828, -best_delta / temperature):
                    accept = True
            
            if accept:
                colors[edge_idx] = best_new_c
                mono_count += best_delta
                
                if mono_count < 0:
                    mono_count = 0 # Safety, shouldn't happen if delta is correct
                
                if best_delta < 0:
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if no_improve > no_improve_limit:
                    # Restart or cool down
                    temperature *= 0.5
                    no_improve = 0
                    if temperature < min_temp:
                        break # Break to new restart
            
            temperature *= cooling
            if temperature < min_temp:
                temperature = min_temp
                
        if mono_count == 0:
            return {'n': N, 'colors': colors}

    # If we fail to find n=50, we should return the best valid coloring we can generate quickly.
    # A valid n=16 coloring is easy.
    # Let's construct the Greenwood-Gleason n=16 coloring.
    def gg_coloring_16():
        # GF(16)
        # Polynomial x^4 + x + 1
        # We need log map for GF(16)
        # This is complex to implement from scratch in the fallback.
        # Simpler fallback: Random search for n=16?
        # Or just return n=1?
        # The task says "Maximize n". If we can't do 50, we must return something valid.
        # But we *should* be able to do 50.
        # If we fail, let's try a quick search for n=16 to ensure non-zero score.
        # Actually, the previous program returned n=32. We should try to return at least that.
        # Let's implement a quick n=32 generator.
        
        # The GG coloring of K_16 is known.
        # K_32 can be formed by two copies of K_16 with all cross edges colored 3.
        # Wait, if cross edges are all 3, and K_16 has no mono triangles, does K_32?
        # A triangle with vertices in different copies has 2 vertices in one copy, 1 in other.
        # Edges within copy: colors from GG.
        # Edges to other copy: color 3.
        # If the two vertices in the copy have edge color 3, and the cross edges are 3, we have a mono triangle.
        # So we need the GG coloring of K_16 to NOT contain any edge of color 3?
        # The GG coloring of K_16 uses 3 colors (0,1,2). It is a 3-coloring of K_16 without mono triangles.
        # So if we use colors 0,1,2 for the two K_16s, and color 3 for all cross edges,
        # then any triangle with vertices in different copies has at least one edge of color 3 (the cross edge).
        # For it to be mono, ALL edges must be 3.
        # But the edge within the copy is 0, 1, or 2. So it can't be 3.
        # Thus, no mono triangles.
        # So n=32 is valid.
        
        # Let's generate the GG K_16 coloring.
        # This requires implementing GF(16) arithmetic.
        # It's a bit of code, but ensures a baseline of 32 if 50 fails.
        
        def mul(a, b):
            r = 0
            for i in range(4):
                if (b >> i) & 1:
                    r ^= a << i
            # Modulo x^4 + x + 1
            # x^4 = x + 1
            # x^5 = x^2 + x
            # x^6 = x^3 + x^2
            # x^7 = x^4 + x^3 = x^3 + x + 1
            for i in range(7, 3, -1):
                if (r >> i) & 1:
                    r ^= 0b10011 << (i - 4)
            return r

        log_map = {}
        x = 1
        for e in range(15):
            log_map[x] = e
            x = mul(x, 2)

        def edge_color_core(i, j):
            # Map to GF(16) elements 1..15?
            # Usually vertices are 0..15.
            # If i==j, undefined.
            # The coloring is based on the ratio or difference?
            # For K_16, vertices are elements of GF(16).
            # Edge (u,v) color is log(u/v) mod 3? Or log(u-v) mod 3?
            # Standard construction: Color edge (u,v) by log(u-v) mod 3.
            # Since u!=v, u-v is non-zero.
            # There are 15 non-zero elements. 15/3 = 5.
            # Each color class is a set of 5 edges? No, 5 edges per color?
            # Total edges 120. 120/3 = 40.
            # log(u-v) takes values 0..14.
            # Values 0,3,6,9,12 -> Color 0
            # Values 1,4,7,10,13 -> Color 1
            # Values 2,5,8,11,14 -> Color 2
            return log_map[i ^ j] % 3

        n_base = 32
        base_colors = [0] * (n_base * (n_base - 1) // 2)
        idx = 0
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if i // 16 == j // 16:
                    c = edge_color_core(i % 16, j % 16)
                else:
                    c = 3
                base_colors[idx] = c
                idx += 1
        return base_colors

    fallback_colors = gg_coloring_16()
    return {'n': 32, 'colors': fallback_colors}
