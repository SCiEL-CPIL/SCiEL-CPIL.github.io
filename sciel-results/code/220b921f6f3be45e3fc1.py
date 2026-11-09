"""Optimized local search for edge-colourings of K16 with every K4 spanning >=5 colours."""
import time
import random
import itertools

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    N = 16
    # Precompute edge index mapping: (i,j) -> idx in [0,119]
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[(i,j)] = idx
            idx += 1
    # 120 edges total
    E = 120
    
    # Precompute all K4s and the 6 edges in each
    all_k4s = list(itertools.combinations(range(N), 4))
    k4_edges = []
    for k4 in all_k4s:
        edges = []
        for a, b in itertools.combinations(k4, 2):
            edges.append(edge_idx[(a,b)])
        k4_edges.append(edges)
    
    # Precompute for each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
    
    def count_colors_and_bad(colors):
        """Return (num_colors_used, num_bad_k4s)"""
        color_set = set(colors)
        bad = 0
        for edges in k4_edges:
            # Check if these 6 edges span at least 5 distinct colours
            col_set = set(colors[e] for e in edges)
            if len(col_set) < 5:
                bad += 1
        return len(color_set), bad
    
    def count_bad_k4s_fast(colors):
        """Count bad K4s using precomputed structures"""
        bad = 0
        for edges in k4_edges:
            # Quick check: count distinct colours
            c0 = colors[edges[0]]
            c1 = colors[edges[1]]
            c2 = colors[edges[2]]
            c3 = colors[edges[3]]
            c4 = colors[edges[4]]
            c5 = colors[edges[5]]
            # Count distinct among 6
            distinct = 1
            if c1 != c0: distinct += 1
            if c2 != c0 and c2 != c1: distinct += 1
            if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
            if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
            if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
            if distinct < 5:
                bad += 1
        return bad
    
    def count_colors_used(colors):
        return len(set(colors))
    
    best_colors = None
    best_bad = float('inf')
    best_num_colors = float('inf')
    
    # Try multiple random restarts with local search
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with a random colouring using a small palette
        # Start with 13 colours (the theoretical lower bound)
        palette_size = 13
        colors = [random.randint(0, palette_size - 1) for _ in range(E)]
        
        # Local search: flip random edges to reduce bad K4s
        bad_count = count_bad_k4s_fast(colors)
        num_colors = count_colors_used(colors)
        
        if bad_count < best_bad or (bad_count == best_bad and num_colors < best_num_colors):
            best_bad = bad_count
            best_num_colors = num_colors
            best_colors = colors[:]
        
        if bad_count == 0:
            break
        
        # Simulated annealing / hill climbing
        temperature = 50.0
        cooling = 0.95
        step = 0
        max_steps = 50000
        no_improve = 0
        max_no_improve = 2000
        
        while step < max_steps and time.time() < deadline:
            if no_improve > max_no_improve:
                break
            
            # Pick a random edge
            e = random.randint(0, E - 1)
            old_color = colors[e]
            new_color = random.randint(0, palette_size - 1)
            if new_color == old_color:
                continue
            
            # Compute delta in bad K4s
            delta_bad = 0
            for k4_idx in edge_to_k4s[e]:
                edges = k4_edges[k4_idx]
                # Old count
                old_cols = set(colors[ed] for ed in edges)
                old_distinct = len(old_cols)
                # New count (simulate change)
                new_cols = set(colors[ed] for ed in edges)
                new_cols.discard(old_color)
                new_cols.add(new_color)
                # But we need to be careful: only edge e changes
                # Actually, let's just compute directly
                # For efficiency, let's just count
                delta_bad += (1 if len(new_cols) < 5 else 0) - (1 if old_distinct < 5 else 0)
            
            if delta_bad < 0 or (delta_bad == 0 and random.random() < 0.3):
                colors[e] = new_color
                bad_count += delta_bad
                no_improve = 0
                if bad_count < best_bad or (bad_count == best_bad and count_colors_used(colors) < best_num_colors):
                    best_bad = bad_count
                    best_num_colors = count_colors_used(colors)
                    best_colors = colors[:]
                if bad_count == 0:
                    break
            else:
                # Accept with probability based on temperature
                if delta_bad > 0 and random.random() < (0.5 ** delta_bad):
                    colors[e] = new_color
                    bad_count += delta_bad
                no_improve += 1
            
            step += 1
            if step % 1000 == 0:
                temperature *= cooling
        
        # After local search, try to reduce palette
        if best_bad == 0 and best_colors is not None:
            # Try to compress colours
            for target_palette in range(13, palette_size + 1):
                if target_palette >= palette_size:
                    break
                # Try to remap colours to use only target_palette colours
                test_colors = best_colors[:]
                # Build mapping
                color_map = {}
                next_color = 0
                for c in test_colors:
                    if c not in color_map:
                        if next_color < target_palette:
                            color_map[c] = next_color
                            next_color += 1
                        else:
                            # Keep as is (shouldn't happen if we check)
                            color_map[c] = c
                test_colors = [color_map[c] for c in test_colors]
                if count_bad_k4s_fast(test_colors) == 0:
                    best_colors = test_colors
                    best_num_colors = target_palette
                    break
    
    if best_colors is None:
        # Fallback: use the seed program's approach
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_colors
