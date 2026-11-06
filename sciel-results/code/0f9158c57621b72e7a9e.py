import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Pre-compute K4s and their edges
    k4s = []
    k4_edges = [] # list of list of edge indices
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_indices = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        k4s.append(comb)
        k4_edges.append(e_indices)
        
    num_k4s = len(k4s)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Initial colors: 1-factorization of K16
    # Vertices 0..14 and 15 (infinity)
    # Matchings M_k for k in 0..14
    # M_k contains (15, k) and ( (k+i)%15, (k-i)%15 ) for i=1..7
    colors = [0] * num_edges
    for k in range(15):
        # Edge (15, k)
        e_idx = edge_index[(min(k, 15), max(k, 15))]
        colors[e_idx] = k
        
        for i in range(1, 8):
            u = (k + i) % 15
            v = (k - i) % 15
            if u > v:
                u, v = v, u
            e_idx = edge_index[(u, v)]
            colors[e_idx] = k

    # Verify seed is valid
    seed_valid = True
    for k4_idx in range(num_k4s):
        e_list = k4_edges[k4_idx]
        if len({colors[e] for e in e_list}) < 5:
            seed_valid = False
            break
    
    if not seed_valid:
        # Fallback to (u+v)%17
        for i, (u, v) in enumerate(edges):
            colors[i] = (u + v) % 17

    best_colors = colors[:]
    best_num_colors = max(colors) + 1
    
    # Optimization: Pre-compute the set of K4s for each edge for faster access
    # We already have edge_to_k4s.
    
    def can_change_edge(e_idx, new_color, colors):
        old_color = colors[e_idx]
        if old_color == new_color:
            return True
        
        for k4_idx in edge_to_k4s[e_idx]:
            e_list = k4_edges[k4_idx]
            # Calculate colors of other 5 edges
            # We can avoid creating a new set every time if we are careful,
            # but for 6 edges, set creation is cheap.
            other_colors = set()
            for e in e_list:
                if e != e_idx:
                    other_colors.add(colors[e])
            
            # The set of colors in this K4 will be other_colors U {new_color}
            # It is valid if len(other_colors U {new_color}) >= 5
            
            # Optimization:
            # If new_color in other_colors: size = len(other_colors)
            # Else: size = len(other_colors) + 1
            
            if new_color in other_colors:
                if len(other_colors) < 5:
                    return False
            else:
                if len(other_colors) < 4:
                    return False
        return True

    def compact_colors(colors):
        unique_c = sorted(set(colors))
        if len(unique_c) <= 1:
            return colors
        mapping = {old: new for new, old in enumerate(unique_c)}
        new_colors = [mapping[c] for c in colors]
        return new_colors

    # Local Search Loop
    # We use a simple simulated annealing-like approach or just greedy with kicks.
    # Given the small size, greedy with kicks works well.
    
    iter_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        # Try to improve
        improved = False
        
        # Randomly select an edge to try to reduce
        # We can also iterate through all edges if we want, but random is faster for kicks
        # Let's try a batch of random edges
        for _ in range(1000):
            e_idx = random.randint(0, num_edges - 1)
            current_c = colors[e_idx]
            
            # Try to lower
            # Find the minimum valid color
            # We can try 0, 1, ... up to current_c - 1
            # If current_c is 0, we can't lower.
            
            min_valid = -1
            for c in range(0, current_c):
                if can_change_edge(e_idx, c, colors):
                    min_valid = c
                    break
            
            if min_valid != -1:
                colors[e_idx] = min_valid
                improved = True
                # Update best if we reduced the max color?
                # We should check the actual number of colors used
                # To be efficient, we can just track max and check compacted size periodically
                # Or just compact immediately if we suspect a reduction in unique count?
                # Compacting every step is expensive.
                # Let's just update best_colors if max(color) < best_num_colors?
                # Actually, max(color) is not the same as number of unique colors.
                # But if we lower a color, the max might not drop.
                # However, if we lower a color to a value already in use, unique count stays same.
                # If we lower a color to a value NOT in use, unique count increases? No, we are lowering.
                # If we lower 5 to 3, and 3 is already there, unique count same.
                # If we lower 5 to 3, and 3 is NOT there, unique count same (5 replaced by 3).
                # So lowering never increases unique count.
                # It can decrease unique count if the old color was unique and the new color is shared.
                
                # Let's just check if we can compact to fewer colors
                # We'll do a quick check: if max(colors) < best_num_colors, it's a candidate.
                # But best_num_colors is the count, not the max index.
                # Let's maintain best_num_colors as the count of unique colors.
                
                # To avoid expensive set() calls every step, we'll just record the state
                # and verify/compact at the end of the batch or if max drops.
                pass
        
        if improved:
            no_improve_count = 0
            # Check if we have a new best
            # We can approximate by checking if max(colors) < best_num_colors
            # If so, compact and update
            if max(colors) < best_num_colors:
                compacted = compact_colors(colors)
                num_unique = max(compacted) + 1
                if num_unique < best_num_colors:
                    best_num_colors = num_unique
                    best_colors = compacted
                    colors = compacted # Keep search state compacted
        else:
            no_improve_count += 1
            
            # Kick: Try to change a random edge to a random valid color
            # This might increase the color count, but allows escape from local minima
            e_idx = random.randint(0, num_edges - 1)
            current_c = colors[e_idx]
            
            # Try a few random candidates
            found_kick = False
            # Range: 0 to best_num_colors + 2 (allow some expansion)
            max_kick_color = best_num_colors + 5
            for _ in range(20):
                new_c = random.randint(0, max_kick_color)
                if new_c == current_c:
                    continue
                if can_change_edge(e_idx, new_c, colors):
                    colors[e_idx] = new_c
                    found_kick = True
                    break
            
            if not found_kick:
                # If we can't even kick, maybe we are in a very tight spot.
                # Try a more aggressive kick: allow larger colors?
                # Or just continue, hoping another edge works.
                pass

        iter_count += 1
        if iter_count % 5000 == 0:
            # Periodic compaction of current state to tidy up
            compacted = compact_colors(colors)
            num_unique = max(compacted) + 1
            if num_unique < best_num_colors:
                best_num_colors = num_unique
                best_colors = compacted
            # Always use compacted state for search to keep color space small
            colors = compacted

    # Final compact best
    final_compact = compact_colors(best_colors)
    return final_compact
