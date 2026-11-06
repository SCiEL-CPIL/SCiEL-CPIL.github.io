import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (10s buffer)
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Pre-compute K4s and their edges
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
        k4_edges.append(e_indices)
        
    num_k4s = len(k4_edges)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Initial colors: Seed (17 colors)
    # (u + v) % 17 is a known valid construction for this problem
    colors = [0] * num_edges
    for i, (u, v) in enumerate(edges):
        colors[i] = (u + v) % 17
        
    # Verify seed is valid (sanity check)
    # for k4_idx in range(num_k4s):
    #     e_list = k4_edges[k4_idx]
    #     if len({colors[e] for e in e_list}) < 5:
    #         raise ValueError("Seed construction invalid")

    best_colors = colors[:]
    best_num_colors = max(colors) + 1
    
    # Helper to compact colors to 0..k-1
    def compact_colors(cols):
        unique_c = sorted(set(cols))
        if len(unique_c) <= 1:
            return cols[:]
        mapping = {old: new for new, old in enumerate(unique_c)}
        return [mapping[c] for c in cols]
    
    # Fast validity check for changing one edge
    def can_change_edge(e_idx, new_color, colors):
        old_color = colors[e_idx]
        if old_color == new_color:
            return True
        
        for k4_idx in edge_to_k4s[e_idx]:
            e_list = k4_edges[k4_idx]
            # Calculate colors of other 5 edges
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

    # Local Search Loop
    # We use a greedy descent with random kicks to escape local minima.
    # The objective is to minimize the number of distinct colors.
    
    no_improve_count = 0
    current_best = best_num_colors
    
    # Track used colors to allow efficient random kicks within range
    # We'll just use a fixed range for kicks based on best_num_colors
    
    while time.time() < deadline:
        improved = False
        
        # Batch of random edge selections for descent
        # We try to find the minimum valid color for a random edge
        for _ in range(500):
            if time.time() > deadline:
                break
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
        
        if improved:
            no_improve_count = 0
            # Check if we have a new best
            # We compact to see if unique count dropped
            # Doing this every improved step might be slightly expensive but ensures we track best accurately
            # However, we can just check max(colors) first as a cheap filter
            if max(colors) < current_best:
                compacted = compact_colors(colors)
                num_unique = max(compacted) + 1
                if num_unique < current_best:
                    current_best = num_unique
                    best_num_colors = num_unique
                    best_colors = compacted
                    colors = compacted # Keep search state compacted to reduce range
        else:
            no_improve_count += 1
            
            # Kick: Try to change a random edge to a random valid color
            # This might increase the color count, but allows escape from local minima
            e_idx = random.randint(0, num_edges - 1)
            current_c = colors[e_idx]
            
            # Try a few random candidates
            found_kick = False
            # Range: 0 to best_num_colors + 5 (allow some expansion to explore)
            # Using a slightly larger range helps escape deep local minima
            max_kick_color = current_best + 5
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
                # Or just continue.
                pass

        # Periodic full compaction to ensure we are tracking the best possible state
        # and to normalize the color space for future kicks
        if no_improve_count % 10 == 0 or improved:
             # Only compact if it might improve the state or if we are stuck
             # Compacting when stuck doesn't change the set of colors, just their labels
             # But it ensures max(colors) is tight, which helps the "max < current_best" check
             pass

    # Final compact best
    unique_c = sorted(set(best_colors))
    if len(unique_c) > 1:
        mapping = {old: new for new, old in enumerate(unique_c)}
        for i in range(num_edges):
            best_colors[i] = mapping[best_colors[i]]
            
    return best_colors

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Colors used: {max(res) + 1}")
    # Verify
    n = 16
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            idx += 1
    valid = True
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
        if len({res[e] for e in e_indices}) < 5:
            valid = False
            break
    print(f"Valid: {valid}")
