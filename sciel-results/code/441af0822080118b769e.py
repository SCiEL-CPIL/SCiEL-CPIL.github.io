import itertools
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Seed coloring
    n = 16
    colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Precompute all K4s
    k4s = list(itertools.combinations(range(n), 4))
    # Map edges to indices in the colors list
    # Index for (i, j) with i < j is i*n - i*(i+1)//2 + j - i - 1 ?
    # Simpler: use a 2D array or dictionary
    # Let's use a 2D list for fast access
    edge_idx = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
    
    def get_colors_of_k4(k4, col_list):
        c = set()
        for i in range(4):
            for j in range(i+1, 4):
                u, v = k4[i], k4[j]
                c.add(col_list[edge_idx[u][v]])
        return c

    def count_bad_k4s(col_list):
        bad_count = 0
        for k4 in k4s:
            if len(get_colors_of_k4(k4, col_list)) < 5:
                bad_count += 1
        return bad_count

    def is_valid(col_list):
        return count_bad_k4s(col_list) == 0

    # Local search to fix bad K4s
    def repair(col_list, target_k):
        # col_list is a list of colors, each in 0..target_k-1
        # We try to change colors of edges in bad K4s to make them valid
        # We allow using any color in 0..target_k-1
        curr_bad = count_bad_k4s(col_list)
        if curr_bad == 0:
            return col_list
        
        # Identify bad K4s
        bad_k4s = [k4 for k4 in k4s if len(get_colors_of_k4(k4, col_list)) < 5]
        
        # Simulated Annealing / Hill Climbing
        # Temperature
        temp = 1.0
        for step in range(20000):
            if time.time() > deadline:
                break
            
            if not bad_k4s:
                return col_list
            
            # Pick a random bad K4
            k4 = random.choice(bad_k4s)
            # Pick a random edge in this K4
            u, v = random.sample(k4, 2)
            idx_e = edge_idx[u][v]
            old_c = col_list[idx_e]
            
            # Try a new color
            new_c = random.randint(0, target_k - 1)
            if new_c == old_c:
                continue
                
            col_list[idx_e] = new_c
            
            # Recalculate badness for this K4 and its neighbors?
            # For simplicity, just check if this specific K4 is fixed
            # and if we introduced new badness in K4s sharing edges with this one.
            # This is expensive. Let's just do a global check every 100 steps or use a delta.
            # Given 1820 K4s, global check is fast enough in Python if optimized?
            # 1820 * 6 edges * set ops. Maybe ~1ms per check. 20000 steps = 20s. Too slow.
            
            # Better: Track which K4s are bad.
            # When we change an edge (u,v), it affects K4s containing {u,v}.
            # There are (n-2) such K4s.
            
            # Let's implement a delta update.
            # This is complex. Let's stick to a simpler heuristic:
            # Randomly recolor edges in bad K4s, then check global validity periodically.
            
            if step % 50 == 0:
                curr_bad = count_bad_k4s(col_list)
                # Update bad_k4s list
                if curr_bad == 0:
                    return col_list
                bad_k4s = [k4 for k4 in k4s if len(get_colors_of_k4(k4, col_list)) < 5]
            
            # Cooling
            temp *= 0.99
            
        return col_list

    # Try to reduce colors from 17 down to 13
    for target_k in range(16, 12, -1):
        if time.time() > deadline:
            break
            
        # Start from the current best (initially seed 17)
        # We need to map 17 colors to target_k colors.
        # Simple mapping: c % target_k
        # This might create many bad K4s.
        
        # Better: Start from a valid 17-coloring and merge colors one by one.
        # But we are jumping to target_k directly.
        
        # Let's try a few random mappings from 17 to target_k
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Randomly assign each of the 17 seed colors to one of target_k colors
            mapping = [random.randint(0, target_k - 1) for _ in range(17)]
            new_colors = [mapping[c] for c in colors]
            
            # Check if valid
            if is_valid(new_colors):
                return new_colors
            
            # Try to repair
            repaired = repair(new_colors, target_k)
            if is_valid(repaired):
                return repaired

    # If no improvement found, return seed
    return colors
