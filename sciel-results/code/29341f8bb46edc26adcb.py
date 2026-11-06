import random
import time
import itertools

def construct():
    deadline = time.time() + 115
    N = 16
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    E = len(edges)
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Pre-compute K4 edge sets
    k4s = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                    edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(k4_edges)
        
    # Map each edge to the K4s it belongs to
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, k4_edges in enumerate(k4s):
        for e in k4_edges:
            edge_to_k4s[e].append(k4_idx)

    # Helper to calculate violations for a specific K4 given colors
    def get_k4_violation(colors, k4_edges):
        # Count distinct colors
        # Since max colors is small (<=15), we can use a small set or bitset
        # Set is fine for 6 elements
        distinct = len(set(colors[e] for e in k4_edges))
        return max(0, 5 - distinct)

    def try_find_coloring(max_colors, time_limit):
        rng = random.Random()
        # Initialize randomly
        colors = [rng.randint(0, max_colors - 1) for _ in range(E)]
        
        # Calculate initial metrics
        color_counts = [0] * max_colors
        for c in colors:
            color_counts[c] += 1
        palette_size = sum(1 for c in color_counts if c > 0)
        
        total_violations = 0
        for k4e in k4s:
            total_violations += get_k4_violation(colors, k4e)
            
        best_colors = None
        best_violations = total_violations
        best_palette = palette_size
        
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                T = T_min
                
            # Pick random edge
            e = rng.randint(0, E - 1)
            c_old = colors[e]
            c_new = rng.randint(0, max_colors - 1)
            
            if c_new == c_old:
                continue
                
            # Calculate delta violations
            delta_viol = 0
            for k4_idx in edge_to_k4s[e]:
                k4e = k4s[k4_idx]
                # Old distinct
                old_colors = [colors[ee] for ee in k4e]
                old_dist = len(set(old_colors))
                old_viol = max(0, 5 - old_dist)
                
                # New distinct
                # We need to replace c_old with c_new in the set
                # Optimization:
                # If c_old == c_new, no change (handled)
                # If c_new is already in the set of other 5 edges, distinct count doesn't change
                # If c_old was the only instance of its color, distinct count decreases by 1
                # If c_old is still present, distinct count doesn't change
                # If c_new is not present, distinct count increases by 1
                
                # Let's just compute set for simplicity, 6 elements is fast
                new_colors = [c_new if ee == e else colors[ee] for ee in k4e]
                new_dist = len(set(new_colors))
                new_viol = max(0, 5 - new_dist)
                
                delta_viol += (new_viol - old_viol)
                
            # Calculate delta palette
            # Update counts tentatively
            color_counts[c_old] -= 1
            color_counts[c_new] += 1
            
            delta_pal = 0
            if color_counts[c_old] == 0:
                delta_pal -= 1
            if color_counts[c_new] == 1:
                delta_pal += 1
                
            # Cost
            # We want 0 violations. Palette size is secondary.
            # If we have violations, we must fix them.
            # Cost = violations * 100 + palette_size
            delta_cost = delta_viol * 100 + delta_pal
            
            # Metropolis
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                if delta_cost > 50 * T:
                    accept = False
                else:
                    ratio = delta_cost / T
                    if ratio < 700:
                        accept = rng.random() < (2.718281828 ** (-ratio))
            
            if accept:
                colors[e] = c_new
                total_violations += delta_viol
                palette_size += delta_pal
                
                if total_violations == 0:
                    if palette_size < best_palette:
                        best_palette = palette_size
                        best_colors = colors[:]
                        if best_palette == max_colors: # Can't get better than using all allowed? No, we want min palette.
                            # Actually if we restrict to max_colors, min palette is what we want.
                            # If we found a valid one, we can try to reduce palette further?
                            # But we are in a fixed max_colors loop.
                            # If we find valid with K colors, we return.
                            # But maybe we can find valid with K colors but using fewer?
                            # The objective in SA is min palette. So it will try to use fewer.
                            pass
            else:
                # Revert
                color_counts[c_new] -= 1
                color_counts[c_old] += 1
                
            T *= cooling
            
        # After time limit, check if best is valid
        if best_violations == 0:
            return best_colors
        return None

    # Try K=13, 14, 15
    # Allocate time
    # 13 is hardest, 15 is easiest.
    # Give 13 a chance, then 14, then 15.
    # If 15 is the upper bound, it should be solvable.
    
    remaining = deadline - time.time()
    
    # Try 13
    t13 = min(20, remaining * 0.3)
    res = try_find_coloring(13, t13)
    if res:
        return res
        
    remaining = deadline - time.time()
    if remaining < 5:
        # Fallback
        return [(i + j) % 17 for (i, j) in edges]
        
    # Try 14
    t14 = min(40, remaining * 0.5)
    res = try_find_coloring(14, t14)
    if res:
        return res
        
    remaining = deadline - time.time()
    if remaining < 5:
        return [(i + j) % 17 for (i, j) in edges]
        
    # Try 15
    t15 = remaining - 2
    res = try_find_coloring(15, t15)
    if res:
        return res
        
    # Fallback to known 17-coloring if all fail
    return [(i + j) % 17 for (i, j) in edges]
