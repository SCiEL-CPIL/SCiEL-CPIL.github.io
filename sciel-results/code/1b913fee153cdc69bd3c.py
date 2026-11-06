import itertools
import time
import random

def construct():
    # Precompute cycles for n up to 17? 
    # Memory: 17 -> ~2M cycles? 
    # Binom(17,6) = 12376. * 60 = 742,560.
    # This is fine.
    
    def get_cycles_6(n):
        cycles = []
        for verts in itertools.combinations(range(n), 6):
            min_v = min(verts)
            for perm in itertools.permutations(verts):
                if perm[0] != min_v:
                    continue
                if perm[1] > perm[5]:
                    continue
                edges = []
                for i in range(6):
                    u, v = perm[i], perm[(i+1)%6]
                    if u > v: u, v = v, u
                    edges.append((u, v))
                cycles.append(frozenset(edges))
        return list(set(cycles))

    # Map edge to index
    def edge_to_idx(i, j, n):
        if i > j: i, j = j, i
        return i * (n - 1) - (i * (i - 1) // 2) + (j - i - 1)

    def solve(n, time_limit=100):
        start_time = time.time()
        m = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(m)]
        
        cycles = get_cycles_6(n)
        
        # Precompute for each edge, the list of cycles it belongs to
        edge_to_cycles = {e: [] for e in itertools.combinations(range(n), 2)}
        for c in cycles:
            for e in c:
                edge_to_cycles[e].append(c)
                
        # Helper to check if a cycle is monochromatic
        def is_mono(cycle, colors):
            col = -1
            for e in cycle:
                i, j = e
                idx = edge_to_idx(i, j, n)
                c_val = colors[idx]
                if col == -1:
                    col = c_val
                elif col != c_val:
                    return False
            return True

        # Initial conflict count
        total_conflicts = 0
        # We don't need per-edge count if we just pick a random conflicting cycle.
        # But picking a random one is slow to find.
        # Let's maintain a list of conflicting cycles? 
        # Or just scan? 
        # Scanning 700k cycles takes ~0.5s.
        # If we scan every iteration, we do ~200 iterations in 100s.
        # This is likely enough for Local Search.
        
        while time.time() - start_time < time_limit:
            # Find a conflicting cycle
            conf_cycle = None
            for c in cycles:
                if is_mono(c, colors):
                    conf_cycle = c
                    break
            
            if conf_cycle is None:
                return colors
                
            # Pick a random edge from the conflicting cycle
            e = random.choice(list(conf_cycle))
            i, j = e
            idx = edge_to_idx(i, j, n)
            
            # Try all 4 colors
            best_color = colors[idx]
            min_conf = 1 # We know there is at least 1 conflict
            
            # Optimization: Only count conflicts for cycles containing e?
            # No, changing e might break other cycles not containing e? No, only cycles containing e are affected.
            # So we can calculate the delta.
            # Current conflicts involving e:
            # This is complex. Let's just do full scan if n is small enough.
            # For n=17, full scan is ~0.5s.
            # So we can do ~200 iterations.
            
            for c in range(4):
                if c == colors[idx]:
                    continue
                colors[idx] = c
                # Count conflicts
                conf = 0
                for cyc in cycles:
                    if is_mono(cyc, colors):
                        conf += 1
                        if conf >= min_conf: # Early exit if already worse
                            break
                if conf < min_conf:
                    min_conf = conf
                    best_color = c
                if conf == 0:
                    break
            
            colors[idx] = best_color
            
        return None

    for n in range(17, 8, -1):
        time_limit = 50 if n >= 17 else (30 if n == 16 else 10)
        result = solve(n, time_limit)
        if result is not None:
            # Final verification
            cycles = get_cycles_6(n)
            valid = True
            for c in cycles:
                if is_mono(c, result):
                    valid = False
                    break
            if valid:
                return {'n': n, 'colors': result}
                
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

# EVOLVE-BLOCK-END
