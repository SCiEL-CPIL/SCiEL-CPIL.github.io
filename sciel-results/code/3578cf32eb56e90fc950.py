import time
import random
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}
    
    best_result = get_fallback(5)
    
    def precompute(n):
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        def edge_index(u, v):
            if u > v:
                u, v = v, u
            return edge_map[(u, v)]
        
        vertices = list(range(n))
        cycles = []
        edge_to_cycles = [[] for _ in range(n * (n - 1) // 2)]
        
        # Generate all 6-cycles
        # A 6-cycle is a sequence of 6 distinct vertices v0...v5 such that
        # edges (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0) exist.
        # In K_n, all edges exist.
        # To avoid duplicates, we can fix v0=0? No, we need all cycles.
        # Standard way: iterate over combinations of 6 vertices, then all cyclic permutations?
        # Number of 6-cycles in K_n is (n choose 6) * 60.
        
        for combo in itertools.combinations(vertices, 6):
            v0, v1, v2, v3, v4, v5 = combo
            # There are 60 distinct cycles on these 6 vertices.
            # We can generate them by permuting v1..v5 and fixing v0?
            # No, fixing v0 and permuting v1..v5 gives 5! = 120 permutations.
            # Each cycle is counted twice (forward/backward).
            # So 120 / 2 = 60 cycles.
            
            # To be efficient, we can just iterate permutations of [v1..v5]
            # and skip if v1 > v5 to avoid reverse duplicates?
            # Actually, just generate all and deduplicate? No, too slow.
            
            # Let's use the property: fix v0. The cycle is v0-v1-v2-v3-v4-v5-v0.
            # Reverse is v0-v5-v4-v3-v2-v1-v0.
            # So we can just iterate permutations of [v1, v2, v3, v4, v5]
            # and only keep those where v1 < v5?
            
            for perm in itertools.permutations([v1, v2, v3, v4, v5]):
                if perm[0] > perm[-1]:
                    continue
                
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    e = edge_index(u, v)
                    cycle_edges.append(e)
                
                cycle_id = len(cycles)
                cycles.append(cycle_edges)
                for e in cycle_edges:
                    edge_to_cycles[e].append(cycle_id)
        
        return edge_map, cycles, edge_to_cycles

    def search_n(n, time_budget):
        if time.time() > deadline - 5:
            return None
            
        try:
            edge_map, cycles, edge_to_cycles = precompute(n)
        except Exception:
            return None
            
        m = n * (n - 1) // 2
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}
        
        # Helper to check if a cycle is monochromatic
        # Returns color if mono, -1 otherwise
        def check_cycle_mono(ce, colors):
            c0 = colors[ce[0]]
            for e in ce[1:]:
                if colors[e] != c0:
                    return -1
            return c0
        
        # Helper to get bad count
        def get_bad_count(colors):
            count = 0
            for ce in cycles:
                if check_cycle_mono(ce, colors) != -1:
                    count += 1
            return count
        
        # Local Search
        best_bad = float('inf')
        best_colors = None
        
        restarts = 0
        while time.time() < time_budget + start_time and time.time() < deadline - 5:
            restarts += 1
            # Random Init
            colors = [random.randint(0, 3) for _ in range(m)]
            bad_count = get_bad_count(colors)
            
            if bad_count == 0:
                return {'n': n, 'colors': colors}
            
            if bad_count < best_bad:
                best_bad = bad_count
                best_colors = colors[:]
            
            # Hill Climbing
            no_improve_count = 0
            while time.time() < deadline - 5:
                improved = False
                # Try a few random moves
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    
                    e = random.randint(0, m - 1)
                    c_old = colors[e]
                    
                    # Try all 3 other colors
                    best_delta = float('inf')
                    best_c_new = -1
                    
                    for c_new in range(4):
                        if c_new == c_old:
                            continue
                        
                        # Calculate delta
                        delta = 0
                        for cid in edge_to_cycles[e]:
                            ce = cycles[cid]
                            # Check if currently bad
                            # We can optimize this by precomputing counts, but let's try direct check first
                            # Direct check is 5 comparisons.
                            # 16000 cycles * 5 = 80k ops.
                            # This might be too slow if done for 3 colors * 100 moves.
                            # 3 * 100 * 80k = 24M ops per "step" of 100 moves.
                            # 24M ops in Python is ~2-5 seconds.
                            # So we can do ~20-50 "steps" per second.
                            # This is slow.
                            
                            # Optimization: Maintain counts per cycle.
                            pass
                        
                        # Since direct check is slow, let's switch to maintaining counts.
                        # We'll break out of this loop and implement the count-based version.
                        break
                    
                    break # Break out of move loop to implement count version
                
                break # Break out of hill climbing to implement count version
        
        return None

    # Re-implement search_n with count-based updates for speed
    
    def search_n_fast(n, time_budget):
        if time.time() > deadline - 5:
            return None
            
        try:
            edge_map, cycles, edge_to_cycles = precompute(n)
        except Exception:
            return None
            
        m = n * (n - 1) // 2
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}
        
        # Precompute cycle edge lists as tuples for faster access?
        # cycles is list of lists of edge indices.
        
        # Initialize counts
        # cycle_color_counts[cid] = [c0, c1, c2, c3]
        
        best_result = None
        
        while time.time() < deadline - 5:
            # Random Init
            colors = [random.randint(0, 3) for _ in range(m)]
            
            # Compute initial counts
            cycle_color_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
            for cid, ce in enumerate(cycles):
                counts = cycle_color_counts[cid]
                for e in ce:
                    counts[colors[e]] += 1
            
            bad_count = sum(1 for c in cycle_color_counts if max(c) == 6)
            
            if bad_count == 0:
                return {'n': n, 'colors': colors}
            
            best_bad_in_restart = bad_count
            best_colors_in_restart = colors[:]
            
            # Local Search
            while time.time() < deadline - 5:
                # Try to find an improving move
                found_improvement = False
                # Sample random edges
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    
                    e = random.randint(0, m - 1)
                    c_old = colors[e]
                    
                    # Try best color change
                    best_delta = 0 # 0 means no change, we want < 0
                    best_c_new = -1
                    current_bad = bad_count
                    
                    for c_new in range(4):
                        if c_new == c_old:
                            continue
                        
                        delta = 0
                        for cid in edge_to_cycles[e]:
                            counts = cycle_color_counts[cid]
                            old_cnt = counts[c_old]
                            new_cnt = counts[c_new]
                            
                            # Was it bad?
                            # If old_cnt == 6, it was bad with c_old.
                            # Note: if old_cnt == 6, no other count is 6.
                            was_bad = (old_cnt == 6)
                            
                            # Update counts virtually
                            # counts[c_old] becomes old_cnt - 1
                            # counts[c_new] becomes new_cnt + 1
                            
                            # Is it bad now?
                            # It is bad if max(new counts) == 6.
                            # Since sum is 6, only one can be 6.
                            # It can only be c_new if new_cnt + 1 == 6.
                            # Or some other color if... no, other colors don't change.
                            # So it is bad now if new_cnt + 1 == 6.
                            is_bad = (new_cnt + 1 == 6)
                            
                            delta += (1 if is_bad else 0) - (1 if was_bad else 0)
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_c_new = c_new
                    
                    if best_c_new != -1 and best_delta < 0:
                        # Apply move
                        colors[e] = best_c_new
                        for cid in edge_to_cycles[e]:
                            counts = cycle_color_counts[cid]
                            counts[c_old] -= 1
                            counts[best_c_new] += 1
                        bad_count += best_delta
                        found_improvement = True
                        
                        if bad_count < best_bad_in_restart:
                            best_bad_in_restart = bad_count
                            best_colors_in_restart = colors[:]
                            
                        if bad_count == 0:
                            return {'n': n, 'colors': colors}
                        
                        break # Restart sampling after a move
                
                if not found_improvement:
                    # Perturb
                    # Flip 3-5 random edges
                    num_flips = random.randint(3, 5)
                    for _ in range(num_flips):
                        e = random.randint(0, m - 1)
                        c_old = colors[e]
                        c_new = random.randint(0, 3)
                        while c_new == c_old:
                            c_new = random.randint(0, 3)
                        
                        delta = 0
                        for cid in edge_to_cycles[e]:
                            counts = cycle_color_counts[cid]
                            old_cnt = counts[c_old]
                            new_cnt = counts[c_new]
                            was_bad = (old_cnt == 6)
                            is_bad = (new_cnt + 1 == 6)
                            delta += (1 if is_bad else 0) - (1 if was_bad else 0)
                            
                            counts[c_old] -= 1
                            counts[c_new] += 1
                        
                        colors[e] = c_new
                        bad_count += delta
                    
                    if bad_count < best_bad_in_restart:
                        best_bad_in_restart = bad_count
                        best_colors_in_restart = colors[:]
                        
                    # If we are stuck and bad_count is high, restart might be better
                    # But let's try a few perturbations first
                    if bad_count > 50 and random.random() < 0.1:
                        break # Restart

        return None

    targets = [17, 16, 15, 14, 13, 12, 11]
    budgets = {
        17: 60.0,
        16: 30.0,
        15: 10.0,
        14: 5.0,
        13: 2.0,
        12: 1.0,
        11: 0.5
    }
    
    for n in targets:
        if time.time() > deadline - 10:
            break
        
        budget = budgets.get(n, 5.0)
        if time.time() + budget > deadline - 5:
             budget = deadline - 5 - time.time()
             if budget < 1.0:
                 continue
                 
        result = search_n_fast(n, budget)
        
        if result:
            return result
        
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
