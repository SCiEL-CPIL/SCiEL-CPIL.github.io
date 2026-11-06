"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_cycles_and_maps(n):
        """
        Returns:
        cycles: np.array of shape (N, 6) containing edge indices
        edge_to_other_edges: list of np.arrays, where edge_to_other_edges[e] 
                             is an array of shape (K, 5) containing the other 5 edges 
                             for each of the K cycles containing edge e.
        """
        m = n * (n - 1) // 2
        cycles_list = []
        verts = list(range(n))
        
        # Generate all C6 cycles
        # A cycle is defined by 6 distinct vertices. 
        # To avoid duplicates (reversals and rotations), we fix the smallest vertex as start,
        # and ensure the second vertex is smaller than the last (or similar canonical form).
        for combo in combinations(verts, 6):
            v0 = combo[0]
            others = list(combo[1:])
            for perm in permutations(others):
                # Canonical form: perm[0] < perm[-1] to avoid double counting reversal
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(edge_index(u, v, n))
                cycles_list.append(cycle_edges)
        
        if not cycles_list:
            return np.array([]).reshape(0, 6), [np.array([]).reshape(0, 5) for _ in range(m)]

        cycles = np.array(cycles_list, dtype=np.int32)
        N = len(cycles)
        
        # Map edge to list of cycle indices
        edge_to_cycles = [[] for _ in range(m)]
        for i in range(N):
            for e in cycles[i]:
                edge_to_cycles[e].append(i)
        
        # Precompute 'other edges' for each edge
        edge_to_other_edges = [None] * m
        for e in range(m):
            cyc_ids = edge_to_cycles[e]
            if not cyc_ids:
                edge_to_other_edges[e] = np.array([]).reshape(0, 5)
                continue
            
            other_edges_list = []
            for cid in cyc_ids:
                ce = cycles[cid]
                # ce is a numpy array of 6 ints
                mask = ce != e
                other_edges_list.append(ce[mask])
            
            edge_to_other_edges[e] = np.array(other_edges_list, dtype=np.int32)
            
        return cycles, edge_to_other_edges

    def calculate_initial_bad(colors, cycles):
        """
        colors: np.array of shape (M,)
        cycles: np.array of shape (N, 6)
        Returns: number of monochromatic cycles
        """
        if len(cycles) == 0:
            return 0
        cycle_colors = colors[cycles]
        first = cycle_colors[:, 0]
        is_mono = np.all(cycle_colors == first[:, np.newaxis], axis=1)
        return np.sum(is_mono)

    def search_n(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        rng = np.random.default_rng(42 + seed_offset)

        cycles, edge_to_other_edges = build_cycles_and_maps(n)
        N = len(cycles)

        if N == 0:
            colors = np.zeros(m, dtype=np.uint8)
            return {'n': n, 'colors': colors.tolist()}

        # Initial coloring
        colors = rng.integers(0, 4, size=m, dtype=np.uint8)
        bad_count = calculate_initial_bad(colors, cycles)
        
        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        best_colors = colors.copy()
        best_bad = bad_count

        T_start = 1.0
        T_end = 0.05
        start_iter_time = time.time()
        
        # Walk-SAT / SA Hybrid
        while time.time() < start_iter_time + time_budget:
            if time.time() > deadline - 5:
                break

            # Pick random edge
            e = rng.integers(0, m)
            c_old = colors[e]
            
            # Pick new color
            c_new = rng.integers(0, 4)
            while c_new == c_old:
                c_new = rng.integers(0, 4)

            # Calculate Delta
            other_edges = edge_to_other_edges[e]
            
            if len(other_edges) == 0:
                delta = 0
            else:
                other_colors = colors[other_edges]
                is_all_old = np.all(other_colors == c_old, axis=1)
                is_all_new = np.all(other_colors == c_new, axis=1)
                delta = np.sum(is_all_new) - np.sum(is_all_old)

            # Acceptance
            accept = False
            if delta <= 0:
                accept = True
            else:
                elapsed = time.time() - start_iter_time
                progress = min(1.0, elapsed / time_budget)
                T = T_start * (T_end / T_start) ** progress
                
                if delta / T > 50:
                    accept = False
                else:
                    prob = np.exp(-delta / T)
                    accept = rng.random() < prob

            if accept:
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors.copy()
                    if best_bad == 0:
                        return {'n': n, 'colors': colors.tolist()}

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Try targets from high to low
    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    
    # Allocate time based on difficulty
    # 17 is hard, give it more time.
    time_alloc = {17: 40, 16: 30, 15: 20, 14: 15, 13: 10, 12: 8, 11: 5, 10: 4, 9: 3, 8: 2, 7: 2, 6: 1}

    found_best = 5
    
    for n in targets:
        if time.time() > deadline - 10:
            break

        tb = time_alloc.get(n, 5)
        
        # If we already found a better n, we might skip lower ones unless we want to be sure?
        # Actually, we want the MAXIMUM n. So if we find 17, we stop.
        # If we fail 17, we try 16.
        
        result = search_n(n, tb, seed_offset=n)
        
        if result:
            best_result = result
            # If we found a valid coloring for n, and n is the highest possible (17), we can stop.
            # But we might find 18? No, max n is 20, but known best is 17.
            # The task says "Maximize n".
            # If we find 17, we are happy.
            # Let's try to see if we can go higher? 
            # The loop goes 17 -> 16 -> ...
            # So if we find 17, we break.
            # If we don't find 17, we continue to 16.
            if n == 17:
                break
            # If we find 16, we break? No, we want to see if 17 was a fluke? 
            # No, if 17 failed, we assume it's hard. 
            # But maybe 17 is possible? 
            # The strategy is: try 17 first. If fail, try 16.
            # If we find 16, we accept it, but we don't stop immediately?
            # Actually, if we find 16, it's a valid result. 
            # But we want the MAX. 
            # So we should only stop if we find the highest target we are trying?
            # Or if we find any, we can try to improve?
            # The current loop structure:
            # for n in targets:
            #   result = search_n(n)
            #   if result: best_result = result; break
            # This means if we find 17, we stop. If we fail 17, we try 16. If we find 16, we stop.
            # This is correct for maximizing n, assuming search_n is reliable enough.
            break

    # If we didn't find anything above 5, return 5.
    # Wait, the fallback is 5. 
    # But we should try to at least get a valid small one if 6+ fails?
    # The loop handles 6..17.
    # If all fail, best_result remains 5.
    
    return best_result
# EVOLVE-BLOCK-END
