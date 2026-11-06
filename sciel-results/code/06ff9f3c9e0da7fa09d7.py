import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
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
        # edge_to_other_edges[e] will be a numpy array of shape (len(edge_to_cycles[e]), 5)
        edge_to_other_edges = [None] * m
        for e in range(m):
            cyc_ids = edge_to_cycles[e]
            if not cyc_ids:
                edge_to_other_edges[e] = np.array([]).reshape(0, 5)
                continue
            # Get all edges in these cycles
            all_edges = cycles[cyc_ids] # shape (K, 6)
            # We need to remove column where value is e
            # Find the position of e in each row
            # This is a bit tricky with numpy if positions vary.
            # But we can just take the 6 edges and filter?
            # Simpler: for each cycle index, get the 6 edges, remove e, append to list.
            other_edges_list = []
            for cid in cyc_ids:
                ce = cycles[cid]
                # ce is a numpy array of 6 ints
                # Create a mask of where ce != e
                mask = ce != e
                # There should be exactly 5 True values
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
        # Get colors for all edges in all cycles
        # cycle_colors shape (N, 6)
        cycle_colors = colors[cycles]
        # Check if all 6 are the same
        # First edge color
        first = cycle_colors[:, 0]
        # Check if all others match first
        # np.all(cycle_colors == first[:, np.newaxis], axis=1)
        is_mono = np.all(cycle_colors == first[:, np.newaxis], axis=1)
        return np.sum(is_mono)

    def get_bad_cycles(colors, cycles):
        """Returns indices of monochromatic cycles"""
        if len(cycles) == 0:
            return np.array([], dtype=np.int32)
        cycle_colors = colors[cycles]
        first = cycle_colors[:, 0]
        is_mono = np.all(cycle_colors == first[:, np.newaxis], axis=1)
        return np.where(is_mono)[0]

    def spectral_init(n, seed_val=42):
        """
        Use spectral embedding to create a structured initial coloring.
        Place vertices on a circle based on their index, then color edges
        based on angular distance to encourage different colors for nearby edges.
        """
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        m = n * (n - 1) // 2
        colors = np.zeros(m, dtype=np.uint8)
        
        # Try multiple spectral-like strategies
        # Strategy 1: Circular layout with color based on edge "length"
        # Place vertices on circle, color by chord length modulo 4
        for i in range(n):
            for j in range(i + 1, n):
                # Angular distance
                dist = abs(i - j)
                # Use a hash-like function to spread colors
                # Mix the distance with vertex positions
                val = (i * 7 + j * 13 + dist * 3) % 4
                colors[edge_index(i, j, n)] = val
        
        return colors

    def repair_phase(colors, cycles, edge_to_other_edges, time_budget, n, m):
        """
        Aggressive repair: iteratively pick a bad cycle and try to fix it
        by changing one of its edges to a color that breaks the monochromatic cycle.
        """
        bad_count = calculate_initial_bad(colors, cycles)
        if bad_count == 0:
            return colors, 0
        
        start_time = time.time()
        best_colors = colors.copy()
        best_bad = bad_count
        
        while time.time() < start_time + time_budget:
            if time.time() > deadline - 5:
                break
            
            bad_cycles = get_bad_cycles(colors, cycles)
            if len(bad_cycles) == 0:
                break
            
            # Pick a random bad cycle
            bc_idx = bad_cycles[np.random.randint(0, len(bad_cycles))]
            cycle_edges = cycles[bc_idx]
            
            # Try each edge in the cycle
            for e in cycle_edges:
                c_old = colors[e]
                other_edges = edge_to_other_edges[e]
                
                # Try each alternative color
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    
                    # Calculate delta for this specific change
                    if len(other_edges) == 0:
                        delta = 0
                    else:
                        other_colors = colors[other_edges]
                        is_all_old = np.all(other_colors == c_old, axis=1)
                        is_all_new = np.all(other_colors == c_new, axis=1)
                        delta = np.sum(is_all_new) - np.sum(is_all_old)
                    
                    # Only accept if it reduces bad count or is neutral
                    if delta <= 0:
                        colors[e] = c_new
                        bad_count += delta
                        if bad_count < best_bad:
                            best_bad = bad_count
                            best_colors = colors.copy()
                        if bad_count == 0:
                            return colors, 0
                        break  # Found a fix for this edge, move on
            
            # If no edge in the cycle could be fixed, try a random edge change
            e = np.random.randint(0, m)
            c_old = colors[e]
            c_new = np.random.randint(0, 4)
            while c_new == c_old:
                c_new = np.random.randint(0, 4)
            
            other_edges = edge_to_other_edges[e]
            if len(other_edges) == 0:
                delta = 0
            else:
                other_colors = colors[other_edges]
                is_all_old = np.all(other_colors == c_old, axis=1)
                is_all_new = np.all(other_colors == c_new, axis=1)
                delta = np.sum(is_all_new) - np.sum(is_all_old)
            
            # Accept with some probability even if worse (to escape local optima)
            if delta <= 0 or (delta < 3 and random.random() < 0.3):
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors.copy()
        
        return best_colors, best_bad

    def search_n(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)
        np.random.seed(42 + seed_offset)

        cycles, edge_to_other_edges = build_cycles_and_maps(n)
        N = len(cycles)

        if N == 0:
            colors = np.zeros(m, dtype=np.uint8)
            return {'n': n, 'colors': colors.tolist()}

        # Try spectral initialization first
        colors = spectral_init(n, seed_val=42 + seed_offset)
        bad_count = calculate_initial_bad(colors, cycles)
        
        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        best_colors = colors.copy()
        best_bad = bad_count

        # Phase 1: Repair phase (targeted fixes)
        repair_time = min(time_budget * 0.4, 20)
        colors, bad_count = repair_phase(colors, cycles, edge_to_other_edges, repair_time, n, m)
        if bad_count < best_bad:
            best_bad = bad_count
            best_colors = colors.copy()
        
        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}

        # Phase 2: Simulated annealing
        T_start = 1.0
        T_end = 0.05
        start_iter_time = time.time()
        sa_budget = time_budget - repair_time - 2
        
        while time.time() < start_iter_time + sa_budget:
            if time.time() > deadline - 5:
                break

            # Pick random edge
            e = np.random.randint(0, m)
            c_old = colors[e]
            
            # Pick new color
            c_new = np.random.randint(0, 4)
            while c_new == c_old:
                c_new = np.random.randint(0, 4)

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
                progress = min(1.0, elapsed / max(sa_budget, 1))
                T = T_start * (T_end / T_start) ** progress
                
                if delta / T > 50:
                    accept = False
                else:
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob

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
    
    # Allocate time - give more time to higher targets
    time_alloc = {17: 45, 16: 30, 15: 20, 14: 15, 13: 10, 12: 8, 11: 5, 10: 4, 9: 3, 8: 2, 7: 2, 6: 1}

    for n in targets:
        if time.time() > deadline - 10:
            break

        tb = time_alloc.get(n, 5)
        # Multiple restarts with different seeds
        num_restarts = 4 if n >= 15 else 3
        restart_budget = tb / num_restarts

        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            result = search_n(n, restart_budget, seed_offset=r * 1000 + n)
            if result:
                best_result = result
                return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
