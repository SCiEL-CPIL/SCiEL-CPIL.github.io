import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

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
        edge_to_other_edges = [None] * m
        for e in range(m):
            cyc_ids = edge_to_cycles[e]
            if not cyc_ids:
                edge_to_other_edges[e] = np.array([]).reshape(0, 5)
                continue
            other_edges_list = []
            for cid in cyc_ids:
                ce = cycles[cid]
                mask = ce != e
                other_edges_list.append(ce[mask])
            
            edge_to_other_edges[e] = np.array(other_edges_list, dtype=np.int32)
            
        return cycles, edge_to_other_edges

    def calculate_initial_bad(colors, cycles):
        if len(cycles) == 0:
            return 0
        cycle_colors = colors[cycles]
        first = cycle_colors[:, 0]
        is_mono = np.all(cycle_colors == first[:, np.newaxis], axis=1)
        return int(np.sum(is_mono))

    def spectral_init(n, m, seed_offset=0):
        """
        Initialize edge colors using spectral methods on the graph Laplacian.
        This breaks long-range correlations that create C6 structures.
        """
        random.seed(42 + seed_offset)
        np.random.seed(42 + seed_offset)
        
        # Build adjacency matrix of K_n
        A = np.ones((n, n), dtype=np.float64) - np.eye(n)
        # Laplacian L = D - A
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Compute eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            # Fallback to random
            return np.random.randint(0, 4, size=m, dtype=np.uint8)
        
        # Use the first few non-trivial eigenvectors (skip the first which is constant)
        # These capture the spectral structure of the complete graph
        # For K_n, eigenvalues are 0, n, n, ..., n (multiplicity n-1)
        # Use eigenvectors 1, 2, 3 (indices 1, 2, 3) for 4 colors
        
        # Get vertex positions from eigenvectors
        # Use eigenvectors 1 and 2 to create 2D coordinates
        if n >= 3:
            # Use the first two non-trivial eigenvectors
            # These give us a 2D embedding
            vec1 = eigenvectors[:, 1]  # First non-trivial
            vec2 = eigenvectors[:, 2]  # Second non-trivial
            
            # Normalize to [-1, 1]
            v1_min, v1_max = vec1.min(), vec1.max()
            v2_min, v2_max = vec2.min(), vec2.max()
            
            if v1_max > v1_min:
                vec1_norm = (vec1 - v1_min) / (v1_max - v1_min) * 2 - 1
            else:
                vec1_norm = np.zeros(n)
                
            if v2_max > v2_min:
                vec2_norm = (vec2 - v2_min) / (v2_max - v2_min) * 2 - 1
            else:
                vec2_norm = np.zeros(n)
            
            # Assign edge colors based on quadrant of the difference vector
            colors = np.zeros(m, dtype=np.uint8)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    # Difference in spectral coordinates
                    dx = vec1_norm[j] - vec1_norm[i]
                    dy = vec2_norm[j] - vec2_norm[i]
                    
                    # Quadrant assignment with some randomness
                    # This naturally breaks correlations
                    if dx >= 0 and dy >= 0:
                        base_color = 0
                    elif dx < 0 and dy >= 0:
                        base_color = 1
                    elif dx < 0 and dy < 0:
                        base_color = 2
                    else:
                        base_color = 3
                    
                    # Add some random perturbation to avoid perfect symmetry
                    if random.random() < 0.15:
                        base_color = random.randint(0, 3)
                    
                    colors[idx] = base_color
                    idx += 1
        else:
            colors = np.random.randint(0, 4, size=m, dtype=np.uint8)
        
        return colors

    def find_bad_cycles(colors, cycles):
        """Find indices of all monochromatic cycles."""
        if len(cycles) == 0:
            return np.array([], dtype=np.int32)
        cycle_colors = colors[cycles]
        first = cycle_colors[:, 0]
        is_mono = np.all(cycle_colors == first[:, np.newaxis], axis=1)
        return np.where(is_mono)[0]

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

        # Use spectral initialization
        colors = spectral_init(n, m, seed_offset)
        bad_count = calculate_initial_bad(colors, cycles)
        
        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        best_colors = colors.copy()
        best_bad = bad_count

        T_start = 1.0
        T_end = 0.05
        start_iter_time = time.time()
        
        # Track bad cycles for WalkSAT-style biasing
        bad_cycle_ids = find_bad_cycles(colors, cycles)
        
        # Build edge frequency in bad cycles for biasing
        edge_bad_freq = np.zeros(m, dtype=np.int32)
        if len(bad_cycle_ids) > 0:
            for cid in bad_cycle_ids:
                for e in cycles[cid]:
                    edge_bad_freq[e] += 1
        
        while time.time() < start_iter_time + time_budget:
            if time.time() > deadline - 5:
                break

            # Choose edge: biased towards edges in bad cycles (WalkSAT-style)
            if len(bad_cycle_ids) > 0 and random.random() < 0.6:
                # Pick from edges in bad cycles with probability proportional to frequency
                # Use a simple approach: pick a random bad cycle, then a random edge in it
                bad_cid = bad_cycle_ids[random.randint(0, len(bad_cycle_ids) - 1)]
                e = int(cycles[bad_cid, random.randint(0, 5)])
            else:
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
                delta = int(np.sum(is_all_new) - np.sum(is_all_old))

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
                
                # Update bad cycle tracking periodically
                if random.random() < 0.01:
                    bad_cycle_ids = find_bad_cycles(colors, cycles)

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Try targets from high to low
    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    
    # Allocate time: prioritize higher n with more restarts for spectral init
    time_alloc = {17: 35, 16: 25, 15: 20, 14: 15, 13: 12, 12: 10, 11: 8, 10: 6, 9: 4, 8: 3, 7: 2, 6: 1}

    for n in targets:
        if time.time() > deadline - 10:
            break

        tb = time_alloc.get(n, 5)
        # More restarts for spectral init to explore different spectral structures
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
