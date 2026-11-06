import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (evaluator kills at 150s)

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_structure(n):
        """
        Build all C6 cycles and mapping from edges to cycles.
        Returns:
        - cycles: list of tuples of 6 edge indices
        - edge_to_cycles: dict mapping edge index -> list of cycle indices
        - edge_to_other: dict mapping edge index -> list of (other 5 edges) tuples
        """
        m = n * (n - 1) // 2
        verts = list(range(n))
        
        # Generate all C6 cycles
        cycles = []
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
                cycles.append(tuple(cycle_edges))
        
        N = len(cycles)
        
        # Map edge to list of cycle indices
        edge_to_cycles = [[] for _ in range(m)]
        for i in range(N):
            for e in cycles[i]:
                edge_to_cycles[e].append(i)
        
        # Precompute 'other edges' for each edge
        edge_to_other = [None] * m
        for e in range(m):
            cyc_ids = edge_to_cycles[e]
            if not cyc_ids:
                edge_to_other[e] = []
                continue
            other_edges_list = []
            for cid in cyc_ids:
                ce = cycles[cid]
                # Remove edge e from the cycle
                other = tuple(x for x in ce if x != e)
                other_edges_list.append(other)
            edge_to_other[e] = other_edges_list
            
        return cycles, edge_to_cycles, edge_to_other

    def count_bad_cycles(colors, cycles):
        """Count monochromatic C6 cycles."""
        if not cycles:
            return 0
        bad = 0
        for cycle in cycles:
            c0 = colors[cycle[0]]
            if all(colors[e] == c0 for e in cycle):
                bad += 1
        return bad

    def count_bad_cycles_fast(colors, cycles):
        """Faster count using numpy if possible."""
        if not cycles:
            return 0
        arr = np.array(cycles, dtype=np.int32)
        cycle_colors = colors[arr]
        first = cycle_colors[:, 0]
        is_mono = np.all(cycle_colors == first[:, np.newaxis], axis=1)
        return int(np.sum(is_mono))

    def search_n(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)
        np.random.seed(42 + seed_offset)

        cycles, edge_to_cycles, edge_to_other = build_structure(n)
        N = len(cycles)

        if N == 0:
            colors = np.zeros(m, dtype=np.uint8)
            return {'n': n, 'colors': colors.tolist()}

        # Initial coloring: random
        colors = np.random.randint(0, 4, size=m, dtype=np.uint8)
        bad_count = count_bad_cycles_fast(colors, cycles)
        
        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        best_colors = colors.copy()
        best_bad = bad_count

        # Simulated Annealing with Walk-SAT bias
        T_start = 2.0
        T_end = 0.01
        start_iter_time = time.time()
        
        # Track bad cycles for biasing
        # We'll recompute bad cycles periodically or just use random edges
        
        while time.time() < start_iter_time + time_budget:
            if time.time() > deadline - 5:
                break

            # Pick an edge to change
            # Bias towards edges in bad cycles if we can identify them quickly
            # For simplicity and speed, pick random edge
            e = np.random.randint(0, m)
            c_old = colors[e]
            
            # Pick new color different from old
            c_new = np.random.randint(0, 4)
            while c_new == c_old:
                c_new = np.random.randint(0, 4)

            # Calculate Delta: change in number of bad cycles
            # A cycle becomes bad if all 6 edges have the same color.
            # Changing edge e from c_old to c_new affects only cycles containing e.
            other_edges_list = edge_to_other[e]
            
            delta = 0
            for other in other_edges_list:
                # Check if the cycle was bad before (all 6 edges c_old)
                was_bad = all(colors[x] == c_old for x in other)
                # Check if the cycle will be bad after (all 5 other edges c_new)
                will_be_bad = all(colors[x] == c_new for x in other)
                
                if will_be_bad and not was_bad:
                    delta += 1
                elif was_bad and not will_be_bad:
                    delta -= 1

            # Acceptance criterion
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

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Try targets from high to low
    # We know n=17 is possible. Try to find it quickly.
    # If we can't find 17, fall back to lower.
    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    
    # Allocate time based on likelihood of success
    # n=17 is the target. Give it the most time.
    time_alloc = {17: 40, 16: 20, 15: 15, 14: 10, 13: 8, 12: 5, 11: 4, 10: 3, 9: 2, 8: 2, 7: 1, 6: 1}

    for n in targets:
        if time.time() > deadline - 15:
            break

        tb = time_alloc.get(n, 5)
        
        # Multiple restarts
        num_restarts = 5 if n == 17 else 3
        restart_budget = tb / num_restarts

        for r in range(num_restarts):
            if time.time() > deadline - 15:
                break
            result = search_n(n, restart_budget, seed_offset=r * 1000 + n)
            if result:
                best_result = result
                # If we found a valid coloring for a high n, we can stop
                # unless we want to try for even higher (but max is 20, and 17 is known)
                # Since we go from high to low, the first valid is the best.
                return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
