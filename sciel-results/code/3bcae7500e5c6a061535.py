import numpy as np
import itertools
import time
import random
import sys
import math

# --- Precomputation of C6 structures on 6 vertices ---
# A C6 is a cycle of length 6. There are (6-1)!/2 = 60 unique cycles on 6 labeled vertices.
# We represent each by a set of 6 edges (normalized pairs).

def get_k6_cycles():
    """
    Generate all unique sets of 6 edges that form a C6 on vertices 0..5.
    """
    cycles = set()
    # Fix vertex 0 to break rotational symmetry.
    # Iterate permutations of 1,2,3,4,5.
    # To break reflection symmetry, require perm[0] < perm[-1].
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        if perm[0] > perm[-1]:
            continue
        v = (0,) + perm
        edges = set()
        for i in range(6):
            u, w = v[i], v[(i + 1) % 6]
            if u > w:
                u, w = w, u
            edges.add((u, w))
        cycles.add(frozenset(edges))
    return [list(c) for c in cycles]

K6_CYCLES = get_k6_cycles()

def construct():
    start_time = time.time()
    # Evaluator kills after 150s. Keep a safety margin.
    deadline = start_time + 145.0

    def build_cycle_data(n):
        """
        Precompute:
        1. c6_edges_arr: shape (num_cycles, 6), dtype int32. Indices of edges in the global edge list.
        2. edge_to_cycles: list of lists. For each edge index, list of cycle indices.
        """
        num_edges = n * (n - 1) // 2

        # Map (i, j) with i < j to a linear index
        # Index = i*(n-1) - i*(i-1)//2 + (j-i-1)
        def get_idx(i, j):
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

        c6_edges_list = []
        # Iterate over all 6-vertex subsets
        for comb in itertools.combinations(range(n), 6):
            # For each C6 structure on 6 vertices
            for cycle_edges in K6_CYCLES:
                global_edges = []
                for u, v in cycle_edges:
                    gu = comb[u]
                    gv = comb[v]
                    if gu > gv:
                        gu, gv = gv, gu
                    global_edges.append(get_idx(gu, gv))
                c6_edges_list.append(global_edges)

        c6_edges_arr = np.array(c6_edges_list, dtype=np.int32)

        # Build edge -> cycles mapping
        edge_to_cycles = [[] for _ in range(num_edges)]
        for ci, edges in enumerate(c6_edges_list):
            for e in edges:
                edge_to_cycles[e].append(ci)

        return c6_edges_arr, edge_to_cycles, get_idx, num_edges

    def solve_n(n, time_limit, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        num_edges = n * (n - 1) // 2

        # Build data
        c6_edges_arr, edge_to_cycles, get_idx, num_edges = build_cycle_data(n)
        num_cycles = len(c6_edges_arr)

        if num_cycles == 0:
            return [0] * num_edges

        # Initialize colors randomly
        colors = np.random.randint(0, 4, num_edges, dtype=np.int8)

        # Initialize cycle color counts
        # cycle_color_count[ci, c] = number of edges in cycle ci with color c
        cycle_color_count = np.zeros((num_cycles, 4), dtype=np.int32)
        for ci in range(num_cycles):
            edges = c6_edges_arr[ci]
            for e in edges:
                cycle_color_count[ci, colors[e]] += 1

        def get_current_score():
            # A cycle is monochromatic if max count is 6
            return int(np.sum(np.max(cycle_color_count, axis=1) == 6))

        current_score = get_current_score()
        if current_score == 0:
            return colors.tolist()

        best_score = current_score
        best_colors = colors.copy()

        # Simulated Annealing with a more aggressive cooling schedule and restarts
        T = 10.0
        min_T = 0.05
        cooling = 0.999
        no_improve_count = 0
        max_no_improve = 5000 # Restart if stuck

        while time.time() < start_time + time_limit:
            # Select a random edge
            idx = np.random.randint(num_edges)
            old_color = colors[idx]
            new_color = np.random.randint(0, 3)
            if new_color >= old_color:
                new_color += 1

            delta = 0
            # Update affected cycles
            for ci in edge_to_cycles[idx]:
                old_mono = 1 if np.max(cycle_color_count[ci]) == 6 else 0

                cycle_color_count[ci, old_color] -= 1
                cycle_color_count[ci, new_color] += 1

                new_mono = 1 if np.max(cycle_color_count[ci]) == 6 else 0
                delta += new_mono - old_mono

            # Acceptance criterion
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[idx] = new_color
                current_score += delta

                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                    no_improve_count = 0
                    if best_score == 0:
                        return best_colors.tolist()
                else:
                    no_improve_count += 1
            else:
                # Revert changes
                for ci in edge_to_cycles[idx]:
                    cycle_color_count[ci, old_color] += 1
                    cycle_color_count[ci, new_color] -= 1

            T *= cooling
            if no_improve_count > max_no_improve or T < min_T:
                # Restart with higher temperature
                T = 10.0
                no_improve_count = 0
                # Optional: randomize current state slightly to escape local minima
                # But keeping current state is standard SA restart.

        if best_score == 0:
            return best_colors.tolist()
        return None

    # Strategy:
    # 1. Try n=17 (guaranteed success, baseline).
    # 2. Try n=18 aggressively (high probability of success if it exists).
    # 3. Try n=19 briefly (low probability, but high reward).

    # We prioritize n=17 first to ensure a valid result, then use remaining time for higher n.
    # However, the prompt says "Maximize n".
    # If we find 18, it's better than 17.
    # Let's try 18 first with a good budget, then 17 with remaining time.
    # If 18 fails, we fall back to 17.

    # Try n=18
    if time.time() < deadline - 20:
        t_start = time.time()
        time_limit = 60.0
        if deadline - t_start < time_limit:
            time_limit = deadline - t_start - 10.0

        if time_limit > 2.0:
            res = solve_n(18, time_limit, seed=2002)
            if res is not None:
                return {'n': 18, 'colors': res}

    # Try n=17
    # This should be very fast.
    if time.time() < deadline - 5:
        t_start = time.time()
        time_limit = 20.0
        if deadline - t_start < time_limit:
            time_limit = deadline - t_start - 2.0
        if time_limit > 1.0:
            res = solve_n(17, time_limit, seed=3003)
            if res is not None:
                return {'n': 17, 'colors': res}

    # Fallback: n=16
    if time.time() < deadline - 2:
        t_start = time.time()
        time_limit = 5.0
        if deadline - t_start < time_limit:
            time_limit = deadline - t_start - 1.0
        if time_limit > 0.5:
            res = solve_n(16, time_limit, seed=4004)
            if res is not None:
                return {'n': 16, 'colors': res}

    # Hard fallback: n=9 (very easy)
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i + j) % 4)
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    result = construct()
    print(f"Found n={result['n']}")
