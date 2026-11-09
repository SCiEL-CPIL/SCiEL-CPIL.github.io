import time
import random
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute edges
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            edges.append((i, j))

    # Map edge (u, v) to index
    edge_idx_map = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx_map[(i, j)] = idx
        edge_idx_map[(j, i)] = idx

    # Generate all K4s and precompute edge indices
    k4s_all = []
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        e_indices = [
            edge_idx_map[(a, b)],
            edge_idx_map[(a, c)],
            edge_idx_map[(a, d)],
            edge_idx_map[(b, c)],
            edge_idx_map[(b, d)],
            edge_idx_map[(c, d)]
        ]
        k4s_all.append(e_indices)

    # For each edge, list of K4 indices it belongs to
    edge_k4s = [[] for _ in range(120)]
    for k4_idx, e_indices in enumerate(k4s_all):
        for ei in e_indices:
            edge_k4s[ei].append(k4_idx)

    def count_distinct(k4_idx, colors):
        e_indices = k4s_all[k4_idx]
        c = set()
        for ei in e_indices:
            c.add(colors[ei])
        return len(c)

    def check_k4(k4_idx, colors):
        return count_distinct(k4_idx, colors) >= 5

    def check_all(colors):
        for k4_idx in range(len(k4s_all)):
            if not check_k4(k4_idx, colors):
                return False
        return True

    # --- Initialization: try several random starts and keep best ---
    best_colors = None
    best_num_colors = 18  # worst case

    # Start with (i+j) % 17 as a baseline
    base = [0] * 120
    for idx, (i, j) in enumerate(edges):
        base[idx] = (i + j) % 17
    if check_all(base):
        nc = len(set(base))
        if nc < best_num_colors:
            best_num_colors = nc
            best_colors = base[:]

    # Try a few random starts with small palette
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        c = [random.randint(0, 14) for _ in range(120)]
        if check_all(c):
            nc = len(set(c))
            if nc < best_num_colors:
                best_num_colors = nc
                best_colors = c[:]

    if best_colors is None:
        best_colors = base[:]
        best_num_colors = len(set(base))

    colors = best_colors[:]

    # Track used colors
    def get_used_colors(cols):
        return sorted(set(cols))

    # --- Simulated Annealing / Local Search ---
    # Objective: minimize number of distinct colors
    # Constraint: every K4 has >= 5 distinct colors

    current = colors[:]
    current_nc = best_num_colors

    # We'll do a greedy + random restart hybrid
    # Phase 1: Greedy - try to reduce color count
    # Phase 2: Random perturbations to escape local optima

    target_colors = list(range(15))  # 0..14

    while time.time() < deadline - 5.0:
        # Greedy phase: try to change each edge to a color that reduces distinct count
        improved = False
        for e_idx in random.sample(range(120), 120):
            if time.time() > deadline - 5.0:
                break
            old_color = current[e_idx]
            # Try each color in 0..14
            for new_color in target_colors:
                if new_color == old_color:
                    continue
                current[e_idx] = new_color
                # Check affected K4s
                valid = True
                for k4_idx in edge_k4s[e_idx]:
                    if not check_k4(k4_idx, current):
                        valid = False
                        break
                if valid:
                    # Accept and check if we improved
                    nc = len(set(current))
                    if nc < current_nc:
                        current_nc = nc
                        improved = True
                        if nc < best_num_colors:
                            best_num_colors = nc
                            best_colors = current[:]
                    # Keep the change even if nc didn't improve (might help later)
                    break
                else:
                    current[e_idx] = old_color

        if not improved:
            # Random perturbation: change a few random edges
            for _ in range(random.randint(3, 8)):
                e_idx = random.randint(0, 119)
                old_color = current[e_idx]
                new_color = random.choice(target_colors)
                current[e_idx] = new_color
                # Check validity
                valid = True
                for k4_idx in edge_k4s[e_idx]:
                    if not check_k4(k4_idx, current):
                        valid = False
                        break
                if not valid:
                    current[e_idx] = old_color

        # Periodically check if we can do better
        nc = len(set(current))
        if nc < best_num_colors:
            best_num_colors = nc
            best_colors = current[:]

    # Final compression
    unique_colors = sorted(set(best_colors))
    mapping = {c: i for i, c in enumerate(unique_colors)}
    final_colors = [mapping[c] for c in best_colors]
    return final_colors

# EVOLVE-BLOCK-END
