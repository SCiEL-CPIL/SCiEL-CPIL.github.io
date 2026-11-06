"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin

    for target_n in [34, 33, 32, 30, 25, 20, 15, 10, 9]:
        if time.time() > deadline:
            break

        n = target_n
        num_triples = n * (n - 1) * (n - 2) // 6

        # Build index map
        idx_map = {}
        idx = 0
        for c in combinations(range(n), 3):
            idx_map[c] = idx
            idx += 1

        # Precompute quads and quints as numpy arrays
        quads_list = []
        for a, b, c, d in combinations(range(n), 4):
            quads_list.append([idx_map[(a, b, c)], idx_map[(a, b, d)],
                               idx_map[(a, c, d)], idx_map[(b, c, d)]])
        quads = np.array(quads_list, dtype=np.int32)

        quints_list = []
        for quint in combinations(range(n), 5):
            quints_list.append([idx_map[t] for t in combinations(quint, 3)])
        quints = np.array(quints_list, dtype=np.int32)

        # Initialize with random coloring
        colors = np.array([random.randint(0, 1) for _ in range(num_triples)], dtype=np.uint8)

        def count_violations(col):
            # Red K4: all 4 triples are 0
            red_k4_count = np.sum(np.all(col[quads] == 0, axis=1))
            # Blue K5: all 10 triples are 1
            blue_k5_count = np.sum(np.all(col[quints] == 1, axis=1))
            return int(red_k4_count + blue_k5_count)

        total_violations = count_violations(colors)
        if total_violations == 0:
            return {'n': n, 'colors': colors.tolist()}

        # Simulated annealing
        temperature = 3.0
        cooling = 0.9998
        max_steps = 20000
        best_colors = colors.copy()
        best_violations = total_violations

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if total_violations == 0:
                return {'n': n, 'colors': colors.tolist()}

            idx = random.randint(0, num_triples - 1)
            old = colors[idx]
            new = 1 - old
            colors[idx] = new

            new_violations = count_violations(colors)
            delta = new_violations - total_violations

            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 0.01)):
                total_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors.copy()
            else:
                colors[idx] = old

            temperature *= cooling

        if best_violations == 0:
            return {'n': n, 'colors': best_colors.tolist()}

    # Fallback: small valid coloring
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
