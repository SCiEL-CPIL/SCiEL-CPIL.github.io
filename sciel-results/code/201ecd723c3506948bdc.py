import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit

    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}

    # Precompute K4s and their edges
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]

    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)
        for e in e_idx:
            edge_to_k4s[e].append(len(k4s) - 1)

    num_k4s = len(k4s)

    # Precompute for each edge, the indices of K4s it belongs to
    # And for each K4, the position of each edge in its edge list
    # To speed up delta calculation
    
    # For fast delta calculation, we need to know, for a given edge e, 
    # which K4s it is in, and what the other 5 edges are.
    
    edge_k4_info = []
    for e in range(num_edges):
        info = []
        for k4_idx in edge_to_k4s[e]:
            e_list = k4_edges[k4_idx]
            # Find the other 5 edges
            others = [x for x in e_list if x != e]
            info.append((k4_idx, others))
        edge_k4_info.append(info)

    def count_bad_k4s(colors):
        """Count K4s with fewer than 5 distinct colors."""
        bad = 0
        for e_list in k4_edges:
            c0 = colors[e_list[0]]
            c1 = colors[e_list[1]]
            c2 = colors[e_list[2]]
            c3 = colors[e_list[3]]
            c4 = colors[e_list[4]]
            c5 = colors[e_list[5]]
            s = {c0, c1, c2, c3, c4, c5}
            if len(s) < 5:
                bad += 1
        return bad

    def delta_bad(e_idx, old_color, new_color, colors):
        """Compute change in bad K4 count when edge e_idx changes from old_color to new_color."""
        delta = 0
        for k4_idx, others in edge_k4_info[e_idx]:
            # Get colors of the other 5 edges
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            c4 = colors[others[3]]
            c5 = colors[others[4]]
            
            # Before change
            s_before = {old_color, c1, c2, c3, c4, c5}
            bad_before = 1 if len(s_before) < 5 else 0
            
            # After change
            s_after = {new_color, c1, c2, c3, c4, c5}
            bad_after = 1 if len(s_after) < 5 else 0
            
            delta += bad_after - bad_before
            
        return delta

    def verify_valid(colors_list):
        """Verify a coloring is valid (all K4s have >= 5 distinct colors)."""
        for e_list in k4_edges:
            cols = set([colors_list[e] for e in e_list])
            if len(cols) < 5:
                return False
        return True

    # Baseline: (i+j) % 17
    def get_baseline():
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    baseline = get_baseline()
    # Verify baseline just in case
    if not verify_valid(baseline):
        # Fallback to random if baseline fails (shouldn't happen)
        return [random.randint(0, 119) for _ in range(num_edges)]

    best_colors_used = 17
    best_result = baseline.copy()

    # Try to find better colorings with fewer colors
    # We know 13 is possible. Let's try 13, 14, 15.
    # To save time, we'll do fewer restarts but faster iterations.
    
    for target_C in [13, 14, 15]:
        if time.time() > deadline - 5:
            break

        # Fewer restarts to ensure we finish in time
        restarts = 5 if target_C == 13 else 10
        max_iterations = 5000

        for r in range(restarts):
            if time.time() > deadline - 2:
                break

            # Initialize with random colors
            colors = [random.randint(0, target_C - 1) for _ in range(num_edges)]
            current_bad = count_bad_k4s(colors)
            best_bad_in_run = current_bad
            best_colors_in_run = colors.copy()

            if current_bad == 0:
                if target_C < best_colors_used:
                    best_colors_used = target_C
                    best_result = colors.copy()
                break

            # Simulated Annealing
            T = 1.0
            cooling = 0.999

            for it in range(max_iterations):
                if time.time() > deadline - 2:
                    break

                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, target_C - 1)
                if new_color == old_color:
                    continue

                delta = delta_bad(e_idx, old_color, new_color, colors)

                if delta < 0:
                    colors[e_idx] = new_color
                    current_bad += delta
                elif delta == 0:
                    if random.random() < 0.5:
                        colors[e_idx] = new_color
                else:
                    if random.random() < np.exp(-delta / max(T, 0.01)):
                        colors[e_idx] = new_color
                        current_bad += delta

                if current_bad < best_bad_in_run:
                    best_bad_in_run = current_bad
                    best_colors_in_run = colors.copy()

                if current_bad == 0:
                    break

                T *= cooling

            if best_bad_in_run == 0:
                if target_C < best_colors_used:
                    best_colors_used = target_C
                    best_result = best_colors_in_run.copy()
                break

        if best_result is not None and best_colors_used <= target_C:
            break

    # Final verification
    if verify_valid(best_result):
        return best_result
    else:
        return baseline
