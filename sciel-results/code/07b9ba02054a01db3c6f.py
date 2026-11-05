import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

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
    k4_edges_np = np.array(k4_edges, dtype=np.int32)

    # Precompute for each edge, the indices of K4s it belongs to
    edge_k4_indices = [np.array(k4s_list, dtype=np.int32) for k4s_list in edge_to_k4s]

    def count_distinct_per_k4(colors_np):
        """Return array of distinct color counts for each K4."""
        k4_colors = colors_np[k4_edges_np]  # (num_k4s, 6)
        sorted_colors = np.sort(k4_colors, axis=1)
        diff = np.diff(sorted_colors, axis=1)
        unique_counts = 1 + np.sum(diff != 0, axis=1)
        return unique_counts

    def count_bad_k4s_np(colors_np):
        """Count K4s with fewer than 5 distinct colors."""
        unique_counts = count_distinct_per_k4(colors_np)
        return int(np.sum(unique_counts < 5))

    def delta_bad(e_idx, old_color, new_color, colors_np):
        """Compute change in bad K4 count when edge e_idx changes from old_color to new_color."""
        k4_indices = edge_k4_indices[e_idx]
        if len(k4_indices) == 0:
            return 0

        # Get the 6 edge indices for each affected K4
        affected_k4_edges = k4_edges_np[k4_indices]  # (num_affected, 6)

        # Get current colors for all affected K4 edges
        current_colors = colors_np[affected_k4_edges]  # (num_affected, 6)

        # Compute distinct counts BEFORE the change
        sorted_before = np.sort(current_colors, axis=1)
        diff_before = np.diff(sorted_before, axis=1)
        unique_before = 1 + np.sum(diff_before != 0, axis=1)

        # Create modified colors: replace old_color with new_color where it appears
        # For each affected K4, the edge e_idx is at some position
        # We need to find where e_idx is in each K4's edge list
        # Actually, let's just modify the specific positions
        modified_colors = current_colors.copy()
        for k in range(len(k4_indices)):
            k4_e_list = k4_edges_np[k4_indices[k]]
            pos = np.where(k4_e_list == e_idx)[0]
            if len(pos) > 0:
                modified_colors[k, pos[0]] = new_color

        # Compute distinct counts AFTER the change
        sorted_after = np.sort(modified_colors, axis=1)
        diff_after = np.diff(sorted_after, axis=1)
        unique_after = 1 + np.sum(diff_after != 0, axis=1)

        bad_before = np.sum(unique_before < 5)
        bad_after = np.sum(unique_after < 5)

        return int(bad_after - bad_before)

    # Baseline fallback: (i+j) % 17
    def get_baseline():
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    def verify_valid(colors_list):
        """Verify a coloring is valid (all K4s have >= 5 distinct colors)."""
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            cols = set([
                colors_list[edge_index[(a, b)]],
                colors_list[edge_index[(a, c)]],
                colors_list[edge_index[(a, d)]],
                colors_list[edge_index[(b, c)]],
                colors_list[edge_index[(b, d)]],
                colors_list[edge_index[(c, d)]]
            ])
            if len(cols) < 5:
                return False
        return True

    # First, verify the baseline is valid
    baseline = get_baseline()
    if not verify_valid(baseline):
        # This shouldn't happen, but handle gracefully
        return [random.randint(0, 119) for _ in range(num_edges)]

    best_result = None
    best_colors_used = 17  # Start with baseline

    # Try to find better colorings with fewer colors
    for target_C in [13, 14, 15]:
        if time.time() > deadline - 10:
            break

        restarts = 30 if target_C == 13 else 50
        max_iterations = 20000

        for r in range(restarts):
            if time.time() > deadline - 3:
                break

            # Initialize with random colors
            colors = np.array([random.randint(0, target_C - 1) for _ in range(num_edges)], dtype=np.int32)
            current_bad = count_bad_k4s_np(colors)
            best_bad_in_run = current_bad
            best_colors_in_run = colors.copy()

            if current_bad == 0:
                if target_C < best_colors_used:
                    best_colors_used = target_C
                    best_result = colors.copy()
                break

            # Simulated annealing
            T = 1.0
            cooling = 0.9995

            for it in range(max_iterations):
                if time.time() > deadline - 3:
                    break

                e_idx = random.randint(0, num_edges - 1)
                old_color = int(colors[e_idx])
                new_color = random.randint(0, target_C - 1)
                if new_color == old_color:
                    continue

                colors[e_idx] = new_color
                delta = delta_bad(e_idx, old_color, new_color, colors_np=colors)
                colors[e_idx] = old_color  # Revert for now

                if delta < 0:
                    # Accept improvement
                    colors[e_idx] = new_color
                    current_bad += delta
                elif delta == 0:
                    # Accept sideways with some probability
                    if random.random() < 0.3:
                        colors[e_idx] = new_color
                else:
                    # Accept worsening with probability exp(-delta/T)
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
            # Found a valid coloring with target_C or fewer colors
            break

    if best_result is not None:
        # Verify the result
        result_list = [int(c) for c in best_result]
        if verify_valid(result_list):
            return result_list
        else:
            return baseline
    else:
        return baseline
