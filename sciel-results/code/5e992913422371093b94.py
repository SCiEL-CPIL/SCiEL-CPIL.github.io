"""Colorings with three colors in every four-term progression.
Uses a constructive CSP approach with minimal color introduction."""
import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s limit
    n = 128

    # Precompute all 4-APs: list of (a, d) where a, a+d, a+2d, a+3d are all in [0, n-1]
    # Using 0-indexed positions 0..127 for positions 1..128
    aps = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 < n:
                aps.append((a, d))

    # For each position, precompute which APs it participates in
    # and what role (0,1,2,3) it plays
    pos_aps = [[] for _ in range(n)]  # pos_aps[i] = list of (ap_index, role)
    for idx, (a, d) in enumerate(aps):
        pos_aps[a].append((idx, 0))
        pos_aps[a + d].append((idx, 1))
        pos_aps[a + 2 * d].append((idx, 2))
        pos_aps[a + 3 * d].append((idx, 3))

    # Strategy: Try to construct with increasing number of colors
    # For each k, try to assign colors from {0, 1, ..., k-1} to positions 0..n-1
    # such that every 4-AP uses at least 3 distinct colors.
    #
    # Key constraint: In any 4-AP, at least 3 colors appear.
    # This means we CANNOT have patterns like:
    #   - All 4 same color
    #   - Exactly 2 colors with distribution (3,1) or (2,2)
    # So valid distributions are: (1,1,1,1) all distinct, or (2,1,1) two same + two other distinct
    #
    # Approach: Greedy assignment with backtracking, trying fewest colors first.
    # Use a smart order: assign positions that are in the most APs first.
    #
    # Better approach: Use iterative refinement.
    # 1. Start with a valid coloring (e.g., modular with known valid k)
    # 2. Try to reduce colors by merging/remapping

    # First, find a valid starting point using modular coloring
    def verify(colors):
        for (a, d) in aps:
            c = {colors[a], colors[a+d], colors[a+2*d], colors[a+3*d]}
            if len(c) < 3:
                return False
        return True

    # Find smallest valid modular coloring
    best_colors = None
    for k in range(2, 128):
        if time.time() > deadline - 5.0:
            break
        colors = [x % k for x in range(n)]
        if verify(colors):
            best_colors = colors
            break

    if best_colors is None:
        # Fallback: use k=43 which is known to work for these problems
        best_colors = [x % 43 for x in range(n)]
        if not verify(best_colors):
            # Try k=45
            best_colors = [x % 45 for x in range(n)]
            if not verify(best_colors):
                # Last resort: greedy construction
                best_colors = [0] * n
                for i in range(n):
                    for c in range(128):
                        best_colors[i] = c
                        if verify(best_colors[:i+1] + [0]*(n-i-1)):
                            break
                    # If no single color works, just use a new one
                    else:
                        best_colors[i] = i % 128

    # Now try to reduce the number of distinct colors used
    def count_colors(colors):
        return len(set(colors))

    current_colors = list(best_colors)
    current_k = count_colors(current_colors)

    # Refinement: try to merge colors by remapping
    # Idea: if two colors never appear in a position that would cause a violation
    # when merged, we can merge them.
    #
    # More practical approach: local search on color assignments
    # Try random remappings that reduce color count while maintaining validity

    if time.time() < deadline - 3.0:
        random.seed(42)
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            used_colors = list(set(current_colors))
            num_used = len(used_colors)
            if num_used <= 3:
                break

            # Try to remove one color by remapping its positions to other colors
            for target_color in used_colors:
                if time.time() > deadline - 3.0:
                    break
                # Find positions with target_color
                target_positions = [i for i in range(n) if current_colors[i] == target_color]
                if not target_positions:
                    continue

                # Try remapping each target position to a different color
                # and check if validity is maintained
                best_remap = None
                for pos in target_positions:
                    for new_color in used_colors:
                        if new_color == target_color:
                            continue
                        old = current_colors[pos]
                        current_colors[pos] = new_color
                        if verify(current_colors):
                            # Check if this reduces color count
                            new_used = set(current_colors)
                            if len(new_used) < num_used:
                                best_remap = (pos, new_color)
                                improved = True
                                num_used = len(new_used)
                                used_colors = list(new_used)
                                break
                        current_colors[pos] = old
                    if improved:
                        break
                if improved:
                    # Apply the remap
                    pos, new_color = best_remap
                    current_colors[pos] = new_color
                    break

    # Try a more aggressive approach: systematic color reduction
    # Try to find a coloring using exactly k colors for k < current count
    if time.time() < deadline - 5.0:
        for target_k in range(3, count_colors(current_colors)):
            if time.time() > deadline - 5.0:
                break

            # Try random assignment with exactly target_k colors
            found = False
            attempts = 200 if time.time() < deadline - 10.0 else 50
            for attempt in range(attempts):
                if time.time() > deadline - 5.0:
                    break
                colors = [random.randint(0, target_k - 1) for _ in range(n)]
                if verify(colors):
                    # Try to improve: see if we can do better
                    if count_colors(colors) <= target_k:
                        current_colors = colors
                        found = True
                        break

            if found:
                # Try to reduce further
                break

    # Final: try local search to minimize colors
    if time.time() < deadline - 2.0:
        best_count = count_colors(current_colors)
        # Simulated annealing-like approach
        temp = 10.0
        for step in range(5000):
            if time.time() > deadline - 2.0:
                break
            pos = random.randint(0, n - 1)
            old_color = current_colors[pos]
            new_color = random.randint(0, max(3, best_count) - 1)
            if new_color == old_color:
                continue
            current_colors[pos] = new_color
            if verify(current_colors):
                new_count = count_colors(current_colors)
                if new_count < best_count:
                    best_count = new_count
                    temp = max(1.0, temp * 0.9)
                else:
                    # Accept with some probability (annealing)
                    if random.random() < temp / best_count:
                        pass  # accept
                    else:
                        current_colors[pos] = old_color
            else:
                # Reject
                current_colors[pos] = old_color
            temp *= 0.999

    return current_colors
