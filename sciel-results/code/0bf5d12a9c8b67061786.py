"""Rewritten program: minimize monochromatic 4-term APs in [1,256] via local search with delta updates."""
import time
import random
import numpy as np

N = 256
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _build_aps():
    """Build list of all 4-term arithmetic progressions (0-indexed) in 0..255."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _count_mono(colors, aps):
    """Count monochromatic 4-APs."""
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count

def _local_search(colors, aps, max_iter=50000):
    """
    Greedy local search using O(1) amortized delta calculation.
    For each position i, precompute which APs contain i.
    Flipping i changes the mono-count by:
      - For each AP containing i: if all 4 were same color, flip breaks it (-1)
                                   if 3 same + i different, flip makes it mono (+1)
    We compute delta for flipping position i.
    """
    colors = list(colors)
    n = len(colors)
    total = _count_mono(colors, aps)
    
    # Build adjacency: for each position, list of (ap_index, position_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_aps[a].append((idx, 0))
        pos_aps[b].append((idx, 1))
        pos_aps[c].append((idx, 2))
        pos_aps[d].append((idx, 3))
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n):
            # Compute delta if we flip position i
            delta = 0
            for (ap_idx, pos_in_ap) in pos_aps[i]:
                a, b, c, d = aps[ap_idx]
                # Current colors of the 4 positions
                vals = [colors[a], colors[b], colors[c], colors[d]]
                # If all 4 same: flipping i breaks it -> delta -= 1
                if vals[0] == vals[1] == vals[2] == vals[3]:
                    delta -= 1
                else:
                    # Count how many of the other 3 (excluding i) match the flipped value
                    new_val = 1 - colors[i]
                    # Check if after flip, all 4 would be same
                    # i.e., the other 3 must all equal new_val
                    others = [vals[j] for j in range(4) if j != pos_in_ap]
                    if all(o == new_val for o in others):
                        delta += 1
            if delta < 0:
                colors[i] = 1 - colors[i]
                total += delta
                improved = True
    
    return colors, total

def _random_coloring():
    return [random.randint(0, 1) for _ in range(N)]

def construct():
    start_time = time.time()
    aps = _build_aps()
    
    best_colors = None
    best_count = float('inf')
    
    # Try multiple random starts
    n_starts = 20
    for s in range(n_starts):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        colors = _random_coloring()
        colors, count = _local_search(colors, aps, max_iter=20000)
        
        if count < best_count:
            best_count = count
            best_colors = colors[:]
        
        # If we've found a very good solution, do more local search on it
        if count <= best_count:
            elapsed = time.time() - start_time
            remaining = TIME_LIMIT - elapsed
            if remaining > 5:
                # Do additional local search passes
                colors, count2 = _local_search(colors, aps, max_iter=min(50000, int(remaining * 1000)))
                if count2 < best_count:
                    best_count = count2
                    best_colors = colors[:]
    
    if best_colors is None:
        # Fallback: simple alternating
        best_colors = [(i // 2) % 2 for i in range(N)]
    
    return best_colors
