import random
import time
import numpy as np

# Precompute all K4s and their edge indices
# Vertices 0..15
# Edges are indexed by (i, j) with i < j
# We need a mapping from (i, j) to index 0..119
# And for each K4 (a, b, c, d), the 6 edge indices.

N = 16
edge_to_idx = {}
idx_to_edge = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = idx
        idx_to_edge.append((i, j))
        idx += 1

# List of K4s
k4s = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                edges = [
                    edge_to_idx[(a, b)],
                    edge_to_idx[(a, c)],
                    edge_to_idx[(a, d)],
                    edge_to_idx[(b, c)],
                    edge_to_idx[(b, d)],
                    edge_to_idx[(c, d)]
                ]
                k4s.append(edges)

NUM_K4 = len(k4s)

def count_bad_k4s(colors):
    """
    colors: numpy array of shape (120,)
    Returns the number of K4s that have fewer than 5 distinct colours.
    """
    bad_count = 0
    for edges in k4s:
        # Get colours of the 6 edges
        c = colors[edges]
        # Count distinct colours
        if len(np.unique(c)) < 5:
            bad_count += 1
    return bad_count

def local_search(k, max_iter=50000, time_limit=10):
    """
    Try to find a valid colouring with k colours.
    Returns (success, colors)
    """
    start_time = time.time()
    # Initialize with random colours
    colors = np.random.randint(0, k, size=120)
    
    current_bad = count_bad_k4s(colors)
    if current_bad == 0:
        return True, colors

    best_bad = current_bad
    best_colors = colors.copy()

    # Simple hill climbing with random restarts
    # Since the space is small, we can try many restarts
    restarts = 0
    max_restarts = 100
    
    while restarts < max_restarts:
        if time.time() - start_time > time_limit:
            break
        
        # Randomize current state if stuck or at start
        if restarts > 0:
            colors = np.random.randint(0, k, size=120)
            current_bad = count_bad_k4s(colors)
            if current_bad == 0:
                return True, colors
            if current_bad < best_bad:
                best_bad = current_bad
                best_colors = colors.copy()

        # Hill climb
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random edge
            e_idx = np.random.randint(0, 120)
            old_color = colors[e_idx]
            new_color = np.random.randint(0, k)
            if new_color == old_color:
                continue
            
            # Calculate delta in bad K4s
            # Only K4s containing edge e_idx are affected
            # Pre-compute which K4s contain which edge for speed?
            # For now, just recalculate if it's fast enough.
            # 1820 K4s * np.unique is fast.
            
            colors[e_idx] = new_color
            new_bad = count_bad_k4s(colors)
            
            if new_bad < current_bad:
                current_bad = new_bad
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = colors.copy()
                    if current_bad == 0:
                        return True, best_colors
            else:
                # Reject
                colors[e_idx] = old_color
            
            # Check time
            if time.time() - start_time > time_limit:
                break

        restarts += 1

    if best_bad == 0:
        return True, best_colors
    return False, best_colors

def construct():
    # Deadline guard: 150s total, keep 10s buffer
    total_start = time.time()
    deadline = 140.0
    
    # Try to find a solution with 13, 14, 15 colours
    # If found, return it.
    # If not, return the baseline (17 colours).
    
    best_valid_colors = None
    
    # Try 13 colours
    if time.time() - total_start < deadline:
        success, colors = local_search(13, time_limit=30)
        if success:
            best_valid_colors = colors
    
    # Try 14 colours
    if time.time() - total_start < deadline and best_valid_colors is None:
        success, colors = local_search(14, time_limit=30)
        if success:
            best_valid_colors = colors

    # Try 15 colours
    if time.time() - total_start < deadline and best_valid_colors is None:
        success, colors = local_search(15, time_limit=30)
        if success:
            best_valid_colors = colors

    # If we found a valid colouring, return it
    if best_valid_colors is not None:
        # Ensure colours are in 0..119 (they are 0..k-1, so fine)
        # Return in lexicographic order
        return list(best_valid_colors)
    
    # Fallback to baseline
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
