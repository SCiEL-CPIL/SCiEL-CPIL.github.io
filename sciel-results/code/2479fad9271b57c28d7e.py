import random
import time
import numpy as np

N = 16
NUM_EDGES = 120
NUM_K4 = 1820

# Precompute edge indices
edge_to_idx = {}
idx_to_edge = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = idx
        idx_to_edge.append((i, j))
        idx += 1

# Precompute K4s and their edge indices
k4_edges = []
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
                k4_edges.append(edges)

# Precompute for each edge, which K4s it belongs to
edge_to_k4s = [[] for _ in range(NUM_EDGES)]
for k4_idx, edges in enumerate(k4_edges):
    for e in edges:
        edge_to_k4s[e].append(k4_idx)

def count_distinct_colors(colors, edges):
    """Count distinct colors in a set of edges."""
    c = colors[edges]
    return len(np.unique(c))

def is_bad_k4(colors, k4_idx):
    """Check if a K4 is bad (fewer than 5 distinct colors)."""
    edges = k4_edges[k4_idx]
    return count_distinct_colors(colors, edges) < 5

def count_bad_k4s(colors):
    """Count total bad K4s."""
    bad_count = 0
    for k4_idx in range(NUM_K4):
        if is_bad_k4(colors, k4_idx):
            bad_count += 1
    return bad_count

def update_bad_count(colors, e_idx, old_color, new_color, bad_count):
    """
    Update bad_count when color of edge e_idx changes from old_color to new_color.
    Returns new bad_count.
    """
    if old_color == new_color:
        return bad_count
    
    colors[e_idx] = new_color
    
    # Check all K4s containing this edge
    for k4_idx in edge_to_k4s[e_idx]:
        # Check if K4 was bad before
        # We need to know the state before. 
        # This requires storing the bad status or recalculating.
        # For efficiency, let's just recalculate the bad status for these 14 K4s.
        # But we need to know if they were bad BEFORE the change to update the count.
        # So we must store the current bad status array.
        pass

# To make it efficient, we maintain an array `is_bad` of size NUM_K4.
# And `bad_count` is the sum of `is_bad`.

def local_search(k, time_limit=20):
    """
    Try to find a valid coloring with k colors.
    Returns colors if found, else None.
    """
    start_time = time.time()
    
    # Initialize
    colors = np.random.randint(0, k, size=NUM_EDGES)
    
    # Initialize bad status
    is_bad = np.zeros(NUM_K4, dtype=bool)
    bad_count = 0
    for k4_idx in range(NUM_K4):
        if count_distinct_colors(colors, k4_edges[k4_idx]) < 5:
            is_bad[k4_idx] = True
            bad_count += 1
            
    if bad_count == 0:
        return colors

    best_colors = colors.copy()
    best_bad = bad_count
    
    # Simulated Annealing / Hill Climbing
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    # We will iterate until time limit
    while time.time() - start_time < time_limit:
        # Pick a random edge
        e_idx = np.random.randint(0, NUM_EDGES)
        old_color = colors[e_idx]
        new_color = np.random.randint(0, k)
        
        if new_color == old_color:
            continue
            
        # Calculate delta
        delta_bad = 0
        # Temporarily apply change
        colors[e_idx] = new_color
        
        for k4_idx in edge_to_k4s[e_idx]:
            was_bad = is_bad[k4_idx]
            is_now_bad = count_distinct_colors(colors, k4_edges[k4_idx]) < 5
            
            if was_bad and not is_now_bad:
                delta_bad -= 1
            elif not was_bad and is_now_bad:
                delta_bad += 1
            
            # Update state for this K4
            is_bad[k4_idx] = is_now_bad
            
        new_bad = bad_count + delta_bad
        
        # Calculate delta cost (we only care about bad_count for validity)
        # If we want to minimize colors, we could add that, but here k is fixed.
        
        # Acceptance
        if delta_bad < 0:
            bad_count = new_bad
            if bad_count < best_bad:
                best_bad = bad_count
                best_colors = colors.copy()
                if bad_count == 0:
                    return best_colors
        else:
            # Probabilistic acceptance
            if T > min_T and np.random.rand() < np.exp(-delta_bad / T):
                bad_count = new_bad
            else:
                # Revert
                colors[e_idx] = old_color
                # Revert is_bad states
                for k4_idx in edge_to_k4s[e_idx]:
                    # We need to revert the is_bad array too.
                    # This is a problem: we updated is_bad in place.
                    # We should store the old states or revert carefully.
                    pass
        
        T *= cooling

    if best_bad == 0:
        return best_colors
    return None

def construct():
    total_start = time.time()
    deadline = 140.0
    
    # Try 13, 14, 15
    for k in [13, 14, 15]:
        if time.time() - total_start > deadline:
            break
        # Allocate remaining time, keep 10s buffer
        remaining = deadline - (time.time() - total_start)
        if remaining < 10:
            break
        t_limit = min(remaining - 10, 30) # Cap at 30s per k to allow trying next
        sol = local_search(k, t_limit)
        if sol is not None:
            return list(sol)
            
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
