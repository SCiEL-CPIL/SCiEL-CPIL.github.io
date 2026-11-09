"""Strong edge colouring via iterative color elimination with constraint propagation."""
import time
import random
from collections import defaultdict

_DEADLINE = 140.0  # safety margin before 150s kill


def _build_instance(n, d, seed):
    rng = random.Random(seed)
    degree = [0] * n
    edges = set()
    for _ in range(4 * n * d):
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        e = (min(u, v), max(u, v))
        if degree[u] < d and degree[v] < d and e not in edges:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)


def _precompute_conflicts(edges):
    """Return conflict_mask[i] as a Python int bitmask of edges conflicting with edge i."""
    m = len(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    conflict_mask = [0] * m
    for i in range(m):
        u1, v1 = edges[i]
        for j in range(i + 1, m):
            u2, v2 = edges[j]
            if u1 in (u2, v2) or v1 in (u2, v2):
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
            elif (u2 in adj[u1] or u2 in adj[v1] or v2 in adj[u1] or v2 in adj[v1]):
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    return conflict_mask


def _dsatur_initial_coloring(edges, conflict_mask):
    """
    DSATUR (Degree of Saturation) heuristic for initial coloring.
    Returns a valid coloring (list of color indices per edge).
    """
    m = len(edges)
    if m == 0:
        return []
    
    # color[e] = color index assigned to edge e, or -1 if uncolored
    color = [-1] * m
    # For each color, track which edges have that color
    color_classes = defaultdict(set)
    # For each vertex, track which colors are "used" (have an incident edge)
    # Actually, we need to track which colors appear on edges incident to each vertex
    # and also which colors appear on edges that conflict with edges incident to each vertex
    
    # DSATUR: pick uncolored edge with highest saturation (number of distinct colors
    # in its conflict neighborhood), break ties by degree (number of conflicts)
    
    for step in range(m):
        best_edge = -1
        best_sat = -1
        best_deg = -1
        
        for e in range(m):
            if color[e] != -1:
                continue
            # Compute saturation: number of distinct colors in conflict neighborhood
            neighbors_colors = set()
            cm = conflict_mask[e]
            for j in range(m):
                if cm & (1 << j) and color[j] != -1:
                    neighbors_colors.add(color[j])
            sat = len(neighbors_colors)
            deg = bin(cm).count('1')
            
            if sat > best_sat or (sat == best_sat and deg > best_deg):
                best_sat = sat
                best_deg = deg
                best_edge = e
        
        if best_edge == -1:
            break
        
        # Assign the smallest available color
        used_colors = set()
        cm = conflict_mask[best_edge]
        for j in range(m):
            if cm & (1 << j) and color[j] != -1:
                used_colors.add(color[j])
        
        c = 0
        while c in used_colors:
            c += 1
        
        color[best_edge] = c
        color_classes[c].add(best_edge)
    
    return color


def _try_merge_colors(color, conflict_mask, c1, c2, m):
    """
    Try to merge color class c2 into c1.
    Returns True if merge is valid (no conflicts within merged class).
    """
    # Check: no two edges in c2's class conflict with any edge in c1's class
    # AND no two edges within c2's class conflict with each other (should already be valid)
    # We need: for every edge e in c2, no edge in c1 conflicts with e
    
    # Get edges in each class
    edges_c1 = [e for e in range(m) if color[e] == c1]
    edges_c2 = [e for e in range(m) if color[e] == c2]
    
    for e2 in edges_c2:
        for e1 in edges_c1:
            if conflict_mask[e2] & (1 << e1):
                return False
    return True


def _eliminate_colors(color, conflict_mask, deadline, rng):
    """
    Iteratively try to eliminate colors by merging classes.
    Returns improved coloring.
    """
    m = len(color)
    if m == 0:
        return color
    
    # Work on a copy
    color = list(color)
    num_colors = max(color) + 1 if color else 0
    
    # Precompute which edges have which color for fast lookup
    # We'll maintain this as we merge
    
    improved = True
    while improved and time.time() < deadline - 0.5:
        improved = False
        num_colors = max(color) + 1 if color else 0
        
        # Try all pairs of colors, attempt to merge
        # To speed up, try pairs in random order
        pairs = [(i, j) for i in range(num_colors) for j in range(i + 1, num_colors)]
        rng.shuffle(pairs)
        
        for c1, c2 in pairs:
            if time.time() > deadline - 0.5:
                break
            
            # Check if both colors still exist
            if not any(c == c1 for c in color) or not any(c == c2 for c in color):
                continue
            
            if _try_merge_colors(color, conflict_mask, c1, c2, m):
                # Merge c2 into c1
                for e in range(m):
                    if color[e] == c2:
                        color[e] = c1
                improved = True
                break  # Restart the pair loop after a merge
    
    return color


def _local_search_swap(color, conflict_mask, deadline, rng):
    """
    Local search: try to reduce colors by swapping edge colors.
    An edge can change color if the new color class doesn't conflict.
    """
    m = len(color)
    if m == 0:
        return color
    
    color = list(color)
    num_colors = max(color) + 1 if color else 0
    
    # Precompute for each color, the set of edges
    # And for each edge, the set of colors it cannot take
    
    for iteration in range(200):
        if time.time() > deadline - 0.5:
            break
        
        # Try to reduce num_colors by 1
        # Pick a color to eliminate (the one with fewest edges)
        color_counts = defaultdict(int)
        for c in color:
            color_counts[c] += 1
        
        if not color_counts:
            break
        
        # Try to eliminate the least-used color
        target_color = min(color_counts, key=lambda c: color_counts[c])
        target_edges = [e for e in range(m) if color[e] == target_color]
        
        if not target_edges:
            break
        
        # For each edge in target_color, try to move it to another color
        success = True
        for e in target_edges:
            # Find a color it can take
            moved = False
            for new_c in range(num_colors):
                if new_c == target_color:
                    continue
                # Check if e can have color new_c
                # i.e., no edge with color new_c conflicts with e
                can_take = True
                cm = conflict_mask[e]
                for j in range(m):
                    if color[j] == new_c and (cm & (1 << j)):
                        can_take = False
                        break
                if can_take:
                    color[e] = new_c
                    moved = True
                    break
            if not moved:
                success = False
                break
        
        if success:
            # Renumber colors to be contiguous
            old_to_new = {}
            new_idx = 0
            for c in range(num_colors):
                if any(color[e] == c for e in range(m)):
                    if c not in old_to_new:
                        old_to_new[c] = new_idx
                        new_idx += 1
            for e in range(m):
                color[e] = old_to_new[color[e]]
            num_colors = new_idx
    
    return color


def _solve_instance(edges, conflict_mask, deadline):
    m = len(edges)
    if m == 0:
        return []
    
    rng = random.Random()
    
    best_color = None
    best_num = m + 1
    
    # Multiple restarts with DSATUR + elimination + local search
    max_restarts = 30
    restarts = 0
    
    while restarts < max_restarts and time.time() < deadline - 1.0:
        restarts += 1
        
        # DSATUR initial coloring
        color = _dsatur_initial_coloring(edges, conflict_mask)
        
        # Iterative color elimination
        color = _eliminate_colors(color, conflict_mask, deadline, rng)
        
        # Local search
        color = _local_search_swap(color, conflict_mask, deadline, rng)
        
        num_colors = max(color) + 1 if color else 0
        if num_colors < best_num:
            best_num = num_colors
            best_color = list(color)
    
    if best_color is None:
        # Fallback: each edge its own color
        best_color = list(range(m))
        best_num = m
    
    return best_color


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    out = []
    
    for idx, spec in enumerate(specs):
        edges = _build_instance(*spec)
        conflict_mask = _precompute_conflicts(edges)
        
        # Allocate time budget per instance
        time_left = deadline - time.time()
        remaining = len(specs) - idx
        instance_deadline = time.time() + max(1.0, (time_left - 1.0) / remaining)
        
        colour = _solve_instance(edges, conflict_mask, instance_deadline)
        out.append(colour)
        
    return out
