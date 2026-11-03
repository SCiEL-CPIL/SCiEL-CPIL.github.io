import time
import random
import sys
from itertools import combinations

sys.setrecursionlimit(10000)


def _row_diff_set(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _generate_sidon_rows(k, M, max_rows=None):
    """Generate all valid Sidon-like rows (differences within row are distinct)
    of length k+1 starting at 0, with max element <= M.
    
    A row is 'Sidon-like' if all internal differences are distinct.
    """
    rows = []
    
    def backtrack(pos, current, curr_diffs):
        if max_rows is not None and len(rows) >= max_rows:
            return
        if pos == k + 1:
            rows.append(current[:])
            return
        
        start = current[-1] + 1
        for val in range(start, M + 1):
            # Check all new differences
            valid = True
            new_diffs = []
            for x in current:
                d = val - x
                if d in curr_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if not valid:
                continue
            
            current.append(val)
            curr_diffs.update(new_diffs)
            backtrack(pos + 1, current, curr_diffs)
            current.pop()
            curr_diffs.difference_update(new_diffs)
    
    backtrack(1, [0], set())
    return rows


def _dsatur_color_graph(adjacency, n_vertices):
    """DSATUR greedy coloring. Returns a list of color classes (sets of vertices).
    adjacency[v] is a set of vertices adjacent to v.
    """
    # Colors assigned so far
    color = [-1] * n_vertices
    # For each color, the set of vertices with that color
    color_classes = []
    
    # Track saturation degree: number of different colors in neighbors
    def get_saturation(v, color):
        if color[v] != -1:
            return 0
        neighbor_colors = set()
        for u in adjacency[v]:
            if color[u] != -1:
                neighbor_colors.add(color[u])
        return len(neighbor_colors)
    
    # Track available colors for each uncolored vertex
    def available_colors(v, color, n_colors):
        neighbor_colors = set()
        for u in adjacency[v]:
            if color[u] != -1:
                neighbor_colors.add(color[u])
        return set(range(n_colors)) - neighbor_colors
    
    # We'll grow the number of colors as needed
    n_colors = 1
    uncolored = set(range(n_vertices))
    
    while uncolored:
        # Pick the vertex with highest saturation degree (DSATUR heuristic)
        # Among ties, pick the one with highest degree to already-colored vertices
        best_v = None
        best_sat = -1
        best_deg = -1
        for v in uncolored:
            sat = get_saturation(v, color)
            # Count colored neighbors
            deg = sum(1 for u in adjacency[v] if color[u] != -1)
            if sat > best_sat or (sat == best_sat and deg > best_deg):
                best_sat = sat
                best_deg = deg
                best_v = v
        
        v = best_v
        uncolored.remove(v)
        
        # Find the lowest available color
        avail = available_colors(v, color, n_colors)
        if avail:
            c = min(avail)
            color[v] = c
            if c >= len(color_classes):
                color_classes.append(set())
            color_classes[c].add(v)
        else:
            # Need a new color
            c = n_colors
            color[v] = c
            color_classes.append({v})
            n_colors += 1
    
    return color_classes


def _build_rows_from_color_classes(rows_list, color_classes, k, M):
    """Build actual row sequences from selected color classes.
    Each color class represents a set of row patterns that don't share differences.
    But we need exactly one row per color class for the final answer.
    Actually, in our formulation, each vertex IS a row pattern.
    We want to select n_rows vertices that are mutually non-adjacent (independent set).
    DSATUR gives us color classes where each class is an independent set.
    We pick one vertex from each of the first n_rows color classes.
    """
    selected = []
    for i, cls in enumerate(color_classes):
        if i >= n_rows:
            break
        # Pick a representative from this color class
        # Prefer the one with smallest max element for tighter scope
        best_v = min(cls, key=lambda v: rows_list[v][-1])
        selected.append(rows_list[best_v])
    return selected


def _greedy_construct(n_rows, k):
    """Fallback greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _local_search_improve(rows, k, max_iter=5000, deadline=None):
    """Try to reduce the scope by local moves on individual elements."""
    if deadline is None:
        deadline = time.time() + 30
    
    best_rows = [row[:] for row in rows]
    best_scope = max(max(row) for row in best_rows)
    used_diffs = set()
    for row in best_rows:
        used_diffs |= _row_diff_set(row)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r_idx in range(len(best_rows)):
            row = best_rows[r_idx]
            for pos in range(1, len(row)):  # Skip 0
                orig_val = row[pos]
                # Try to move this element
                for new_val in range(row[pos-1] + 1, row[pos] + 20):
                    if new_val == orig_val:
                        continue
                    # Check if new position is valid
                    new_row = row[:]
                    new_row[pos] = new_val
                    # Check increasing
                    if new_row[pos] <= new_row[pos-1] or (pos < len(new_row)-1 and new_row[pos] >= new_row[pos+1]):
                        continue
                    new_diffs = _row_diff_set(new_row)
                    # Remove old diffs for this row, add new ones
                    old_diffs = _row_diff_set(row)
                    # Check no collision with other rows' diffs
                    other_diffs = set()
                    for i, other_row in enumerate(best_rows):
                        if i != r_idx:
                            other_diffs |= _row_diff_set(other_row)
                    
                    if not (new_diffs & other_diffs):
                        # Also check internal consistency (already ensured by _row_diff_set being a set)
                        # Accept the move
                        best_rows[r_idx] = new_row
                        new_scope = max(max(r) for r in best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            improved = True
                        break
    
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    # Counting bound
    min_M = n_rows * k * (k + 1) // 2
    
    # Strategy 1: Try to find a solution via DSATUR-based row selection
    # We generate candidate rows and use graph coloring to find independent sets
    
    best_result = None
    best_scope = float('inf')
    
    # Try increasing M values
    # For each M, generate candidate Sidon-like rows, build conflict graph,
    # use DSATUR to find n_rows mutually compatible rows
    
    # To keep generation tractable, limit the number of candidate rows
    max_candidates = 500
    
    for M in range(min_M, min_M + 60):
        if time.time() - start_time > timeout - 10:
            break
        
        # Generate candidate rows for this M
        candidates = _generate_sidon_rows(k, M, max_rows=max_candidates)
        
        if len(candidates) < n_rows:
            continue
        
        # Build conflict graph: two rows conflict if they share any difference
        n_cands = len(candidates)
        row_diffs = [_row_diff_set(c) for c in candidates]
        
        # Build adjacency
        adjacency = [set() for _ in range(n_cands)]
        for i in range(n_cands):
            for j in range(i + 1, n_cands):
                if row_diffs[i] & row_diffs[j]:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        
        # Use DSATUR to find color classes
        color_classes = _dsatur_color_graph(adjacency, n_cands)
        
        # We need n_rows independent sets (one row each)
        # Each color class is an independent set; pick one vertex from each
        if len(color_classes) >= n_rows:
            selected = _build_rows_from_color_classes(candidates, color_classes, k, M)
            if len(selected) == n_rows:
                # Verify validity
                all_diffs = set()
                valid = True
                for row in selected:
                    diffs = _row_diff_set(row)
                    if diffs & all_diffs:
                        valid = False
                        break
                    all_diffs |= diffs
                if valid:
                    scope = max(max(row) for row in selected)
                    if scope < best_scope:
                        best_scope = scope
                        best_result = [row[:] for row in selected]
        
        # If we found a solution at the minimum possible M, we can stop
        if best_result is not None and best_scope <= M:
            break
    
    # Try to improve via local search
    if best_result is not None:
        improved_rows, improved_scope = _local_search_improve(
            best_result, k, max_iter=3000, 
            deadline=time.time() + 20
        )
        if improved_scope < best_scope:
            best_result = improved_rows
            best_scope = improved_scope
    
    # Fallback to greedy if no solution found
    if best_result is None:
        best_result = _greedy_construct(n_rows, k)
    
    return best_result
