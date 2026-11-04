"""
Difference triangle set (7,5) solver.

Strategy:
1. Generate a pool of candidate rows using randomized greedy with a heuristic
   that prioritizes preserving small differences (inspired by the 'small_count'
   heuristic from the 1.8052 solution).
2. Build a compatibility graph where two rows are compatible if their difference
   sets are disjoint.
3. Use a branch-and-bound maximum clique algorithm to find the largest set of
   mutually compatible rows.
4. Among solutions with the maximum number of rows, minimize the scope.
5. If time remains, try to improve the scope via local search and multiple
   randomized restarts with the heuristic-based greedy constructor.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return frozenset of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= set(_diffs_of_row(r))
    return used


def _generate_row_heuristic(used, k, rng, small_threshold=15, pool_size=50, best_scope=None):
    """
    Heuristic row construction that prioritizes candidates with fewer small differences.
    Returns (row, diffs) or None if stuck.
    """
    row = [0]
    new_diffs = set()
    
    while len(row) < k + 1:
        last_val = row[-1]
        valid_candidates = []
        
        # Determine upper bound for search
        if best_scope is not None:
            upper_bound = best_scope
        else:
            upper_bound = 200
        
        # Collect valid candidates
        for x in range(last_val + 1, upper_bound + 1):
            new = {x - a for a in row}
            if new & used or new & new_diffs:
                continue
            
            # Calculate heuristic score: number of small diffs introduced
            small_count = sum(1 for d in new if d < small_threshold)
            valid_candidates.append((small_count, x, new))
            
            if len(valid_candidates) >= pool_size:
                break
        
        if not valid_candidates:
            # Expand search if we hit the upper bound
            for x in range(upper_bound + 1, upper_bound + 100):
                new = {x - a for a in row}
                if new & used or new & new_diffs:
                    continue
                small_count = sum(1 for d in new if d < small_threshold)
                valid_candidates.append((small_count, x, new))
                if len(valid_candidates) >= pool_size:
                    break
            
            if not valid_candidates:
                return None, None
        
        # Sort candidates: Primary key: small_count (ascending), Secondary key: x (ascending)
        valid_candidates.sort(key=lambda item: (item[0], item[1]))
        
        # Selection Strategy:
        # Pick the best one (index 0) with 70% probability,
        # and a random one from the top 10 with 30% probability.
        if rng.random() < 0.7:
            chosen = valid_candidates[0]
        else:
            top_k = min(10, len(valid_candidates))
            chosen = rng.choice(valid_candidates[:top_k])
        
        x = chosen[1]
        new_diffs |= chosen[2]
        row.append(x)
    
    return row, frozenset(new_diffs)


def _generate_row_greedy_pure(used, k):
    """Pure greedy: always pick the smallest valid candidate."""
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, frozenset(new_diffs)


def _generate_candidate_pool(n_rows, k, rng, pool_size=200):
    """
    Generate a pool of candidate rows.
    Returns list of (row, diffs) tuples.
    """
    candidates = []
    
    # Start with pure greedy for a base
    used = set()
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k)
        if row is not None:
            candidates.append((row, diffs))
            used |= diffs
    
    # Generate more candidates with heuristic
    for _ in range(pool_size):
        # Use a random subset of already-used differences to guide generation
        used_subset = set()
        for c in candidates:
            if rng.random() < 0.3:
                used_subset |= c[1]
        
        row, diffs = _generate_row_heuristic(used_subset, k, rng)
        if row is not None:
            candidates.append((row, diffs))
    
    return candidates


def _build_compatibility_graph(candidates):
    """
    Build a compatibility graph where edge (i, j) means candidates i and j
    have disjoint difference sets.
    Returns adjacency list as list of sets.
    """
    n = len(candidates)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not (candidates[i][1] & candidates[j][1]):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _max_clique_branch_and_bound_v2(adj, deadline):
    """
    Improved branch-and-bound for maximum clique.
    Uses a more efficient coloring-based upper bound.
    """
    n = len(adj)
    if n == 0:
        return []
    
    best_clique = []
    best_size = 0
    node_count = [0]
    
    # Greedy coloring for upper bound
    def color_upper_bound(vertices):
        """
        Returns (ordered_vertices, upper_bounds) where upper_bounds[i]
        is an upper bound on the clique size that can be formed from
        vertices[i:] in the ordered list.
        """
        if not vertices:
            return [], []
        
        # Greedy sequential coloring
        color = [0] * len(vertices)
        vertex_colors = {}
        
        # Sort by degree descending for better coloring
        verts_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
        
        for v in verts_sorted:
            # Find smallest color not used by already-colored neighbors
            used = set()
            for u in adj[v]:
                if u in vertex_colors:
                    used.add(vertex_colors[u])
            c = 0
            while c in used:
                c += 1
            vertex_colors[v] = c
        
        # Now order by color ascending, then by degree
        def sort_key(v):
            return (vertex_colors[v], -len(adj[v]))
        
        ordered = sorted(vertices, key=sort_key)
        
        # Compute upper bounds: the color of vertex i gives an upper bound
        # on the number of vertices in a clique that includes vertex i
        # and only vertices from position i onward (in this ordering)
        upper_bounds = [0] * len(ordered)
        max_color = 0
        for i in range(len(ordered) - 1, -1, -1):
            max_color = max(max_color, vertex_colors[ordered[i]] + 1)
            upper_bounds[i] = max_color
        
        return ordered, upper_bounds
    
    def expand(clique, candidates):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        node_count[0] += 1
        
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        ordered, upper_bounds = color_upper_bound(candidates)
        
        for i, v in enumerate(ordered):
            if time.time() > deadline:
                return
            
            # Upper bound check
            if len(clique) + upper_bounds[i] <= best_size:
                break
            
            # Check compatibility with all in clique
            compatible = True
            for u in clique:
                if u not in adj[v]:
                    compatible = False
                    break
            if not compatible:
                continue
            
            clique.append(v)
            new_candidates = [u for u in ordered[i + 1:] if u in adj[v]]
            expand(clique, new_candidates)
            clique.pop()
    
    all_vertices = list(range(n))
    expand([], all_vertices)
    
    return best_clique


def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    scope = _total_scope(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort rows by their max element, descending
        row_indices = sorted(range(len(rows)), key=lambda i: -rows[i][-1])
        
        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]
            
            # Try to reduce the last element
            last = row[-1]
            for new_last in range(last - 1, row[-2], -1):
                old_diffs = {last - a for a in row[:-1]}
                new_diffs = {new_last - a for a in row[:-1]}
                remaining = used - old_diffs
                if not (new_diffs & remaining):
                    row[-1] = new_last
                    used = (used - old_diffs) | new_diffs
                    scope = _total_scope(rows)
                    improved = True
                    break
            
            # Try to reduce other elements
            for pos in range(len(row) - 2, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        scope = _total_scope(rows)
                        improved = True
                        break
    
    return rows


def _greedy_construct(n_rows, k, rng=None):
    """Pure greedy construction as fallback."""
    if rng is None:
        rng = random.Random(42)
    used = set()
    rows = []
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= diffs
    return rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Step 1: Multiple restarts with heuristic-based greedy construction
    best_result = None
    best_scope = float('inf')
    
    # Use the heuristic-based greedy constructor with multiple restarts
    max_restarts = 200
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 30:
            break
        
        used_diffs = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            if time.time() > deadline - 30:
                success = False
                break
            
            row, diffs = _generate_row_heuristic(used_diffs, k, rng, best_scope=best_scope if best_scope < float('inf') else None)
            if row is None:
                success = False
                break
            
            rows.append(row)
            used_diffs |= diffs
        
        if success and len(rows) == n_rows:
            current_scope = _total_scope(rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_result = [r[:] for r in rows]
                if best_scope <= 105:
                    break
    
    # Step 2: If we have a good result, try to improve it via local search
    if best_result is not None and time.time() < deadline - 10:
        best_result = _local_search_scope(best_result, k, rng, deadline - 10)
        best_scope = _total_scope(best_result)
    
    # Step 3: If we still have time, try the clique-based approach
    if time.time() < deadline - 30:
        candidates = _generate_candidate_pool(n_rows, k, rng, pool_size=150)
        
        # Deduplicate: keep only unique difference sets
        seen_diffs = set()
        unique_candidates = []
        for row, diffs in candidates:
            if diffs not in seen_diffs:
                seen_diffs.add(diffs)
                unique_candidates.append((row, diffs))
        
        candidates = unique_candidates
        
        if len(candidates) >= n_rows:
            # Build compatibility graph
            adj = _build_compatibility_graph(candidates)
            
            # Find maximum clique
            clique = _max_clique_branch_and_bound_v2(adj, deadline - 15)
            
            if len(clique) >= n_rows:
                selected = clique[:n_rows]
                rows = [candidates[i][0] for i in selected]
                scope = _total_scope(rows)
                
                if scope < best_scope:
                    best_scope = scope
                    best_result = rows
                    
                    # Try local search on this result
                    if time.time() < deadline - 10:
                        best_result = _local_search_scope(best_result, k, rng, deadline - 10)
                        best_scope = _total_scope(best_result)
    
    # Step 4: Fallback if no valid result found
    if best_result is None:
        rows = _greedy_construct(n_rows, k, rng)
        if rows is None:
            # Very simple fallback
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                rows.append(row)
            return rows
        best_result = rows
    
    return best_result

# EVOLVE-BLOCK-END
