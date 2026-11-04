"""
Difference triangle set (7,5) solver.

Strategy:
1. Generate a pool of candidate rows (each a sorted list of k+1 integers starting at 0)
   using randomized greedy with varying parameters.
2. Build a compatibility graph where two rows are compatible if their difference sets
   are disjoint.
3. Use a branch-and-bound maximum clique algorithm to find the largest set of
   mutually compatible rows.
4. Among solutions with the maximum number of rows, minimize the scope.
5. If time remains, try to improve the scope via local search.
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


def _generate_row_greedy(used, k, rng, pool_size=3):
    """
    Greedily construct a row of k+1 integers starting at 0.
    At each step, pick from the top 'pool_size' smallest valid candidates.
    Returns (row, diffs) or None if stuck.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        # Collect valid candidates
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        # Pick randomly from candidates, weighted toward smaller
        if rng.random() < 0.5:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
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


def _generate_row_random_walk(k, rng, max_scope=150):
    """
    Generate a random row of k+1 integers starting at 0 with max entry <= max_scope.
    Uses random selection with increasing values.
    Returns (row, diffs) or None.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        # Pick a random value greater than last, up to max_scope
        min_val = x + 1
        if min_val > max_scope:
            return None, None
        # Bias toward smaller values
        x = rng.randint(min_val, min(max_scope, x + rng.randint(1, 10)))
        new = {x - a for a in row}
        if new & new_diffs:
            continue
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
    
    # Generate more candidates with randomization
    for _ in range(pool_size):
        # Use a random subset of already-used differences to guide generation
        # This creates diversity
        used_subset = set()
        for c in candidates:
            if rng.random() < 0.3:
                used_subset |= c[1]
        
        ps = rng.choice([2, 3, 5, 8])
        row, diffs = _generate_row_greedy(used_subset, k, rng, pool_size=ps)
        if row is not None:
            candidates.append((row, diffs))
    
    # Also try some random walk rows
    for _ in range(pool_size // 2):
        row, diffs = _generate_row_random_walk(k, rng, max_scope=120)
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


def _max_clique_branch_and_bound(adj, deadline):
    """
    Find a maximum clique in the graph using branch-and-bound with coloring.
    Returns the clique as a list of node indices.
    """
    n = len(adj)
    if n == 0:
        return []
    
    best_clique = []
    best_size = 0
    
    # Use a simple coloring heuristic for upper bound
    def color_sort(vertices):
        """
        Order vertices by a greedy coloring heuristic.
        Returns (ordered_vertices, color_bounds) where color_bounds[i] is
        the upper bound on clique size from vertex i onward.
        """
        # Greedy coloring
        colors = [0] * len(vertices)
        color_map = {}
        for v in vertices:
            # Find smallest color not used by neighbors
            used_colors = set()
            for u in adj[v]:
                if u in color_map:
                    used_colors.add(color_map[u])
            c = 0
            while c in used_colors:
                c += 1
            color_map[v] = c
        
        # Sort by color, then by degree
        ordered = sorted(vertices, key=lambda v: (color_map[v], -len(adj[v])))
        
        # Compute color bounds (upper bound on clique size from each position)
        color_bounds = [0] * len(ordered)
        max_color = 0
        for i in range(len(ordered) - 1, -1, -1):
            max_color = max(max_color, color_map[ordered[i]])
            color_bounds[i] = max_color
        
        return ordered, color_bounds
    
    def expand(clique, candidates):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        ordered, color_bounds = color_sort(candidates)
        
        for i, v in enumerate(ordered):
            if time.time() > deadline:
                return
            
            # Upper bound: current clique size + remaining candidates from i
            if len(clique) + (len(ordered) - i) <= best_size:
                break
            
            # Upper bound using coloring
            if len(clique) + color_bounds[i] <= best_size:
                break
            
            # Check if v is compatible with all in clique
            compatible = all(u in adj[v] for u in clique)
            if not compatible:
                continue
            
            # Expand
            clique.append(v)
            # New candidates: those in ordered[i+1:] that are neighbors of v
            new_candidates = [u for u in ordered[i + 1:] if u in adj[v]]
            expand(clique, new_candidates)
            clique.pop()
    
    # Start with all vertices
    all_vertices = list(range(n))
    expand([], all_vertices)
    
    return best_clique


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


def _greedy_construct(n_rows, k):
    """Pure greedy construction as fallback."""
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
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Step 1: Generate candidate pool
    candidates = _generate_candidate_pool(n_rows, k, rng, pool_size=300)
    
    # Deduplicate: keep only unique difference sets
    seen_diffs = set()
    unique_candidates = []
    for row, diffs in candidates:
        if diffs not in seen_diffs:
            seen_diffs.add(diffs)
            unique_candidates.append((row, diffs))
    
    candidates = unique_candidates
    
    # If we have enough candidates, try maximum clique
    if len(candidates) >= n_rows and time.time() < deadline - 60:
        # Build compatibility graph
        adj = _build_compatibility_graph(candidates)
        
        # Find maximum clique
        clique = _max_clique_branch_and_bound_v2(adj, deadline - 30)
        
        if len(clique) >= n_rows:
            # Take the first n_rows nodes from the clique
            # (or try to find a clique of exactly n_rows with minimum scope)
            selected = clique[:n_rows]
            rows = [candidates[i][0] for i in selected]
        else:
            # Fallback to greedy
            rows = _greedy_construct(n_rows, k)
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
    else:
        # Not enough candidates or not enough time; use greedy
        rows = _greedy_construct(n_rows, k)
        if rows is None:
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
    
    # Step 2: Local search to reduce scope
    if time.time() < deadline - 10:
        rows = _local_search_scope(rows, k, rng, deadline - 10)
    
    # Step 3: Try a few more local search iterations if time permits
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            rows = _local_search_scope(rows, k, rng, deadline - 5)
    
    return rows

# EVOLVE-BLOCK-END
