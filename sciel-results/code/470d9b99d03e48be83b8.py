import time
import random
import sys

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

def _generate_row_randomized_greedy(used, k, rng, max_scope=200):
    """
    Randomized greedy construction of a row.
    Favors smaller values to keep scope low.
    """
    row = [0]
    new_diffs = set()
    x = 0
    
    while len(row) < k + 1:
        # Find valid candidates
        valids = []
        cap = max_scope
        if used:
            # Estimate a tighter cap based on used diffs
            # This is a heuristic to speed up search
            cap = min(max_scope, max(used) + 10)
            
        for x in range(row[-1] + 1, cap + 1):
            new_diffs_x = {x - a for a in row}
            if not (new_diffs_x & used) and not (new_diffs_x & new_diffs) and len(new_diffs_x) == len(row):
                valids.append(x)
            if len(valids) > 20:
                break
        
        if not valids:
            return None, None
        
        # Weighted random choice, favoring smaller x
        if len(valids) > 1:
            consider = valids[:10]
            weights = [1.0 / (v + 1) for v in consider]
            total_w = sum(weights)
            r = rng.uniform(0, total_w)
            acc = 0
            chosen = consider[0]
            for v, w in zip(consider, weights):
                acc += w
                if r <= acc:
                    chosen = v
                    break
        else:
            chosen = valids[0]
        
        row.append(chosen)
        for a in row[:-1]:
            new_diffs.add(chosen - a)
    
    return row, frozenset(new_diffs)

def _generate_candidate_pool(n_rows, k, rng, pool_size=500):
    """
    Generate a pool of candidate rows.
    """
    candidates = []
    seen_diffs = set()
    
    # 1. Pure greedy base
    used = set()
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k)
        if row is not None:
            if diffs not in seen_diffs:
                candidates.append((row, diffs))
                seen_diffs.add(diffs)
            used |= diffs
    
    # 2. Randomized greedy
    for _ in range(pool_size):
        # Use a random subset of used diffs to guide generation
        used_subset = set()
        for c in candidates:
            if rng.random() < 0.2:
                used_subset |= c[1]
        
        row, diffs = _generate_row_randomized_greedy(used_subset, k, rng, max_scope=150)
        if row is not None and diffs not in seen_diffs:
            candidates.append((row, diffs))
            seen_diffs.add(diffs)
            
    return candidates

def _build_compatibility_graph(candidates):
    """
    Build a compatibility graph where edge (i, j) means candidates i and j
    have disjoint difference sets.
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
    """
    n = len(adj)
    if n == 0:
        return []
    
    best_clique = []
    best_size = 0
    
    def color_sort(vertices):
        """
        Order vertices by a greedy coloring heuristic.
        """
        if not vertices:
            return [], []
            
        color_map = {}
        verts_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
        
        for v in verts_sorted:
            used_colors = set()
            for u in adj[v]:
                if u in color_map:
                    used_colors.add(color_map[u])
            c = 0
            while c in used_colors:
                c += 1
            color_map[v] = c
        
        ordered = sorted(vertices, key=lambda v: (color_map[v], -len(adj[v])))
        
        color_bounds = [0] * len(ordered)
        max_color = 0
        for i in range(len(ordered) - 1, -1, -1):
            max_color = max(max_color, color_map[ordered[i]] + 1)
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
            
            if len(clique) + color_bounds[i] <= best_size:
                break
            
            compatible = all(u in adj[v] for u in clique)
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
    best_rows = [r[:] for r in rows]
    best_scope = _total_scope(rows)
    
    num_passes = 20
    for p in range(num_passes):
        if time.time() > deadline:
            break
            
        order = list(range(len(rows)))
        rng.shuffle(order)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in order:
                if time.time() > deadline:
                    break
                row = rows[idx]
                
                # Precompute diffs of other rows
                other_diffs = set()
                for r_idx, r in enumerate(rows):
                    if r_idx != idx:
                        other_diffs |= set(_diffs_of_row(r))
                
                # Try to reduce each element from last to first
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    min_val = row[pos - 1] + 1
                    max_val = old_val - 1
                    
                    if max_val < min_val:
                        continue
                    
                    found = False
                    # Try smaller values
                    for new_val in range(min_val, max_val + 1):
                        # Check if new value is valid
                        is_valid = True
                        # Check strictly increasing
                        if new_val <= row[pos - 1]:
                            is_valid = False
                        if pos < len(row) - 1 and new_val >= row[pos + 1]:
                            is_valid = False
                        
                        if not is_valid:
                            continue
                        
                        # Compute new diffs for this row
                        new_row = row[:]
                        new_row[pos] = new_val
                        new_diffs = set(_diffs_of_row(new_row))
                        
                        # Check for internal duplicates
                        expected_diffs = len(new_row) * (len(new_row) - 1) // 2
                        if len(new_diffs) != expected_diffs:
                            continue
                        
                        # Check for conflicts with other rows
                        if new_diffs & other_diffs:
                            continue
                        
                        # Accept change
                        rows[idx] = new_row
                        new_scope = _total_scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [r[:] for r in rows]
                        improved = True
                        found = True
                        break
                    
                    if found:
                        break
    
    return best_rows

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
    deadline = start_time + 110  # 10s safety margin
    
    # Step 1: Generate candidate pool
    candidates = _generate_candidate_pool(n_rows, k, rng, pool_size=300)
    
    # If we have enough candidates, try maximum clique
    if len(candidates) >= n_rows and time.time() < deadline - 30:
        # Build compatibility graph
        adj = _build_compatibility_graph(candidates)
        
        # Find maximum clique
        clique = _max_clique_branch_and_bound(adj, deadline - 20)
        
        if len(clique) >= n_rows:
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
    if time.time() < deadline - 5:
        rows = _local_search_scope(rows, k, rng, deadline - 5)
        
    return rows

# EVOLVE-BLOCK-END
