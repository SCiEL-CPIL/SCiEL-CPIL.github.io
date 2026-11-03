"""Difference Triangle Set (4,7) solver.

Strategy:
1. Fast greedy baseline.
2. Enumerate all valid row patterns (Sidon-like sequences) up to a scope bound.
3. Formulate row selection as a maximum independent set problem on a conflict graph.
4. Solve the MIS problem with a custom branch-and-bound solver.
5. Local compaction to reduce the scope further.
6. Simulated annealing for final polishing.
"""
import time
import random
import sys
from collections import defaultdict

def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k, rng=None):
    """Greedy ruler construction."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < 0.02:
                x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _generate_row_patterns(k, max_scope, rng=None):
    """Generate all valid row patterns (sequences of length k+1 starting at 0)
    with maximum entry <= max_scope that have unique internal differences."""
    L = k + 1
    patterns = []
    
    def backtrack(pos, current_row, used_diffs):
        if pos == L:
            patterns.append(tuple(current_row))
            return
        
        # Determine minimum possible next value
        min_val = current_row[-1] + 1 if current_row else 1
        # Determine maximum possible next value
        max_val = max_scope
        
        for val in range(min_val, max_val + 1):
            # Check if adding val creates duplicate differences
            new_diffs = set()
            valid = True
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                backtrack(pos + 1, current_row + [val], used_diffs | new_diffs)
    
    backtrack(0, [0], set())
    return patterns

def _solve_mis_branch_and_bound(adj, n, time_limit):
    """Solve maximum independent set using branch and bound.
    adj: list of sets where adj[i] contains neighbors of i
    Returns the maximum independent set as a list of indices."""
    
    best_set = []
    best_size = 0
    start_time = time.time()
    
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    def bound(subset):
        """Upper bound on size of independent set from subset."""
        if not subset:
            return 0
        # Simple bound: sum of 1/(1+degree) for nodes in subset
        # This is a valid upper bound for MIS
        total = 0.0
        for v in subset:
            total += 1.0 / (1.0 + degrees[v])
        return int(total) + 1
    
    def search(subset, current_set):
        nonlocal best_set, best_size
        if time.time() - start_time > time_limit:
            return
        
        if not subset:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Bound check
        if len(current_set) + bound(subset) <= best_size:
            return
        
        # Select vertex with maximum degree in induced subgraph
        max_deg = -1
        pivot = -1
        for v in subset:
            deg = sum(1 for u in subset if u in adj[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        # Branch 1: Include pivot
        new_current = current_set + [pivot]
        new_subset = [v for v in subset if v != pivot and v not in adj[pivot]]
        search(new_subset, new_current)
        
        # Branch 2: Exclude pivot
        new_subset = [v for v in subset if v != pivot]
        search(new_subset, current_set)
    
    search(list(range(n)), [])
    return best_set

def _construct_from_patterns(patterns, n_rows, k, time_limit):
    """Select n_rows non-conflicting patterns from the list."""
    if len(patterns) < n_rows:
        return None
    
    # Build conflict graph: two patterns conflict if they share any difference
    n = len(patterns)
    pattern_diffs = []
    for pat in patterns:
        diffs = set()
        for i in range(len(pat)):
            for j in range(i + 1, len(pat)):
                diffs.add(pat[j] - pat[i])
        pattern_diffs.append(diffs)
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if pattern_diffs[i] & pattern_diffs[j]:
                adj[i].add(j)
                adj[j].add(i)
    
    # Solve MIS
    mis = _solve_mis_branch_and_bound(adj, n, time_limit)
    
    if len(mis) >= n_rows:
        # Select first n_rows patterns from the MIS
        selected = mis[:n_rows]
        return [list(patterns[i]) for i in selected]
    
    return None

def _local_compact(rows, n_rows, k, time_limit):
    """Iteratively compact each element to its smallest feasible value."""
    work = [list(r) for r in rows]
    used_set = set()
    L = k + 1
    for row in work:
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(row[j] - row[i])
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for r_idx in range(n_rows):
            row = work[r_idx]
            for pos in range(L - 1, 0, -1):
                old_val = row[pos]
                lower = row[pos - 1] + 1
                if lower >= old_val:
                    continue
                
                old_diffs = set()
                for i in range(L):
                    if i < pos:
                        old_diffs.add(row[pos] - row[i])
                    elif i > pos:
                        old_diffs.add(row[i] - row[pos])
                
                for new_val in range(lower, old_val):
                    if pos < L - 1 and new_val >= row[pos + 1]:
                        break
                    
                    new_diffs = set()
                    for i in range(L):
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        elif i > pos:
                            new_diffs.add(row[i] - new_val)
                    
                    other = used_set - old_diffs
                    if new_diffs & other:
                        continue
                    
                    for d in old_diffs:
                        used_set.discard(d)
                    for d in new_diffs:
                        used_set.add(d)
                    row[pos] = new_val
                    improved = True
                    break
    
    return work

def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = set()
    for row in work:
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(row[j] - row[i])
    
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max
    
    start_temp = max(1, current_max // 8)
    end_temp = 0.05
    cooling = 0.9995
    
    t = start_temp
    stagnation_counter = 0
    max_stagnation = 2000
    
    while time.time() < time_limit and t > end_temp:
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to decrease a random element
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                t *= cooling
                continue
            
            new_val = rng.randint(lower, old_val - 1)
            if new_val >= old_val:
                t *= cooling
                continue
            
            old_diffs = set()
            for i in range(L):
                if i < pos:
                    old_diffs.add(row[pos] - row[i])
                elif i > pos:
                    old_diffs.add(row[i] - row[pos])
            
            new_diffs = set()
            for i in range(L):
                if i < pos:
                    new_diffs.add(new_val - row[i])
                elif i > pos:
                    new_diffs.add(row[i] - new_val)
            
            other = used_set - old_diffs
            if not (new_diffs & other):
                for d in old_diffs:
                    used_set.discard(d)
                for d in new_diffs:
                    used_set.add(d)
                row[pos] = new_val
                
                # Try to compact further
                for p in range(L - 1, 0, -1):
                    if p == pos:
                        continue
                    ov = row[p]
                    lo = row[p - 1] + 1
                    if lo >= ov:
                        continue
                    od = set()
                    for i in range(L):
                        if i < p:
                            od.add(row[p] - row[i])
                        elif i > p:
                            od.add(row[i] - row[p])
                    for nv in range(lo, ov):
                        if p < L - 1 and nv >= row[p + 1]:
                            break
                        nd = set()
                        for i in range(L):
                            if i < p:
                                nd.add(nv - row[i])
                            elif i > p:
                                nd.add(row[i] - nv)
                        ot = used_set - od
                        if nd & ot:
                            continue
                        for d in od:
                            used_set.discard(d)
                        for d in nd:
                            used_set.add(d)
                        row[p] = nv
                        break
                
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
        
        elif move_type < 0.7:
            # Increase a random element, then recompact
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            upper_limit = old_val + rng.randint(1, max(2, current_max // 3))
            new_val = rng.randint(old_val + 1, upper_limit)
            if new_val <= old_val:
                t *= cooling
                continue
            
            old_diffs = set()
            for i in range(L):
                if i < pos:
                    old_diffs.add(row[pos] - row[i])
                elif i > pos:
                    old_diffs.add(row[i] - row[pos])
            
            new_diffs = set()
            for i in range(L):
                if i < pos:
                    new_diffs.add(new_val - row[i])
                elif i > pos:
                    new_diffs.add(row[i] - new_val)
            
            other = used_set - old_diffs
            if not (new_diffs & other):
                for d in old_diffs:
                    used_set.discard(d)
                for d in new_diffs:
                    used_set.add(d)
                row[pos] = new_val
                
                # Recompact
                for p in range(L - 1, 0, -1):
                    ov = row[p]
                    lo = row[p - 1] + 1
                    if lo >= ov:
                        continue
                    od = set()
                    for i in range(L):
                        if i < p:
                            od.add(row[p] - row[i])
                        elif i > p:
                            od.add(row[i] - row[p])
                    for nv in range(lo, ov):
                        if p < L - 1 and nv >= row[p + 1]:
                            break
                        nd = set()
                        for i in range(L):
                            if i < p:
                                nd.add(nv - row[i])
                            elif i > p:
                                nd.add(row[i] - nv)
                        ot = used_set - od
                        if nd & ot:
                            continue
                        for d in od:
                            used_set.discard(d)
                        for d in nd:
                            used_set.add(d)
                        row[p] = nv
                        break
                
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
        else:
            # Full recompact
            for r_idx in range(n_rows):
                row = work[r_idx]
                for pos in range(L - 1, 0, -1):
                    ov = row[pos]
                    lo = row[pos - 1] + 1
                    if lo >= ov:
                        continue
                    od = set()
                    for i in range(L):
                        if i < pos:
                            od.add(row[pos] - row[i])
                        elif i > pos:
                            od.add(row[i] - row[pos])
                    for nv in range(lo, ov):
                        if pos < L - 1 and nv >= row[pos + 1]:
                            break
                        nd = set()
                        for i in range(L):
                            if i < pos:
                                nd.add(nv - row[i])
                            elif i > pos:
                                nd.add(row[i] - nv)
                        ot = used_set - od
                        if nd & ot:
                            continue
                        for d in od:
                            used_set.discard(d)
                        for d in nd:
                            used_set.add(d)
                        row[pos] = nv
                        break
            
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation_counter = 0
            else:
                stagnation_counter += 1
        
        if stagnation_counter > max_stagnation:
            fresh = _greedy_construct(n_rows, k, rng=rng)
            if _check_valid(fresh, k):
                fresh_max = _max_entry(fresh)
                if fresh_max < best_max:
                    best_max = fresh_max
                    best_rows = [list(r) for r in fresh]
                    current_max = best_max
                    work = [list(r) for r in fresh]
                    used_set = set()
                    for row in work:
                        for i in range(L):
                            for j in range(i + 1, L):
                                used_set.add(row[j] - row[i])
                    t = start_temp
                    stagnation_counter = 0
        
        t *= cooling
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    time_limit = start_time + 140
    
    # 1) Greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)
    
    # 2) Generate row patterns and solve MIS
    rng = random.Random(42)
    pattern_deadline = time_limit - 40
    
    # Start with a modest scope bound and increase if needed
    scope_bound = best_max
    while time.time() < pattern_deadline and scope_bound >= 100:
        try:
            patterns = _generate_row_patterns(k, scope_bound)
            if len(patterns) >= n_rows:
                result = _construct_from_patterns(patterns, n_rows, k, time_limit - 30)
                if result is not None and _check_valid(result, k):
                    rmax = _max_entry(result)
                    if rmax < best_max:
                        best_rows = result
                        best_max = rmax
            scope_bound -= 5
        except Exception:
            break
    
    # 3) Local compaction
    compacted = _local_compact(best_rows, n_rows, k, time_limit - 20)
    if _check_valid(compacted, k):
        cmax = _max_entry(compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax
    
    # 4) Simulated annealing
    if time.time() < time_limit - 3:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, time_limit - 3)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax
    
    # 5) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    
    return best_rows
