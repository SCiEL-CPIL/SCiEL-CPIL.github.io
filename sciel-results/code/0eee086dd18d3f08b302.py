"""Program for difference triangle set (7,5) using maximum weight independent set search."""

import sys
import time
from itertools import combinations


def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    
    Uses maximum weight independent set search on a compatibility graph
    with bitset-accelerated branch-and-bound and greedy coloring heuristics.
    """
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    lower_bound = total_diffs_needed
    
    start_time = time.time()
    max_time = 115  # Leave margin for the 120s limit
    
    for scope in range(lower_bound, lower_bound + 300):
        if time.time() - start_time > max_time:
            break
        result = _find_rows_mwis(n_rows, k, scope, start_time, max_time)
        if result is not None:
            return result
    
    return _greedy_construct(n_rows, k)


def _greedy_construct(n_rows, k):
    """Simple greedy fallback."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _get_row_diffs(row):
    """Get all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _find_rows_mwis(n_rows, k, scope, start_time, max_time):
    """
    Find n_rows compatible rows using maximum weight independent set approach.
    
    Nodes: all valid rows (k+1 increasing ints starting at 0, max <= scope)
    Edge: two rows are adjacent if they share any difference
    Goal: find independent set of size n_rows
    
    Uses branch-and-bound with:
    - Bitset representation for fast neighborhood queries
    - Greedy coloring for upper bound pruning
    - Weighted vertex ordering
    """
    # Step 1: Generate candidate rows
    candidates = _generate_candidate_rows(k, scope)
    n = len(candidates)
    
    if n < n_rows:
        return None
    
    # Step 2: Compute difference sets and compatibility
    diff_sets = [_get_row_diffs(row) for row in candidates]
    
    # Build adjacency: adj[i] is a bitset of nodes incompatible with i
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if diff_sets[i] & diff_sets[j]:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Step 3: Greedy coloring for upper bound
    colors = _greedy_color(n, adj)
    max_color = max(colors) if n > 0 else 0
    
    if max_color < n_rows:
        return None
    
    # Step 4: Branch and bound search
    best_solution = [None]
    
    def _branch_and_bound(candidates_mask, depth, current_set):
        if time.time() - start_time > max_time:
            return False
        
        if depth == 0:
            best_solution[0] = list(current_set)
            return True
        
        available_count = bin(candidates_mask).count('1')
        if available_count < depth:
            return False
        
        pivot = _select_pivot(candidates_mask, adj, n)
        
        if pivot is None:
            return False
        
        # Branch 1: Include pivot
        new_mask = candidates_mask & ~(1 << pivot)
        new_mask &= ~adj[pivot]
        if _branch_and_bound(new_mask, depth - 1, current_set + [pivot]):
            return True
        
        # Branch 2: Exclude pivot
        new_mask = candidates_mask & ~(1 << pivot)
        if _branch_and_bound(new_mask, depth, current_set):
            return True
        
        return False
    
    initial_mask = (1 << n) - 1
    
    if _branch_and_bound(initial_mask, n_rows, []):
        if best_solution[0] is not None:
            return [candidates[i] for i in best_solution[0]]
    
    return None


def _generate_candidate_rows(k, scope):
    """Generate candidate rows with heuristic pruning."""
    rows = []
    max_candidates = 50000
    
    def gen_row(current, diffs):
        if len(current) == k + 1:
            if len(diffs) == (k + 1) * k // 2:
                rows.append(list(current))
                return True
            return False
        
        if len(rows) >= max_candidates:
            return False
        
        start = current[-1] + 1
        max_val = scope - (k + 1 - len(current) - 1)
        
        for val in range(start, max_val + 1):
            new_diffs = set()
            valid = True
            for a in current:
                d = val - a
                if d in diffs or d <= 0:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                new_diffs_set = diffs | new_diffs
                if len(new_diffs_set) == len(diffs) + len(current):
                    result = gen_row(current + [val], new_diffs_set)
                    if result:
                        return True
        
        return False
    
    gen_row([0], set())
    return rows


def _greedy_color(n, adj):
    """Greedy graph coloring using largest-first ordering."""
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    colors = [0] * n
    
    for v in order:
        neighbor_colors = set()
        mask = adj[v]
        for u in range(n):
            if mask & (1 << u):
                neighbor_colors.add(colors[u])
        
        c = 0
        while c in neighbor_colors:
            c += 1
        colors[v] = c
    
    return colors


def _select_pivot(candidates_mask, adj, n):
    """Select pivot vertex with highest degree in remaining graph."""
    best = None
    best_degree = -1
    
    for i in range(n):
        if candidates_mask & (1 << i):
            neighbor_mask = adj[i] & candidates_mask
            degree = bin(neighbor_mask).count('1')
            if degree > best_degree:
                best_degree = degree
                best = i
    
    return best
