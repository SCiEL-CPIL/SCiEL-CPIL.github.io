import time
import random
import math
import sys
import heapq
from itertools import combinations

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m

def _compute_all_diffs(rows):
    """Return (global_mask, row_masks) for all rows."""
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks

def _generate_valid_rows(num_elements, M, rng, max_rows=5000):
    """
    Generate up to max_rows valid rows (strictly increasing, starting at 0,
    max entry <= M) with internally distinct differences.
    Returns list of (row_tuple, diff_mask).
    """
    rows = []
    seen_masks = set()
    attempts = 0
    max_attempts = max_rows * 50
    
    while len(rows) < max_rows and attempts < max_attempts:
        attempts += 1
        # Random row generation
        row = [0]
        current_max = 0
        for _ in range(num_elements - 1):
            # Pick next element
            min_next = row[-1] + 1
            if min_next > M:
                break
            max_next = M
            # Bias towards larger gaps for later elements
            if len(row) < num_elements // 2:
                max_next = min(M, row[-1] + max(1, (M - row[-1]) // 2))
            val = rng.randint(min_next, max_next)
            row.append(val)
        
        if len(row) != num_elements:
            continue
        
        # Check internal distinctness
        diffs = set()
        n = len(row)
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if not valid:
                break
        
        if not valid:
            continue
        
        mask = 0
        for d in diffs:
            mask |= (1 << d)
        
        if mask not in seen_masks:
            seen_masks.add(mask)
            rows.append((tuple(row), mask))
    
    return rows

def _max_clique_branch_and_bound(nodes, n_clique, time_limit, rng):
    """
    Maximum clique search using branch and bound.
    nodes: list of (row_tuple, mask)
    n_clique: target clique size
    Returns list of n_clique row_tuples if found, else None.
    """
    start = time.time()
    N = len(nodes)
    if N < n_clique:
        return None
    
    # Build adjacency: two nodes are compatible if their masks don't overlap
    # For efficiency, we'll compute on the fly with bit operations
    
    # Order nodes by some heuristic - maybe by number of compatible neighbors
    # For now, just use the given order
    
    clique = []
    candidates = list(range(N))
    
    def dfs(candidates, current_clique, depth):
        if time.time() > time_limit:
            return False
        
        if depth == n_clique:
            current_clique.extend(clique)
            return True
        
        if len(candidates) < n_clique - depth:
            return False
        
        # Bound: if we can't reach target size, prune
        # Simple bound: need at least (n_clique - depth) candidates
        
        # Pick a pivot node for branching
        # Try each candidate
        for i in range(len(candidates)):
            if time.time() > time_limit:
                return False
            
            node_idx = candidates[i]
            node_mask = nodes[node_idx][1]
            
            # Check if compatible with all in current_clique
            compatible = True
            for c_idx in current_clique:
                if nodes[c_idx][1] & node_mask:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # Add to clique
            current_clique.append(node_idx)
            
            # Filter candidates: must be compatible with new node and come after
            new_candidates = []
            for j in range(i + 1, len(candidates)):
                other_idx = candidates[j]
                if not (nodes[other_idx][1] & node_mask):
                    # Also check against all in current_clique
                    ok = True
                    for c_idx in current_clique[:-1]:  # exclude the one we just added
                        if nodes[other_idx][1] & nodes[c_idx][1]:
                            ok = False
                            break
                    if ok:
                        new_candidates.append(other_idx)
            
            if dfs(new_candidates, current_clique, depth + 1):
                return True
            
            current_clique.pop()
            
            # If we've tried enough, return
            if len(current_clique) == 0 and len(clique) > 0:
                return False
        
        return False
    
    # Try multiple times with different orderings
    for attempt in range(3):
        if time.time() > time_limit:
            break
        
        if attempt == 0:
            order = list(range(N))
        elif attempt == 1:
            order = list(range(N))
            rng.shuffle(order)
        else:
            # Sort by max difference (higher first) - might be more constrained
            order = sorted(range(N), key=lambda i: nodes[i][0][-1], reverse=True)
        
        # Remap nodes according to order
        ordered_nodes = [nodes[i] for i in order]
        
        # Reset
        clique = []
        
        # Use iterative approach with explicit stack for better control
        # But for simplicity, use recursion with increased limit
        sys.setrecursionlimit(10000)
        
        result = dfs(list(range(len(ordered_nodes))), [], 0)
        
        if result:
            # Convert back to original indices
            result_indices = [order[i] for i in clique]
            return [nodes[i][0] for i in result_indices]
    
    return None

def _greedy_construct(n_rows, num_elements, rng, target_scope):
    """
    Greedy construction: build rows one by one, each row using the earliest
    feasible values that don't conflict with previously used differences.
    """
    rows = []
    used_diffs = set()
    
    for _ in range(n_rows):
        row = [0]
        found = False
        
        for val in range(1, target_scope + 1):
            # Check if we can add val to current row
            local_diffs = set()
            valid = True
            for prev in row:
                d = val - prev
                if d in used_diffs or d in local_diffs:
                    valid = False
                    break
                local_diffs.add(d)
            
            if valid:
                row.append(val)
                if len(row) == num_elements:
                    found = True
                    break
        
        if not found:
            return None
        
        used_diffs.update(val - prev for prev in row for val in [row[-1]])
        # More careful update
        row_diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                row_diffs.add(row[j] - row[i])
        used_diffs.update(row_diffs)
        
        rows.append(row)
    
    return rows

def _construct_fallback(n_rows, num_elements):
    """
    Simple fallback: construct rows with increasing offsets to guarantee validity.
    """
    rows = []
    used_diffs = set()
    
    for r in range(n_rows):
        row = [0]
        base = r * 1000  # large offset to avoid conflicts
        x = 0
        while len(row) < num_elements:
            x += 1
            val = base + x
            new_diffs = {val - a for a in row}
            if not (new_diffs & used_diffs):
                row.append(val)
                used_diffs.update(new_diffs)
        rows.append(row)
    
    return rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 135.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try greedy construction with various scopes
    for scope in range(counting_bound, counting_bound + 50):
        if time.time() > DEADLINE - 60:
            break
        
        rows = _greedy_construct(n_rows, num_elements, rng, scope)
        if rows is not None and _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
    
    # Phase 2: Maximum clique search
    if time.time() < DEADLINE - 30:
        # Generate candidate rows for various M values
        for M in range(counting_bound, min(counting_bound + 30, 200)):
            if time.time() > DEADLINE - 20:
                break
            
            candidates = _generate_valid_rows(num_elements, M, rng, max_rows=2000)
            
            if len(candidates) < n_rows:
                continue
            
            # Try to find clique of size n_rows
            time_limit = time.time() + 5.0
            clique = _max_clique_branch_and_bound(candidates, n_rows, time_limit, rng)
            
            if clique is not None:
                # Convert tuples to lists
                rows = [list(row) for row in clique]
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
    
    # Fallback if no solution found
    if best_rows is None:
        best_rows = _construct_fallback(n_rows, num_elements)
        best_scope = _scope(best_rows)
    
    # Phase 3: Local search refinement
    if time.time() < DEADLINE - 10:
        # Simple local search: try to reduce scope by moving elements down
        current_rows = [r[:] for r in best_rows]
        current_scope = _scope(current_rows)
        global_mask, row_masks = _compute_all_diffs(current_rows)
        
        max_iter = 50000
        for it in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, num_elements - 1)
            row = current_rows[row_idx]
            old_val = row[pos]
            
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < num_elements - 1 else current_scope
            
            if low >= old_val:
                continue
            
            new_val = rng.randint(low, old_val - 1)
            
            # Try the move
            row[pos] = new_val
            
            # Recompute row mask
            diffs_in_row = set()
            n = len(row)
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in diffs_in_row:
                        valid = False
                        break
                    diffs_in_row.add(d)
            
            if not valid:
                row[pos] = old_val
                continue
            
            new_row_mask = 0
            for d in diffs_in_row:
                new_row_mask |= (1 << d)
            
            other_mask = global_mask & ~row_masks[row_idx]
            if new_row_mask & other_mask:
                row[pos] = old_val
                continue
            
            # Accept the move
            row_masks[row_idx] = new_row_mask
            global_mask = (global_mask & ~row_masks[row_idx]) | new_row_mask
            current_scope = _scope(current_rows)
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        return _construct_fallback(n_rows, num_elements)
