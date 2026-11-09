import time
import random
import math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 145.0  # 5s safety margin for 150s limit

    def remaining():
        return DEADLINE - (time.time() - t0)

    needed = k * (k + 1) // 2  # 28 differences for k=7

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def diff_mask(row):
        """Return an integer bitmask representing the set of differences."""
        mask = 0
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                mask |= (1 << d)
        return mask

    def scope_of(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for r in rows:
            if r[0] != 0 or any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            d = row_diffs(r)
            if len(d) != needed:
                return False
            if d & used:
                return False
            used |= d
        return True

    # Greedy baseline to get a valid start
    def greedy():
        used = set()
        rows = []
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

    # Generate a candidate row using randomized greedy
    def gen_row(rng, max_val_hint=None):
        row = [0]
        x = 0
        # Try to keep values small by scanning from bottom up, but with randomness
        while len(row) < k + 1:
            # Determine a search range
            # Bias towards smaller numbers but allow jumps
            start = x + 1
            end = max_val_hint if max_val_hint else start + 20
            # If we are close to the end of the hint, expand
            if end <= start:
                end = start + 20
            
            found = False
            # Try a few random candidates in the range
            for _ in range(10):
                y = rng.randint(start, end)
                new = {y - a for a in row}
                # Check internal distinctness (guaranteed by strict increase? No, diffs must be distinct)
                if len(new) != len(row): # If len(row) is m, new has m diffs. If any overlap, len < m.
                    continue
                # Check against used? We don't track global used here, just internal.
                # The global conflict check happens later in the MIS search.
                # However, to make rows "good", we should avoid common diffs.
                row.append(y)
                x = y
                found = True
                break
            
            if not found:
                # Fallback: find the smallest valid y
                y = start
                while True:
                    new = {y - a for a in row}
                    if len(new) == len(row):
                        row.append(y)
                        x = y
                        break
                    y += 1
        return row

    # Backtracking search for an independent set of size `target_size`
    # pool: list of (row, mask, max_entry)
    # We want to minimize the max entry of the selected rows.
    def find_independent_set(pool, target_size, current_best_scope):
        # Sort pool by max_entry to prioritize small scopes
        pool_sorted = sorted(pool, key=lambda x: x[2])
        
        # Prune: if the smallest possible max entry in the pool is >= current_best_scope, 
        # we might still find a set, but we only care about improving it.
        # Actually, we just want ANY set with max_entry < current_best_scope.
        
        n = len(pool_sorted)
        if n < target_size:
            return None

        selected = []
        used_mask = 0
        
        def dfs(idx, depth):
            if depth == target_size:
                return selected[:]
            
            # Pruning: not enough candidates left
            if n - idx < target_size - depth:
                return None
            
            # Heuristic: if the max entry of the current candidate is already >= current_best_scope,
            # and we haven't found a better one, we can skip? 
            # No, we need to check if we can complete the set with smaller entries.
            # But since we sorted by max_entry, if pool_sorted[idx][2] >= current_best_scope,
            # all subsequent ones are also >= current_best_scope.
            # If we just want to find a set with scope < current_best_scope, we can stop.
            if pool_sorted[idx][2] >= current_best_scope:
                return None

            for i in range(idx, n):
                row, mask, max_e = pool_sorted[i]
                
                # If this row's max entry is too big, break (since sorted)
                if max_e >= current_best_scope:
                    break

                # Check conflict
                if mask & used_mask:
                    continue
                
                # Try to select this row
                selected.append(row)
                new_used = used_mask | mask
                res = dfs(i + 1, depth + 1)
                if res:
                    return res
                selected.pop()
                
                # Optional: If we fail, we can try skipping. The loop handles skipping.
            
            return None

        return dfs(0, 0)

    # Main logic
    best_rows = greedy()
    best_scope = scope_of(best_rows)
    
    rng = random.Random(42)
    
    # Phase 1: Generate a large pool of diverse candidates
    # We want candidates with small max entries.
    # The greedy baseline gives us a scope S. Let's aim for candidates with max < S.
    
    while remaining() > 10:
        pool = []
        # Generate candidates
        # We vary the "hint" to encourage different structures
        for i in range(2000):
            # Randomize the strategy slightly
            bias = rng.uniform(0.5, 1.5)
            # Generate row
            row = gen_row(rng)
            # Filter: only keep rows with max entry < best_scope
            if max(row) < best_scope:
                mask = diff_mask(row)
                # Ensure internal diffs are distinct (gen_row should guarantee this, but double check)
                if len(row_diffs(row)) == needed:
                    pool.append((row, mask, max(row)))
        
        if not pool:
            continue
            
        # Search for independent set
        # We want to find 4 rows.
        # To speed up, we can limit the pool size or use a heuristic ordering.
        # The DFS above sorts by max_entry.
        
        # Optimization: If pool is very large, DFS might be slow.
        # Let's limit the pool to the best 500 candidates by scope.
        pool_sorted = sorted(pool, key=lambda x: x[2])
        pool_limited = pool_sorted[:1000]
        
        found = find_independent_set(pool_limited, n_rows, best_scope)
        
        if found:
            new_scope = scope_of(found)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = found
                # We found a better scope, so we can restart with a tighter bound
                # The loop condition `max(row) < best_scope` will now be stricter.
                # To avoid getting stuck, we can randomize the generator more.
                continue
        else:
            # If we didn't find a set, maybe the pool wasn't diverse enough.
            # We can try a different random seed or generation strategy.
            pass

    # Final safety check
    if not is_valid(best_rows):
        best_rows = greedy()
        
    return best_rows
