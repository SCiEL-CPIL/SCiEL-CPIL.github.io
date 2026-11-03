import time
import random
import copy
import sys

def construct(n_rows, k):
    row_len = k + 1
    n_diffs_per_row = row_len * (row_len - 1) // 2
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def row_diffs(row):
        """Compute set of all positive differences in a row."""
        d = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d.add(row[j] - ri)
        return d

    def greedy_row(used, max_val_hint=200):
        """Greedy construction of a single row avoiding 'used' diffs."""
        row = [0]
        diffs = set()
        x = 0
        while len(row) < row_len:
            x += 1
            if x > max_val_hint:
                break
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and not (new_diffs & diffs):
                row.append(x)
                diffs |= new_diffs
        if len(row) < row_len:
            return None
        return row, diffs

    def find_row_dfs(used, best_max_init, S_cap, node_budget, dl, rng):
        """
        Search for a row of length row_len starting at 0.
        Constraints:
        - All elements strictly increasing.
        - Max element < best_max_init.
        - All differences must not be in 'used'.
        - All differences must be unique within the row (implied by distinctness requirement, 
          but here we just ensure they don't clash with 'used' and each other implicitly 
          by checking 'new & rdiffs' in the recursion).
        
        Returns (row, diffs) or None.
        """
        best_row = [None]
        best_max = [best_max_init]
        nodes = [0]
        
        def dfs(row, rdiffs, last):
            if time.time() > dl or nodes[0] >= node_budget:
                return
            nodes[0] += 1
            
            if len(row) == row_len:
                m = row[-1]
                if m < best_max[0]:
                    best_max[0] = m
                    best_row[0] = row[:]
                return
            
            hi = min(best_max[0] - 1, S_cap)
            lo = last + 1
            if lo > hi:
                return
            
            # Randomize order slightly to help escape local minima in DFS? 
            # For now, standard order is better for finding *tightest* fit.
            # We want the smallest max, so we try small x first.
            for x in range(lo, hi + 1):
                new = {x - a for a in row}
                if new & used:
                    continue
                if new & rdiffs:
                    continue
                
                row.append(x)
                rdiffs |= new
                dfs(row, rdiffs, x)
                rdiffs -= new
                row.pop()
                
                if nodes[0] >= node_budget or time.time() > dl:
                    return
                if best_row[0] is not None and best_max[0] == lo:
                    # Can't do better than lo
                    return

        dfs([0], set(), 0)
        if best_row[0] is None:
            return None
        return best_row[0], row_diffs(best_row[0])

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def run_optimization(seed, dl):
        rng = random.Random(seed)
        
        # 1. Initial Construction
        used = set()
        rows = []
        for _ in range(n_rows):
            # Try DFS first for tightness, fallback to greedy
            # Use a generous budget for initial construction
            res = find_row_dfs(used, 200, 200, 50000, dl, rng)
            if res is None:
                res = greedy_row(used)
                if res is None:
                    # Greedy failed? Shouldn't happen for small n,k if limit is high
                    # Force a valid row by expanding limit
                    res = greedy_row(used, 500)
            if res:
                row, rd = res
                rows.append(row)
                used |= rd
            else:
                # Last resort: just add a greedy row with very high limit
                row, rd = greedy_row(used, 1000)
                rows.append(row)
                used |= rd
        
        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        
        # 2. Iterative Local Search: Row Replacement
        # We repeatedly pick a row, remove it, and try to rebuild it to be smaller
        # or to allow other rows to be smaller later.
        
        for it in range(1000):
            if time.time() > dl - 2:
                break
            
            # Order rows: prioritize those with large scope or random
            # Sorting by scope descending helps reduce the bottleneck first
            order = list(range(n_rows))
            order.sort(key=lambda i: (-rows[i][-1], rng.random()))
            
            improved = False
            
            # Rebuild 'used' set efficiently? 
            # For n_rows=7, recomputing is fast.
            used = set()
            for r in rows:
                used |= row_diffs(r)
            
            for idx in order:
                if time.time() > dl - 2:
                    break
                
                r = rows[idx]
                d = row_diffs(r)
                used -= d
                
                cur_max = r[-1]
                
                # Try to find a better (smaller max) row
                # Search space: max < cur_max
                # If cur_max is small, DFS is fast.
                # Budget for DFS depends on remaining time
                time_left = dl - time.time()
                node_budget = int(min(50000, time_left * 2000))
                if node_budget < 100:
                    break
                
                res = find_row_dfs(used, cur_max, cur_max - 1, node_budget, dl, rng)
                
                if res is not None:
                    nr, nd = res
                    rows[idx] = nr
                    used |= nd
                    improved = True
                else:
                    # If we couldn't find a strictly smaller row, 
                    # maybe we can find a *different* row of same size 
                    # that frees up differences for others?
                    # For now, stick to the found row or original.
                    # If res is None, we keep the old row.
                    used |= d
            
            if improved:
                s = scope_of(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = [r[:] for r in rows]
            
            # Random perturbation: occasionally rebuild a random row from scratch
            # to escape local minima where rows are interlocked.
            if it % 10 == 0 and time.time() < dl - 5:
                pi = rng.randrange(n_rows)
                d = row_diffs(rows[pi])
                used -= d
                # Try to rebuild with a slightly looser constraint to allow rearrangement
                # Use greedy first to ensure validity, then maybe optimize?
                # Actually, let's just use DFS with a high cap to see if we can find a 
                # structurally different row.
                res = find_row_dfs(used, 200, 200, 20000, dl, rng)
                if res:
                    nr, nd = res
                    rows[pi] = nr
                    used |= nd
                else:
                    nr, nd = greedy_row(used)
                    if nr:
                        rows[pi] = nr
                        used |= nd
                    else:
                        used |= d
                
                s = scope_of(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = [r[:] for r in rows]

        return best_scope, best_rows

    # Main Loop: Multiple Restarts
    overall_best_scope = float('inf')
    overall_best_rows = None
    
    # First run with a long budget to establish a strong baseline
    seed = 0
    first_dl = min(deadline - 5.0, start_time + 60.0)
    sc, rows = run_optimization(seed, first_dl)
    if sc < overall_best_scope:
        overall_best_scope = sc
        overall_best_rows = rows
    
    # Subsequent runs with shorter budgets to explore different local optima
    while time.time() < deadline - 3.0:
        seed += 1
        # Allocate remaining time divided by estimated remaining runs
        # Just use a fixed chunk for now, e.g., 15s per run
        dl = min(deadline - 1.0, time.time() + 15.0)
        sc, rows = run_optimization(seed, dl)
        if sc < overall_best_scope:
            overall_best_scope = sc
            overall_best_rows = rows
            
        # If we've hit the theoretical lower bound or very close, we can stop early?
        # Bound is 105. If we hit 105, stop.
        if overall_best_scope <= 105:
            break

    if overall_best_rows is None:
        # Fallback: Pure Greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            res = greedy_row(used)
            if res:
                r, d = res
                rows.append(r)
                used |= d
            else:
                # Should not happen
                r = [0, 1, 2, 3, 4, 5]
                rows.append(r)
        return rows

    return overall_best_rows

# Keep the original function name and signature as required
# The function above is the implementation.
# Ensure no global state persists between calls if construct is called multiple times
# (though typically it's called once per evaluation).

if __name__ == "__main__":
    # Quick test
    # rows = construct(7, 5)
    # print(rows)
    # print(scope_of(rows))
    pass
