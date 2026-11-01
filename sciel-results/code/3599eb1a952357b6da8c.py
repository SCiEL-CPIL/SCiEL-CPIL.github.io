import time
import random

def construct(n_rows, k):
    deadline = time.time() + 140
    n = k + 1
    rng = random.Random(42)
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(n) for j in range(i + 1, n)}
        
    def is_valid(rows):
        used = set()
        for row in rows:
            d = get_diffs(row)
            if used & d:
                return False
            used |= d
        return True

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    rows = greedy_init()
    best_rows = [list(r) for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    def optimize_row_deterministic(forbidden, max_scope_limit):
        def dfs(idx, current_row):
            if time.time() > deadline:
                return None
            if idx == n:
                return current_row
            
            start = current_row[-1] + 1
            if start + (n - idx - 1) > max_scope_limit:
                return None
            
            for v in range(start, max_scope_limit + 1):
                new_diffs = {v - a for a in current_row}
                if not (new_diffs & forbidden):
                    res = dfs(idx + 1, current_row + [v])
                    if res is not None:
                        return res
            return None
            
        return dfs(1, [0])
        
    def optimize_row_random(forbidden, max_scope_limit):
        def dfs(idx, current_row):
            if time.time() > deadline:
                return None
            if idx == n:
                return current_row
            
            start = current_row[-1] + 1
            if start + (n - idx - 1) > max_scope_limit:
                return None
            
            candidates = list(range(start, max_scope_limit + 1))
            rng.shuffle(candidates)
            
            for v in candidates:
                new_diffs = {v - a for a in current_row}
                if not (new_diffs & forbidden):
                    res = dfs(idx + 1, current_row + [v])
                    if res is not None:
                        return res
            return None
            
        return dfs(1, [0])

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        ri = rng.randint(0, n_rows - 1)
        
        forbidden = set()
        for i in range(n_rows):
            if i != ri:
                forbidden |= get_diffs(rows[i])
                
        # 1. Try to improve global scope
        if best_scope > n:
            new_row = optimize_row_deterministic(forbidden, best_scope - 1)
            if new_row:
                rows[ri] = new_row
                new_scope = new_row[-1]
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
        else:
            # 2. Try to diversify if scope is already minimal (or close)
            # Use random search to find a different valid row
            # This helps escape local minima where deterministic search fails
            # because it always picks the same "greedy" path.
            new_row = optimize_row_random(forbidden, best_scope)
            if new_row and new_row != rows[ri]:
                rows[ri] = new_row

        # 3. Local improvement: try to find a row with scope <= current row's scope
        # This is useful if we haven't improved global scope yet.
        current_scope = rows[ri][-1]
        if current_scope > n:
            # Only do this if we didn't just replace it with a better one
            # Or always? It's cheap if it fails fast.
            # But it might overwrite a "kick" if we do it immediately.
            # Let's do it every 5th iteration to balance.
            if attempts % 5 == 0:
                new_row = optimize_row_deterministic(forbidden, current_scope)
                if new_row and new_row != rows[ri]:
                    rows[ri] = new_row
                    # This doesn't update best_scope, but changes the landscape.
                    pass

    return best_rows
