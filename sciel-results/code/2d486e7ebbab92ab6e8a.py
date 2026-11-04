import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 35s margin before 150s kill
    
    # For the specific case n_rows=7, k=5, we know the answer is around 112
    # Let's use a combination of strategies
    
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Enumerate all valid rows up to a bound, then select best combination
    # This is feasible for k=5 (6 elements per row)
    
    # First, let's try a smart greedy with local search
    rows = _smart_greedy(n_rows, k, deadline)
    if rows is not None:
        scope = max(row[-1] for row in rows)
        best_rows = rows
        best_scope = scope
    
    # Now try local search to improve
    if time.time() < deadline:
        rows = _local_search(rows, n_rows, k, deadline)
        if rows is not None:
            scope = max(row[-1] for row in rows)
            if scope < best_scope:
                best_rows = rows
                best_scope = scope
    
    # Strategy 2: Try different random seeds for greedy
    seed = 1
    while time.time() < deadline and seed < 200:
        rows = _smart_greedy(n_rows, k, deadline, seed)
        if rows is not None:
            scope = max(row[-1] for row in rows)
            if scope < best_scope:
                best_rows = rows
                best_scope = scope
                # Try local search on this improved solution
                if time.time() < deadline:
                    rows2 = _local_search(rows, n_rows, k, deadline)
                    if rows2 is not None:
                        scope2 = max(row[-1] for row in rows2)
                        if scope2 < best_scope:
                            best_rows = rows2
                            best_scope = scope2
        seed += 1
    
    # Strategy 3: For the specific case, try to find rows that pack tightly
    # by using a backtracking approach for each row
    if time.time() < deadline:
        rows = _backtracking_construct(n_rows, k, deadline)
        if rows is not None:
            scope = max(row[-1] for row in rows)
            if scope < best_scope:
                best_rows = rows
                best_scope = scope
                if time.time() < deadline:
                    rows2 = _local_search(rows, n_rows, k, deadline)
                    if rows2 is not None:
                        scope2 = max(row[-1] for row in rows2)
                        if scope2 < best_scope:
                            best_rows = rows2
                            best_scope = scope2
    
    if best_rows is None:
        # Fallback: simple greedy
        best_rows = _simple_greedy(n_rows, k)
    
    return best_rows


def _row_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _valid(rows):
    """Check if rows are valid."""
    all_diffs = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_diffs:
            return False
        all_diffs |= rd
    return True


def _simple_greedy(n_rows, k):
    """Simple greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            used.update(new_diffs)
        rows.append(row)
    return rows


def _smart_greedy(n_rows, k, deadline, seed=0):
    """Greedy with weighted candidate selection."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        row = [0]
        for pos in range(k):
            # Collect valid candidates
            valid_xs = []
            last_x = row[-1]
            # Limit search to reasonable range
            max_search = 200
            x = last_x + 1
            while x <= last_x + max_search and len(valid_xs) < 50:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                # Expand search
                x = last_x + 1
                while len(valid_xs) < 50:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            # Weighted selection: prefer smaller values but with some randomness
            # Use exponential weighting to favor smaller values
            weights = []
            min_x = min(valid_xs)
            for x in valid_xs:
                w = 1.0 / (1 + (x - min_x) * 0.05)
                weights.append(w)
            
            # Normalize
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            
            # Select
            r = rng.random()
            cum = 0
            chosen = valid_xs[0]
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen = valid_xs[i]
                    break
            
            row.append(chosen)
            new_diffs = {chosen - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows


def _local_search(rows, n_rows, k, deadline):
    """Local search to improve the solution."""
    if rows is None:
        return None
    
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    
    current_scope = max(row[-1] for row in rows)
    best_rows = [list(r) for r in rows]
    best_scope = current_scope
    
    rng = random.Random(42)
    
    # Try to improve by replacing rows
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        # Try replacing a random row
        row_idx = rng.randint(0, n_rows - 1)
        other_rows = [r for i, r in enumerate(rows) if i != row_idx]
        other_diffs = set()
        for r in other_rows:
            other_diffs |= _row_diffs(r)
        
        # Find a better row
        new_row = _find_better_row(other_diffs, k, deadline)
        if new_row is not None:
            old_scope = max(row[-1] for row in rows)
            rows[row_idx] = new_row
            new_scope = max(row[-1] for row in rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
            # Keep the improvement even if not better than best
            # because it might lead to further improvements
            used = set()
            for row in rows:
                used |= _row_diffs(row)
    
    return best_rows


def _find_better_row(used_diffs, k, deadline):
    """Find a row that avoids used_diffs and has small max element."""
    # Try to find a row with small max element
    for max_elem in range(k + 1, 200):
        if time.time() > deadline:
            return None
        # Try to find a valid row ending at most at max_elem
        row = _search_single_row(used_diffs, k, max_elem)
        if row is not None:
            return row
    return None


def _search_single_row(used_diffs, k, max_elem):
    """Search for a single row of k+1 elements starting at 0, max <= max_elem, avoiding used_diffs."""
    # Backtracking
    def backtrack(row, used):
        if len(row) == k + 1:
            return row[:]
        
        last = row[-1]
        for x in range(last + 1, max_elem + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            if new_diffs & used_diffs:
                continue
            result = backtrack(row + [x], used | new_diffs)
            if result is not None:
                return result
        return None
    
    return backtrack([0], set())


def _backtracking_construct(n_rows, k, deadline):
    """Construct rows using backtracking for each row."""
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        # Find a row that avoids used diffs
        row = _find_better_row(used, k, deadline)
        if row is None:
            return None
        
        rows.append(row)
        used |= _row_diffs(row)
    
    return rows
