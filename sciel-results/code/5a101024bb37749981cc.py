"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization and seed offset."""
    rng = random.Random(seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                max_step = max(1, int((k + 1) * 2.0))
                found = False
                for _attempt in range(30):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows


def _compress_rows(rows, k, deadline):
    """Aggressively compress rows by trying to reduce each element.
    Uses incremental difference tracking for speed."""
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    
    # Build global used set
    used = set()
    for r in best:
        used |= _row_diffs(r)
    
    n_rows = len(best)
    k1 = k + 1
    
    while time.time() < deadline:
        improved = False
        
        # Sort positions by current value descending to target largest first
        positions = []
        for ri in range(n_rows):
            for pi in range(1, k1):
                positions.append((best[ri][pi], ri, pi))
        positions.sort(reverse=True)
        
        for val, ri, pi in positions:
            if time.time() > deadline:
                break
            
            row = best[ri]
            old_val = row[pi]
            if old_val <= 1:
                continue
            
            lower = row[pi - 1] + 1
            upper = old_val - 1
            
            if lower > upper:
                continue
            
            # Try to find a smaller value that works
            found_new = None
            
            for new_val in range(lower, upper + 1):
                new_diffs = set()
                for i in range(pi):
                    new_diffs.add(new_val - row[i])
                for i in range(pi + 1, k1):
                    new_diffs.add(row[i] - new_val)
                
                old_diffs = set()
                for i in range(pi):
                    old_diffs.add(old_val - row[i])
                for i in range(pi + 1, k1):
                    old_diffs.add(row[i] - old_val)
                
                remaining = used - old_diffs
                if new_diffs & remaining:
                    continue
                
                found_new = new_val
                break
            
            if found_new is not None:
                row[pi] = found_new
                
                old_diffs = set()
                for i in range(pi):
                    old_diffs.add(old_val - row[i])
                for i in range(pi + 1, k1):
                    old_diffs.add(row[i] - old_val)
                
                new_diffs = set()
                for i in range(pi):
                    new_diffs.add(found_new - row[i])
                for i in range(pi + 1, k1):
                    new_diffs.add(row[i] - found_new)
                
                used -= old_diffs
                used |= new_diffs
                
                new_scope = _scope(best)
                if new_scope < best_scope:
                    best_scope = new_scope
                    improved = True
                improved = True
        
        if not improved:
            if random.random() < 0.3:
                ri = random.randint(0, n_rows - 1)
                pi = random.randint(1, k1 - 2)
                saved = best[ri][pi]
                delta = random.randint(1, 5)
                best[ri][pi] = saved + delta
                if _valid(best):
                    used = set()
                    for r in best:
                        used |= _row_diffs(r)
                else:
                    best[ri][pi] = saved
            else:
                break
    
    return best, best_scope


def _backtrack_construct(n_rows, k, deadline):
    """
    Constraint satisfaction approach: build rows one at a time using backtracking.
    For each row, use a "minimum remaining values" heuristic to pick the next
    element that leaves the most flexibility for future elements.
    """
    used = set()
    rows = []
    
    def _build_row(row, deadline):
        """Build a single row given the current used set and partial row."""
        if len(row) == k + 1:
            return True
        
        if time.time() > deadline:
            return False
        
        current_max = row[-1]
        # Try candidate values in order of increasing "constraint cost"
        # We want values that create differences that are "rare" or leave more room
        candidates = []
        # We'll try a bounded range to keep it tractable
        # Start from current_max + 1 up to some reasonable bound
        # The bound can be adaptive based on how many rows are left
        rows_remaining = n_rows - len(rows)
        # Heuristic bound: we need space for remaining elements
        max_search = current_max + 50 + rows_remaining * 10
        
        for x in range(current_max + 1, max_search + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            # Score: prefer values that create "spread out" differences
            # and leave more room. Simple heuristic: smaller x is generally better
            # but we also want to avoid clustering.
            candidates.append((x, new_diffs))
        
        # Shuffle candidates slightly to add randomness, but keep smaller ones preferred
        random.shuffle(candidates)
        # Sort by value to prefer smaller, but with some randomness
        candidates.sort(key=lambda c: c[0] + random.random() * 5)
        
        for x, new_diffs in candidates:
            row.append(x)
            used.update(new_diffs)
            
            if _build_row(row, deadline):
                return True
            
            row.pop()
            used.difference_update(new_diffs)
        
        return False
    
    for i in range(n_rows):
        if time.time() > deadline:
            return None
        row = [0]
        if _build_row(row, deadline):
            rows.append(row)
        else:
            return None
    
    return rows


def _local_search(rows, k, deadline):
    """
    Local search with simulated annealing-like acceptance.
    Try to move elements to reduce the maximum value.
    """
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    used = set()
    for r in best:
        used |= _row_diffs(r)
    
    n_rows = len(best)
    k1 = k + 1
    temp = 100.0
    
    while time.time() < deadline and temp > 0.1:
        improved = False
        
        # Pick a random row and position
        ri = random.randint(0, n_rows - 1)
        pi = random.randint(1, k1 - 1)
        
        row = best[ri]
        old_val = row[pi]
        
        # Try a random new value
        # Bias towards smaller values
        if random.random() < 0.7:
            # Try to decrease
            lower = row[pi - 1] + 1
            upper = old_val - 1
            if lower > upper:
                continue
            new_val = random.randint(lower, upper)
        else:
            # Try a small increase (for perturbation)
            new_val = old_val + random.randint(1, 10)
        
        if new_val == old_val:
            continue
        
        # Check if new_val maintains strict increase
        if pi > 0 and new_val <= row[pi - 1]:
            continue
        if pi < k1 - 1 and new_val >= row[pi + 1]:
            continue
        
        # Compute diff changes
        old_diffs = set()
        for i in range(pi):
            old_diffs.add(old_val - row[i])
        for i in range(pi + 1, k1):
            old_diffs.add(row[i] - old_val)
        
        new_diffs = set()
        for i in range(pi):
            new_diffs.add(new_val - row[i])
        for i in range(pi + 1, k1):
            new_diffs.add(row[i] - new_val)
        
        remaining = used - old_diffs
        if new_diffs & remaining:
            continue
        
        # Accept the move
        row[pi] = new_val
        used -= old_diffs
        used |= new_diffs
        
        new_scope = _scope(best)
        if new_scope < best_scope:
            best_scope = new_scope
            improved = True
        else:
            # Accept with some probability (simulated annealing)
            if random.random() < 0.1:
                improved = True
            else:
                # Reject
                row[pi] = old_val
                used -= new_diffs
                used |= old_diffs
        
        temp *= 0.995
    
    return best, best_scope


def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy constructions
    for seed in range(30):
        if time.time() > deadline - 20.0:
            break
        rows = _greedy_construct(n_rows, k, randomize=(seed > 0), seed_offset=seed * 1000 + 42)
        sc = _scope(rows)
        if sc < best_scope:
            best_rows = [list(r) for r in rows]
            best_scope = sc
    
    # Phase 2: Backtracking constructions
    for seed in range(20):
        if time.time() > deadline - 20.0:
            break
        random.seed(seed * 7919 + 13)
        rows = _backtrack_construct(n_rows, k, deadline - 20.0)
        if rows:
            sc = _scope(rows)
            if sc < best_scope:
                best_rows = [list(r) for r in rows]
                best_scope = sc
    
    # Phase 3: Local search on best solution
    if time.time() < deadline - 10.0:
        comp_rows, comp_scope = _local_search(best_rows, k, deadline - 10.0)
        if comp_scope < best_scope:
            best_rows = comp_rows
            best_scope = comp_scope
    
    # Phase 4: More greedy + local search
    attempts = 0
    while time.time() < deadline - 15.0 and attempts < 40:
        attempts += 1
        seed = attempts * 7919 + 13
        random.seed(seed)
        rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=seed)
        sc = _scope(rows)
        
        if time.time() < deadline - 15.0:
            comp_rows, comp_scope = _local_search(rows, k, deadline - 15.0)
            if comp_scope < best_scope:
                best_rows = comp_rows
                best_scope = comp_scope
    
    # Phase 5: Final compression
    if time.time() < deadline - 3.0:
        comp_rows, comp_scope = _compress_rows(best_rows, k, deadline - 1.0)
        if comp_scope < best_scope:
            best_rows = comp_rows
            best_scope = comp_scope
    
    return best_rows

# EVOLVE-BLOCK-END
