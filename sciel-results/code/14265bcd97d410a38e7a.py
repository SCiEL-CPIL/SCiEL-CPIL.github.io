import time
import random
import math
from collections import Counter, defaultdict

_BEST = None
_DEADLINE = 0.0

def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        # Check strictly increasing
        for i in range(1, len(r)):
            if r[i] <= r[i-1]:
                return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _generate_configs(k, max_scope, rng, count):
    """
    Generate random valid row configurations (subsets of integers starting at 0).
    Returns a list of (row_tuple, diff_set).
    We limit the scope of individual rows to max_scope to keep the search space manageable.
    """
    configs = []
    seen_diff_sets = set() # To avoid duplicate configs in the pool
    
    attempts = 0
    max_attempts = count * 100
    
    while len(configs) < count and attempts < max_attempts:
        attempts += 1
        # Randomly generate a row
        # To ensure we get diverse configs, we can use a randomized greedy approach
        # or just random sampling with validation.
        # Given k is small (5), we can be slightly more structured.
        
        row = [0]
        x = 0
        valid = True
        local_diffs = set()
        
        for _ in range(k):
            # Pick next element randomly in a range that allows completion
            # Heuristic: pick from [x+1, max_scope - (k - len(row))]
            min_next = x + 1
            max_next = max_scope - (k - len(row))
            
            if min_next > max_next:
                valid = False
                break
                
            # Bias towards smaller numbers to keep scope low, but allow some spread
            # Use a distribution that favors lower values but samples the range
            # Simple approach: pick random int, but ensure it's not too large to block future steps
            # Actually, just picking random in range is fine for diversity, 
            # but we must check for internal diff conflicts.
            
            # To be faster, we can try a few random candidates
            found = False
            # Try a few random candidates
            for _ in range(5):
                t = rng.randint(min_next, max_next)
                # Check internal conflicts
                conflict = False
                new_diffs = []
                for a in row:
                    d = t - a
                    if d in local_diffs:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if not conflict:
                    row.append(t)
                    local_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
            
            if not found:
                # If random failed, try sequential from min_next
                for t in range(min_next, max_next + 1):
                    conflict = False
                    new_diffs = []
                    for a in row:
                        d = t - a
                        if d in local_diffs:
                            conflict = True
                            break
                        new_diffs.append(d)
                    if not conflict:
                        row.append(t)
                        local_diffs.update(new_diffs)
                        x = t
                        found = True
                        break
                
                if not found:
                    valid = False
                    break
        
        if valid and len(row) == k + 1:
            diff_tuple = tuple(sorted(local_diffs))
            if diff_tuple not in seen_diff_sets:
                seen_diff_sets.add(diff_tuple)
                configs.append((tuple(row), frozenset(local_diffs)))
                
    return configs

def _greedy_packing(configs, n_rows, rng):
    """
    Greedily select n_rows non-conflicting configurations.
    Returns list of rows.
    """
    selected = []
    used_diffs = set()
    
    # Shuffle configs to randomize selection
    shuffled = list(configs)
    rng.shuffle(shuffled)
    
    # Sort by scope (largest element) to prefer compact rows? 
    # Or just random. Let's try sorting by scope ascending to see if it helps.
    # Actually, for packing, smaller rows might be "harder" to pack if they use common small diffs.
    # Let's just iterate.
    
    for row, diffs in shuffled:
        if len(selected) == n_rows:
            break
        if not (used_diffs & diffs):
            selected.append(list(row))
            used_diffs.update(diffs)
            
    if len(selected) < n_rows:
        return None
        
    return selected

def _local_search(rows, k, n_rows, rng, deadline, configs_pool):
    """
    Local search: Swap rows with other configs from the pool to reduce scope.
    """
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    # Precompute diff sets for current rows
    current_diff_sets = []
    for r in current_rows:
        ds = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                ds.add(r[j]-r[i])
        current_diff_sets.append(frozenset(ds))
        
    # Index configs by their scope to allow targeted swaps
    # We want to swap a high-scope row with a low-scope row that fits.
    
    # For efficiency, we won't rebuild the whole pool every step.
    # We will randomly pick a row to replace, and try to find a config in the pool
    # that fits the current used diffs (excluding the row being replaced).
    
    # Optimization: Pre-filter pool? No, pool is static.
    # We need to check compatibility quickly.
    
    # Let's maintain a list of (row, diffs, scope) for the pool
    pool_data = [(r, d, max(r)) for r, d in configs_pool]
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        if iters % 1000 == 0 and time.time() > deadline:
            break
            
        # Pick a row to replace
        i = rng.randint(0, n_rows - 1)
        old_row = current_rows[i]
        old_diffs = current_diff_sets[i]
        old_scope = max(old_row)
        
        # Current scope is dominated by this row if it's the max?
        # Let's just try to replace it with something that has lower scope 
        # or simply fits better to allow other rows to shrink later.
        
        # Calculate used diffs without row i
        # This is O(N * K^2) which is fine for small N.
        used_without_i = set()
        for j in range(n_rows):
            if j != i:
                used_without_i.update(current_diff_sets[j])
                
        # Try to find a replacement from pool
        # We prefer configs with small scope.
        # Let's sample a few candidates from the pool.
        # To be smart, we can filter pool by scope < old_scope + margin
        
        candidates = []
        # Sample 20 random candidates from pool
        for _ in range(20):
            idx = rng.randint(0, len(pool_data) - 1)
            r, d, s = pool_data[idx]
            # Check if scope is not worse
            if s <= old_scope + 5: # Allow slight increase if it helps packing? No, we want min scope.
                # Check compatibility
                if not (used_without_i & d):
                    candidates.append((s, r, d))
                    
        if not candidates:
            continue
            
        # Pick the one with smallest scope
        candidates.sort(key=lambda x: x[0])
        new_scope, new_row, new_diffs = candidates[0]
        
        # Apply swap
        current_rows[i] = new_row
        current_diff_sets[i] = new_diffs
        
        new_total_scope = _scope(current_rows)
        
        if new_total_scope < best_scope:
            best_scope = new_total_scope
            best_rows = [list(r) for r in current_rows]
            
        # With some probability, keep the change even if scope didn't improve immediately
        # to escape local minima? 
        # If scope increased, revert.
        if new_total_scope > current_scope:
             # Revert
             current_rows[i] = old_row
             current_diff_sets[i] = old_diffs
        elif new_total_scope == current_scope:
             # Keep or revert? Randomly keep to explore
             if rng.random() < 0.5:
                 current_rows[i] = old_row
                 current_diff_sets[i] = old_diffs
                 
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 140.0 # Safety margin
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    # Estimate a reasonable max_scope for individual rows
    # The theoretical minimum scope for a single row of k+1 elements is roughly k(k+1)/2? No.
    # For k=5, min scope is 11 (0,1,2,4,8,11? No, 0,1,2,3,4,5 -> diffs 1..5. 0,1,3,7,12,18? Sidon sets).
    # Let's just use a generous bound for the pool generation, e.g., 2 * theoretical min.
    # Theoretical min for Sidon set of size 6 is 18? (0,1,4,10,18,?) 
    # Actually, let's just generate rows up to scope 100.
    
    max_row_scope = 100
    pool_size = 5000
    
    # Generate pool
    pool = _generate_configs(k, max_row_scope, rng, pool_size)
    
    if not pool:
        # Fallback to simple greedy
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = 0
                for a in row:
                    new |= (1 << (x - a))
                if (used & new) == 0:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Phase 1: Greedy Packing with multiple restarts
    # We want to find a valid packing first.
    
    best_valid = None
    best_scope = float('inf')
    
    restarts = 20
    for r in range(restarts):
        if time.time() > _DEADLINE - 30:
            break
            
        # Shuffle pool for each restart? Or just use random selection in greedy.
        # _greedy_packing shuffles internally.
        rows = _greedy_packing(pool, n_rows, rng)
        
        if rows:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_valid = [list(r) for r in rows]
                
    if best_valid is None:
        # If greedy failed, we need a fallback.
        # This shouldn't happen if pool is large enough.
        # But let's ensure we return something.
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = 0
                for a in row:
                    new |= (1 << (x - a))
                if (used & new) == 0:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    consider(best_valid)
    
    # Phase 2: Local Search Optimization
    # Use the best valid packing as a seed for local search.
    
    if time.time() < _DEADLINE - 5:
        # Run local search for remaining time
        sa_deadline = _DEADLINE - 2
        s, rows = _local_search(best_valid, k, n_rows, rng, sa_deadline, pool)
        consider(rows)
        
        # If we have time, try a few more local searches from different greedy packings
        # Or just continue optimizing the best one?
        # Let's try optimizing the current best.
        if _BEST is not None and time.time() < _DEADLINE - 2:
             # We already did one pass. Let's do another pass with the best result found.
             s2, rows2 = _local_search(_BEST[1], k, n_rows, rng, _DEADLINE - 1, pool)
             consider(rows2)

    if _BEST is not None:
        return _BEST[1]
    return best_valid

# Fallback if construct is not called or fails
if __name__ == "__main__":
    # Quick test
    pass
