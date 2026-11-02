import time
import random
import math
from collections import Counter

_BEST = None
_DEADLINE = time.time() + 110.0

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _compute_diffs(rows):
    """Compute all positive differences across all rows."""
    diffs = []
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diffs.append(r[j] - r[i])
    return diffs

def _count_collisions(diffs):
    """Count number of collisions (duplicates) in a list of differences."""
    seen = set()
    collisions = 0
    for d in diffs:
        if d in seen:
            collisions += 1
        else:
            seen.add(d)
    return collisions

def _fast_greedy(k, n_rows, rng=None, spread=False):
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
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _sa_optimize(rows, k, iters, rng, deadline):
    """
    Simulated Annealing on the scope.
    State: list of rows (each row is list of k+1 ints, row[0]=0).
    Objective: scope (max entry). We want to minimize it.
    Uses efficient difference tracking for fast cost evaluation.
    """
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in current_rows]
    best_scope = _scope(current_rows)
    
    # Precompute difference counts for fast cost evaluation
    # diff_count[d] = number of times difference d appears
    # collisions = sum over d of max(0, diff_count[d] - 1)
    
    def build_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts
    
    def get_scope_and_collisions(rows, diff_counts):
        scope = max(a for r in rows for a in r[1:])
        collisions = sum(max(0, c - 1) for c in diff_counts.values())
        return scope, collisions
    
    def get_cost(rows, diff_counts):
        scope, collisions = get_scope_and_collisions(rows, diff_counts)
        return scope + 1000 * collisions
    
    # Initial state
    current_diff_counts = build_diff_counts(current_rows)
    current_scope, current_collisions = get_scope_and_collisions(current_rows, current_diff_counts)
    current_cost = current_scope + 1000 * current_collisions
    best_cost = current_cost
    
    T = 10.0
    cooling = 0.999
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        # Pick a random element to change
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)  # index in row, 1..k
        
        old_val = current_rows[i][j]
        
        # Determine range for new value
        # Must be > prev and < next
        prev_val = current_rows[i][j-1]
        next_val = current_rows[i][j+1] if j < k else current_scope + 10
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
            
        # Propose new value
        # Sometimes small jump, sometimes larger
        if rng.random() < 0.7:
            span = hi - lo
            # Gaussian-like jump
            delta = int(rng.gauss(0, max(1, span * 0.2)))
            new_val = old_val + delta
        else:
            new_val = rng.randint(lo, hi)
            
        if new_val == old_val:
            continue
            
        # Check strict increase before applying
        row = current_rows[i]
        if new_val <= prev_val:
            continue
        if j < k and new_val >= row[j+1]:
            continue
        
        # Compute the changes in differences
        # Old differences involving position j:
        # For each t < j: old_diff = old_val - row[t]
        # For each t > j: old_diff = row[t] - old_val
        # New differences:
        # For each t < j: new_diff = new_val - row[t]
        # For each t > j: new_diff = row[t] - new_val
        
        # Count how many times each difference appears (before change)
        # We'll track the delta in collision count
        
        # First, remove old contributions
        old_diffs_removed = []
        for t in range(j):
            d = old_val - row[t]
            old_diffs_removed.append(d)
        for t in range(j+1, k+1):
            d = row[t] - old_val
            old_diffs_removed.append(d)
        
        # Add new contributions
        new_diffs_added = []
        for t in range(j):
            d = new_val - row[t]
            new_diffs_added.append(d)
        for t in range(j+1, k+1):
            d = row[t] - new_val
            new_diffs_added.append(d)
        
        # Compute collision change
        # For each difference d, collision contribution = max(0, count[d] - 1)
        # We need to track how counts change
        
        collision_delta = 0
        
        # Process removals
        temp_counts = dict(current_diff_counts)
        for d in old_diffs_removed:
            if d in temp_counts:
                old_count = temp_counts[d]
                # Removing one: collision contribution changes from max(0, old_count-1) to max(0, old_count-2)
                old_coll = max(0, old_count - 1)
                new_coll = max(0, old_count - 2)
                collision_delta += (new_coll - old_coll)
                temp_counts[d] = old_count - 1
                if temp_counts[d] == 0:
                    del temp_counts[d]
        
        # Process additions
        for d in new_diffs_added:
            old_count = temp_counts.get(d, 0)
            # Adding one: collision contribution changes from max(0, old_count-1) to max(0, old_count)
            old_coll = max(0, old_count - 1)
            new_coll = max(0, old_count)
            collision_delta += (new_coll - old_coll)
            temp_counts[d] = old_count + 1
        
        new_collisions = current_collisions + collision_delta
        new_scope = max(current_scope, new_val)
        new_cost = new_scope + 1000 * new_collisions
        
        # Metropolis
        if new_cost < current_cost or (T > 0.01 and rng.random() < math.exp(-(new_cost - current_cost) / max(T, 0.1))):
            # Accept: update state
            current_rows[i][j] = new_val
            current_diff_counts = temp_counts
            current_scope = new_scope
            current_collisions = new_collisions
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Reject: no change needed since we didn't modify current_rows yet
            pass
        T *= cooling
        
    return best_scope, best_rows

def _polish(rows, k, deadline, rng):
    """
    Final high-precision polish pass: try to reduce scope by 1 repeatedly.
    For a target scope S, check if we can fit all rows within [0, S]
    with all differences distinct. Use local search to find such a configuration.
    """
    current_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    best_scope = current_scope
    
    # Try to reduce scope by 1 each time
    target_scope = best_scope - 1
    max_attempts = 5  # Try to reduce by at most 5
    
    for _ in range(max_attempts):
        if time.time() > deadline:
            break
        if target_scope < k:  # Can't have scope less than k for k+1 elements starting at 0
            break
        
        # Try to find a valid configuration with max entry <= target_scope
        # Use local search on the current best
        found = False
        search_rows = [list(r) for r in best_rows]
        
        # Clamp values to target_scope
        for i in range(len(search_rows)):
            for j in range(1, k+1):
                if search_rows[i][j] > target_scope:
                    search_rows[i][j] = target_scope - (k - j)  # Ensure strict increase
        
        # Verify strict increase after clamping
        valid_structure = True
        for i in range(len(search_rows)):
            for j in range(1, k+1):
                if search_rows[i][j] <= search_rows[i][j-1]:
                    valid_structure = False
                    break
            if not valid_structure:
                break
        
        if not valid_structure:
            target_scope -= 1
            continue
        
        # Local search to fix collisions
        diff_counts = {}
        for r in search_rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        collisions = sum(max(0, c-1) for c in diff_counts.values())
        
        if collisions == 0 and _valid(search_rows, k):
            best_rows = [list(r) for r in search_rows]
            best_scope = target_scope
            found = True
            target_scope -= 1
            continue
        
        # Do local search to reduce collisions
        T = 5.0
        cooling = 0.998
        iters = 3000
        
        for it in range(iters):
            if time.time() > deadline:
                break
            if collisions == 0:
                break
                
            i = rng.randint(0, len(search_rows) - 1)
            j = rng.randint(1, k)
            
            old_val = search_rows[i][j]
            prev_val = search_rows[i][j-1]
            next_val = search_rows[i][j+1] if j < k else target_scope + 1
            
            lo = prev_val + 1
            hi = min(next_val - 1, target_scope)
            
            if lo > hi:
                continue
            
            if rng.random() < 0.7:
                span = hi - lo
                delta = int(rng.gauss(0, max(1, span * 0.15)))
                new_val = old_val + delta
            else:
                new_val = rng.randint(lo, hi)
            
            if new_val == old_val or new_val <= prev_val or (j < k and new_val >= search_rows[i][j+1]):
                continue
            
            # Compute collision change
            temp_counts = dict(diff_counts)
            collision_delta = 0
            
            for t in range(j):
                d = old_val - search_rows[i][t]
                if d in temp_counts:
                    oc = temp_counts[d]
                    collision_delta += (max(0, oc-2) - max(0, oc-1))
                    temp_counts[d] = oc - 1
                    if temp_counts[d] == 0:
                        del temp_counts[d]
            for t in range(j+1, k+1):
                d = search_rows[i][t] - old_val
                if d in temp_counts:
                    oc = temp_counts[d]
                    collision_delta += (max(0, oc-2) - max(0, oc-1))
                    temp_counts[d] = oc - 1
                    if temp_counts[d] == 0:
                        del temp_counts[d]
            
            for t in range(j):
                d = new_val - search_rows[i][t]
                oc = temp_counts.get(d, 0)
                collision_delta += (max(0, oc) - max(0, oc-1))
                temp_counts[d] = oc + 1
            for t in range(j+1, k+1):
                d = search_rows[i][t] - new_val
                oc = temp_counts.get(d, 0)
                collision_delta += (max(0, oc) - max(0, oc-1))
                temp_counts[d] = oc + 1
            
            new_collisions = collisions + collision_delta
            new_cost = new_collisions
            
            if new_cost < collisions or (T > 0.01 and rng.random() < math.exp(-(new_cost - collisions) / max(T, 0.1))):
                search_rows[i][j] = new_val
                diff_counts = temp_counts
                collisions = new_collisions
            T *= cooling
        
        if collisions == 0 and _valid(search_rows, k):
            best_rows = [list(r) for r in search_rows]
            best_scope = target_scope
            target_scope -= 1
    
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 110.0
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    # 1. Generate several initial candidates using different greedy strategies
    candidates = []
    
    # Standard greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    # Spread greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Randomized greedy (shuffle candidates)
    for _ in range(5):
        used = 0
        rows = []
        ok = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a range of candidates
                found = False
                # Randomly decide step size
                max_step = rng.randint(1, 10)
                for t in range(x + 1, x + 1 + max_step * 3):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    # Fallback: find smallest valid
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)
        
    # 2. Run SA on the best few candidates
    # Sort candidates by scope
    candidates.sort(key=_scope)
    
    # Run SA on the top 3 candidates
    for cand in candidates[:3]:
        if time.time() > _DEADLINE:
            break
        # Run SA for a while
        iters = 5000
        s, rows = _sa_optimize(cand, k, iters, rng, _DEADLINE)
        consider(rows)
        
    # 3. If time remains, run SA on the best result found so far
    if _BEST is not None and time.time() < _DEADLINE:
        remaining_time = _DEADLINE - time.time()
        # Estimate iterations based on remaining time
        # Roughly 10000 iters per second is optimistic, let's be safe
        extra_iters = int(remaining_time * 5000)
        if extra_iters > 1000:
            s, rows = _sa_optimize(_BEST[1], k, extra_iters, rng, _DEADLINE)
            consider(rows)
            
    # 4. Final polish pass to try to reduce scope further
    if _BEST is not None and time.time() < _DEADLINE - 5.0:
        s, rows = _polish(_BEST[1], k, _DEADLINE - 2.0, rng)
        consider(rows)
    
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
