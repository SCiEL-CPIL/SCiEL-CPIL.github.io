import time
import random
import math
import sys
from collections import Counter

_DEADLINE = 0.0
_BEST = None

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

def _row_diffs(row):
    return [row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))]

def _greedy_init(n_rows, k, rng, M_hint=120):
    """Greedy construction with bitmask difference tracking."""
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

def _backtracking_solve(n_rows, k, M, forbidden, rng, deadline, max_nodes=5_000_000):
    """
    Backtracking search for n_rows disjoint (k+1,1)-multisets within [1..M].
    forbidden: set of difference values that must NOT be used (Benders cuts).
    Returns list of rows or None.
    """
    total_diffs = n_rows * (k * (k + 1) // 2)
    if total_diffs > M - len(forbidden):
        return None  # counting bound infeasibility
    
    # Precompute: for each possible row (tuple of k+1 increasing ints starting at 0),
    # check if its diffs are within [1..M] and not in forbidden.
    # But enumerating all rows is infeasible. Instead, build rows one at a time.
    
    # State: used_diffs as a set (for global uniqueness), rows built so far.
    used_diffs = set()
    rows = []
    
    # MRV ordering: for each row position, the number of available difference values
    # decreases as we use more. We'll build rows in order, but with a heuristic:
    # try to build the "tightest" row first (smallest max element).
    
    sys.setrecursionlimit(10000)
    
    node_count = [0]
    
    def backtrack(row_idx):
        if time.time() > deadline:
            return False
        node_count[0] += 1
        if node_count[0] > max_nodes:
            return False
        
        if row_idx == n_rows:
            return True
        
        # Build row row_idx
        # The row must have k+1 entries starting at 0, strictly increasing, all in [0..M]
        # All pairwise diffs must be in [1..M] \ forbidden \ used_diffs
        available = set(range(1, M + 1)) - forbidden - used_diffs
        if len(available) < k * (k + 1) // 2:
            return False
        
        # Build the row greedily with backtracking
        row = [0]
        x = 0
        
        def build_row(pos):
            if time.time() > deadline:
                return False
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False
            
            if pos == k + 1:
                return True
            
            # Try candidate values for row[pos]
            # Heuristic: try smaller values first (to keep scope small)
            start = x + 1
            end = min(M, start + 50)  # limit search window for speed
            
            # Collect valid candidates
            candidates = []
            for t in range(start, end + 1):
                # Check all diffs with existing row elements
                ok = True
                new_diffs = []
                for a in row:
                    d = t - a
                    if d <= 0 or d > M or d in forbidden or d in used_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                if ok:
                    candidates.append((t, new_diffs))
            
            # Shuffle candidates with a bias toward smaller values
            if len(candidates) > 1:
                # Weighted shuffle: smaller values more likely to be tried first
                weights = [1.0 / (c[0] + 1) for c in candidates]
                total_w = sum(weights)
                rng.shuffle(candidates)  # simple shuffle for variety
            
            for t, new_diffs in candidates:
                # Try this candidate
                row.append(t)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if build_row(pos + 1):
                    return True
                
                # Undo
                row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        # Try multiple starting approaches for this row
        # Approach 1: greedy smallest-first
        if build_row(1):
            rows.append(list(row))
            if backtrack(row_idx + 1):
                return True
            # Undo row
            rows.pop()
            for d in _row_diffs(row):
                used_diffs.discard(d)
        
        return False
    
    # Initial call
    if backtrack(0):
        return rows
    return None

def _benders_solve(n_rows, k, rng, deadline):
    """
    Benders-style cutting plane: start with small M, try to solve,
    if infeasible, increase M and add cuts.
    """
    # Lower bound from counting: total_diffs = n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    # Need at least total_diffs distinct positive integers in [1..M]
    M_lower = total_diffs
    # Also, each row has k+1 elements, so max element >= k
    M_lower = max(M_lower, k)
    
    # Add some margin
    M = M_lower + 5
    
    forbidden = set()
    best_result = None
    attempts = 0
    
    while time.time() < deadline and M <= 200:
        attempts += 1
        # Try to solve with current M and forbidden set
        sub_deadline = min(time.time() + 8.0, deadline)
        result = _backtracking_solve(n_rows, k, M, forbidden, rng, sub_deadline, max_nodes=3_000_000)
        
        if result is not None:
            best_result = result
            # Try to reduce M further
            # Binary search for minimum M
            lo = M_lower
            hi = M
            while lo < hi and time.time() < deadline:
                mid = (lo + hi) // 2
                sub_deadline2 = min(time.time() + 5.0, deadline)
                r2 = _backtracking_solve(n_rows, k, mid, forbidden, rng, sub_deadline2, max_nodes=2_000_000)
                if r2 is not None:
                    hi = mid
                    best_result = r2
                else:
                    lo = mid + 1
            break
        else:
            # Infeasible: increase M
            # Simple heuristic: increase by a factor
            M = min(M + max(3, M // 10), 200)
            # No explicit Benders cut extraction (complex), just increase M
            # But we can add a "soft cut": forbid the largest differences
            # that are least likely to be useful
            if len(forbidden) < 50:
                # Forbid some large values to force tighter packing
                large_vals = [d for d in range(M, M - 20, -1) if d not in forbidden]
                for v in large_vals[:3]:
                    forbidden.add(v)
    
    return best_result

def _sa_optimize(rows, k, iters, rng, deadline):
    """Lightweight SA polish to fine-tune the backtracking result."""
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    diff_counts = Counter()
    for r in current_rows:
        for d in _row_diffs(r):
            diff_counts[d] += 1
    
    def get_cost(rows, dc):
        scope = _scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 1000 * collisions
    
    current_cost = get_cost(current_rows, diff_counts)
    best_cost = current_cost
    T = 2.0
    cooling = 0.9999
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        # Single-element move
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current_rows[i][j]
        prev_val = current_rows[i][j - 1]
        next_val = current_rows[i][j + 1] if j < k else current_scope + 10
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
        
        span = hi - lo
        if span <= 1:
            continue
        
        delta = int(rng.gauss(0, max(1, span * 0.15)))
        new_val = old_val + delta
        new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            continue
        
        # Remove old diffs
        old_diffs = _row_diffs(current_rows[i])
        for d in old_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
        
        current_rows[i][j] = new_val
        
        # Add new diffs
        new_diffs = _row_diffs(current_rows[i])
        for d in new_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        new_cost = get_cost(current_rows, diff_counts)
        
        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, 0.01)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = _scope(current_rows)
                best_rows = [list(r) for r in current_rows]
        else:
            # Undo
            for d in new_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            current_rows[i][j] = old_val
            
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        T = max(T * cooling, 0.01)
    
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 105.0  # safety margin before 120s limit
    
    def consider(rows):
        global _BEST
        if rows is None:
            return
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
    
    rng = random.Random(42)
    
    # Phase 1: Greedy initialization (fast, gives a valid upper bound)
    greedy_rows = _greedy_init(n_rows, k, rng)
    consider(greedy_rows)
    
    # Phase 2: Benders-style backtracking search
    bt_deadline = _DEADLINE - 20.0  # leave 20s for polish
    bt_result = _benders_solve(n_rows, k, rng, bt_deadline)
    if bt_result is not None:
        consider(bt_result)
    
    # Phase 3: SA polish on the best result found
    if _BEST is not None and time.time() < _DEADLINE:
        remaining = _DEADLINE - time.time()
        if remaining > 5.0:
            iters = int(remaining * 5000)
            iters = max(iters, 1000)
            s, rows = _sa_optimize(_BEST[1], k, iters, rng, _DEADLINE)
            consider(rows)
    
    if _BEST is not None:
        return _BEST[1]
    return greedy_rows
