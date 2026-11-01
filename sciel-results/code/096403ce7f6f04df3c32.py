"""Rewritten program: conflict-directed local search with targeted descent for difference triangle sets."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Return set of all positive differences in a row."""
    L = len(row)
    diffs = set()
    for i in range(L):
        ai = row[i]
        for j in range(i + 1, L):
            diffs.add(row[j] - ai)
    return diffs


def _all_diffs_counter(rows):
    """Return dict mapping difference -> count."""
    cnt = defaultdict(int)
    for row in rows:
        for d in _row_diffs_set(row):
            cnt[d] += 1
    return cnt


def _conflicts_from_counter(cnt):
    """Return list of differences with count > 1."""
    return [d for d, c in cnt.items() if c > 1]


def _total_conflicts(cnt):
    """Return total number of excess conflicts (sum of (count-1) for count>1)."""
    return sum(c - 1 for c in cnt.values() if c > 1)


def _conflict_positions(rows, cnt):
    """
    For each row, find positions (i, j) where the difference row[j]-row[i] is conflicting.
    Returns dict: (row_idx, pos) -> conflict_count
    """
    pos_conf = defaultdict(int)
    for r, row in enumerate(rows):
        L = len(row)
        for i in range(L):
            ai = row[i]
            for j in range(i + 1, L):
                d = row[j] - ai
                if cnt[d] > 1:
                    pos_conf[(r, i)] += 1
                    pos_conf[(r, j)] += 1
    return pos_conf


def _greedy_row(used, k, rng, max_x=500):
    """Build one row greedily."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        best_x = None
        for trial in range(max_x):
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                best_x = x
                break
        if best_x is None:
            for _ in range(500):
                cand = rng.randint(x + 1, x + 200)
                new = {cand - a for a in row}
                if not (new & used):
                    best_x = cand
                    break
        if best_x is None:
            return None
        row.append(best_x)
        used |= {best_x - a for a in row[:-1]}
        x = best_x
    return row


def _construct_greedy(n_rows, k, rng, max_x=500):
    """Construct all rows greedily."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_x)
        if row is None:
            return None
        rows.append(row)
    return rows


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    cnt = _all_diffs_counter(rows)
    return all(c == 1 for c in cnt.values())


def _scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)


def _targeted_descent(rows, n_rows, k, cnt, rng, time_budget):
    """
    Targeted descent: repeatedly pick the position with most conflicts,
    try small perturbations and random values to reduce scope or conflicts.
    """
    start = time.time()
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    best_cnt = dict(cnt)
    best_conflicts = _total_conflicts(cnt)
    
    # Precompute row diff sets for quick updates
    row_diff_sets = [_row_diffs_set(r) for r in rows]
    
    while time.time() - start < time_budget:
        # Find most-conflicted position
        pos_conf = _conflict_positions(rows, cnt)
        if not pos_conf:
            break
        
        max_conf = max(pos_conf.values())
        candidates = [p for p, c in pos_conf.items() if c == max_conf]
        (r, idx) = rng.choice(candidates)
        
        row = rows[r]
        lo = row[idx - 1] + 1 if idx > 0 else 1
        hi = row[idx + 1] - 1 if idx < len(row) - 1 else best_scope + 30
        
        # Generate candidate values: small perturbations + random + targeted low values
        trial_vals = set()
        for delta in range(-8, 9):
            v = row[idx] + delta
            if lo <= v <= hi:
                trial_vals.add(v)
        # Targeted: try values near the lower bound to reduce scope
        for v in range(lo, min(lo + 15, hi + 1)):
            trial_vals.add(v)
        # Random candidates
        for _ in range(40):
            v = rng.randint(lo, hi)
            trial_vals.add(v)
        
        best_new_val = None
        best_new_row = None
        best_new_scope = best_scope + 10000
        best_new_conflicts = best_conflicts + 10000
        best_new_cnt = None
        
        for v in sorted(trial_vals):
            if v > best_scope + 50:
                continue
            new_row = list(row)
            new_row[idx] = v
            if any(new_row[t] >= new_row[t + 1] for t in range(len(new_row) - 1)):
                continue
            
            new_diffs = _row_diffs_set(new_row)
            
            # Update counter: remove old diffs, add new
            new_cnt = dict(cnt)
            for d in row_diff_sets[r]:
                new_cnt[d] -= 1
                if new_cnt[d] == 0:
                    del new_cnt[d]
            for d in new_diffs:
                new_cnt[d] = new_cnt.get(d, 0) + 1
            
            new_conflicts = _total_conflicts(new_cnt)
            new_scope = max(_scope([rows[rr] if rr != r else new_row for rr in range(n_rows)]))
            
            # Accept if it reduces conflicts, or same conflicts with smaller scope
            if new_conflicts < best_new_conflicts or (new_conflicts == best_new_conflicts and new_scope < best_new_scope):
                best_new_conflicts = new_conflicts
                best_new_scope = new_scope
                best_new_val = v
                best_new_row = new_row
                best_new_cnt = new_cnt
        
        if best_new_row is not None:
            rows[r] = best_new_row
            row_diff_sets[r] = _row_diffs_set(best_new_row)
            cnt = best_new_cnt
            cur_scope = _scope(rows)
            cur_conflicts = _total_conflicts(cnt)
            if cur_scope < best_scope or (cur_scope == best_scope and cur_conflicts < best_conflicts):
                best_scope = cur_scope
                best_conflicts = cur_conflicts
                best_rows = [list(r) for r in rows]
                best_cnt = dict(cnt)
        else:
            # Try a different position
            continue
    
    return best_rows, best_scope, best_cnt


def _simulated_annealing(rows, n_rows, k, rng, time_budget):
    """
    Simulated annealing with conflict-directed moves.
    State: complete assignment of all row entries.
    Move: pick a random row and position, change its value.
    Cost: total_conflicts * 1000 + scope
    """
    start = time.time()
    T = 50.0
    T_min = 0.5
    cool_rate = 0.999
    
    cur_rows = [list(r) for r in rows]
    cur_cnt = _all_diffs_counter(cur_rows)
    cur_row_diffs = [_row_diffs_set(r) for r in cur_rows]
    cur_scope = _scope(cur_rows)
    cur_conflicts = _total_conflicts(cur_cnt)
    cur_cost = cur_conflicts * 1000 + cur_scope
    
    best_rows = [list(r) for r in cur_rows]
    best_scope = cur_scope
    best_cnt = dict(cur_cnt)
    best_cost = cur_cost
    
    while time.time() - start < time_budget and T > T_min:
        # Pick a random row and position
        r = rng.randrange(n_rows)
        idx = rng.randrange(len(cur_rows[r]))
        
        row = cur_rows[r]
        lo = row[idx - 1] + 1 if idx > 0 else 1
        hi = row[idx + 1] - 1 if idx < len(row) - 1 else best_scope + 20
        
        # Generate candidate
        if rng.random() < 0.3:
            # Targeted: near lower bound
            v = rng.randint(lo, min(lo + 20, hi))
        else:
            v = rng.randint(lo, hi)
        
        if v == row[idx]:
            continue
        
        new_row = list(row)
        new_row[idx] = v
        if any(new_row[t] >= new_row[t + 1] for t in range(len(new_row) - 1)):
            continue
        
        new_diffs = _row_diffs_set(new_row)
        
        # Update counter
        new_cnt = dict(cur_cnt)
        for d in cur_row_diffs[r]:
            new_cnt[d] -= 1
            if new_cnt[d] == 0:
                del new_cnt[d]
        for d in new_diffs:
            new_cnt[d] = new_cnt.get(d, 0) + 1
        
        new_conflicts = _total_conflicts(new_cnt)
        new_scope = max(_scope([cur_rows[rr] if rr != r else new_row for rr in range(n_rows)]))
        new_cost = new_conflicts * 1000 + new_scope
        
        delta = new_cost - cur_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.1)):
            # Accept move
            cur_rows[r] = new_row
            cur_row_diffs[r] = new_diffs
            cur_cnt = new_cnt
            cur_scope = new_scope
            cur_conflicts = new_conflicts
            cur_cost = new_cost
            
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_scope = cur_scope
                best_rows = [list(r) for r in cur_rows]
                best_cnt = dict(cur_cnt)
        
        T *= cool_rate
    
    return best_rows, best_scope, best_cnt


def construct(n_rows, k):
    start = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    rng = random.Random(12345)
    
    # Phase 1: Multiple greedy constructions
    best_rows = None
    best_scope = float('inf')
    best_cnt = None
    
    # Deterministic greedy
    rows = _construct_greedy(n_rows, k, rng, max_x=500)
    if rows is not None:
        if _is_valid(rows):
            scope = _scope(rows)
            best_rows = [list(r) for r in rows]
            best_scope = scope
            best_cnt = _all_diffs_counter(rows)
    
    # Randomized greedy attempts
    attempts = 0
    while time.time() - start < deadline - 60.0:
        rng2 = random.Random(attempts * 7919 + 1)
        rows = _construct_greedy(n_rows, k, rng2, max_x=500)
        if rows is not None:
            if _is_valid(rows):
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    best_cnt = _all_diffs_counter(rows)
        attempts += 1
        if attempts >= 30:
            break
    
    if best_rows is None:
        # Fallback: trivial valid set
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
    
    # Phase 2: Targeted descent
    time_left = deadline - (time.time() - start) - 20.0
    if time_left > 5.0:
        improved_rows, improved_scope, improved_cnt = _targeted_descent(
            [list(r) for r in best_rows], n_rows, k, best_cnt, rng, time_left
        )
        if improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
            best_cnt = improved_cnt
    
    # Phase 3: Simulated annealing with multiple restarts
    time_left = deadline - (time.time() - start) - 5.0
    if time_left > 3.0:
        # Run annealing from current best
        ann_rows, ann_scope, ann_cnt = _simulated_annealing(
            [list(r) for r in best_rows], n_rows, k, rng, time_left * 0.5
        )
        if ann_scope < best_scope:
            best_rows = ann_rows
            best_scope = ann_scope
            best_cnt = ann_cnt
        
        # If time remains, do another annealing run from a perturbed state
        time_left = deadline - (time.time() - start) - 3.0
        if time_left > 3.0:
            # Perturb the best solution
            perturbed = [list(r) for r in best_rows]
            for _ in range(10):
                r = rng.randrange(n_rows)
                idx = rng.randrange(len(perturbed[r]))
                row = perturbed[r]
                lo = row[idx - 1] + 1 if idx > 0 else 1
                hi = row[idx + 1] - 1 if idx < len(row) - 1 else best_scope + 10
                if hi > lo:
                    row[idx] = rng.randint(lo, hi)
                    # Fix ordering if broken
                    for t in range(len(row) - 1):
                        if row[t] >= row[t + 1]:
                            row[t + 1] = row[t] + 1
                    # Ensure last element doesn't exceed reasonable bound
                    if row[-1] > best_scope + 50:
                        row[-1] = best_scope + 10
                        for t in range(len(row) - 2, -1, -1):
                            if row[t] >= row[t + 1]:
                                row[t] = row[t + 1] - 1
                            if row[t] < (0 if t == 0 else row[t-1] + 1):
                                row[t] = (0 if t == 0 else row[t-1] + 1)
            
            ann2_rows, ann2_scope, ann2_cnt = _simulated_annealing(
                perturbed, n_rows, k, rng, time_left
            )
            if ann2_scope < best_scope:
                best_rows = ann2_rows
                best_scope = ann2_scope
                best_cnt = ann2_cnt
    
    return best_rows

# EVOLVE-BLOCK-END
