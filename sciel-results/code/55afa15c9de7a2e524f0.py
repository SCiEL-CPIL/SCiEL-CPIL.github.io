"""Constraint-satisfaction search for the difference-triangle problem."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_fallback(n_rows, k):
    """Original greedy: always valid, used as safety net."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _scope(rows):
    return max(max(r) for r in rows)


def _row_diffs(row):
    """Return the set of all positive differences within a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the set of all positive differences across all rows."""
    used = set()
    for r in rows:
        used |= _row_diffs(r)
    return used


def _validate(rows, n_rows, k):
    """Check that rows is a valid solution."""
    if len(rows) != n_rows:
        return False
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        for i in range(len(r) - 1):
            if r[i + 1] <= r[i]:
                return False
    # Check all differences are distinct
    seen = set()
    for r in rows:
        rd = _row_diffs(r)
        if rd & seen:
            return False
        seen |= rd
    return True


def _randomized_greedy(n_rows, k, deadline, rng, bias=0.0):
    """
    Randomized greedy construction.
    bias: probability of picking the smallest valid candidate (0=fully random, 1=always smallest)
    """
    used, rows = set(), []
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Collect valid candidates in a window
            window = 15
            candidates = []
            for cand in range(x + 1, x + 1 + window):
                new = {cand - a for a in row}
                if not new & used:
                    candidates.append((sum(new), cand))
            if not candidates:
                # Extend search
                while x < 200:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        candidates.append((sum(new), x))
                        break
            if not candidates:
                # Fallback: just increment
                x += 1
                row.append(x)
                used |= {x - a for a in row[:-1]}
                continue
            
            candidates.sort()
            if rng.random() < bias:
                chosen = candidates[0][1]
            else:
                # Weighted random: prefer smaller sums
                weights = [1.0 / (1 + c[0] / 10) for c in candidates]
                total = sum(weights)
                r = rng.random() * total
                cum = 0
                chosen = candidates[-1][1]
                for c, w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        chosen = c[1]
                        break
            
            row.append(chosen)
            used |= {chosen - a for a in row[:-1]}
            x = chosen
        rows.append(row)
    return rows


def _local_search(rows, k, deadline, rng):
    """
    Local search: try to reduce scope by moving entries down.
    """
    best_rows = [list(r) for r in rows]
    best_sc = _scope(best_rows)
    
    def compute_used(rws):
        used = set()
        for r in rws:
            used |= _row_diffs(r)
        return used
    
    improved = True
    t0 = time.time()
    while improved and time.time() < deadline:
        improved = False
        # Try to reduce each entry in each row
        for ri in range(len(best_rows)):
            row = best_rows[ri]
            # Process from last to first to maintain increasing order
            for pos in range(len(row) - 1, 0, -1):
                lower = row[pos - 1] + 1
                upper = row[pos] - 1
                if upper < lower:
                    continue
                # Try to find a smaller valid value
                old = row[pos]
                found = False
                for cand in range(lower, old):
                    row[pos] = cand
                    # Check validity
                    rd = _row_diffs(row)
                    # Check internal consistency
                    if len(rd) != len(row) * (len(row) - 1) // 2:
                        row[pos] = old
                        continue
                    # Check against other rows
                    valid = True
                    for rj in range(len(best_rows)):
                        if rj == ri:
                            continue
                        other = _row_diffs(best_rows[rj])
                        if rd & other:
                            valid = False
                            break
                    if valid:
                        new_sc = _scope(best_rows)
                        if new_sc <= best_sc:
                            best_sc = new_sc
                            improved = True
                            found = True
                            break
                    row[pos] = old
                if found:
                    continue
                # Try increasing slightly to allow further reductions elsewhere
                # (skip for now, focus on reduction)
        # Also try to reduce the global max
        max_val = best_sc
        max_rows = [i for i, r in enumerate(best_rows) if max(r) == max_val]
        for ri in max_rows:
            row = best_rows[ri]
            pos = len(row) - 1  # last element is the max
            old = row[pos]
            lower = row[pos - 1] + 1
            for cand in range(lower, old):
                row[pos] = cand
                rd = _row_diffs(row)
                if len(rd) != len(row) * (len(row) - 1) // 2:
                    row[pos] = old
                    continue
                valid = True
                for rj in range(len(best_rows)):
                    if rj == ri:
                        continue
                    other = _row_diffs(best_rows[rj])
                    if rd & other:
                        valid = False
                        break
                if valid:
                    new_sc = _scope(best_rows)
                    if new_sc < best_sc:
                        best_sc = new_sc
                        improved = True
                        break
                row[pos] = old
    return best_rows, best_sc


def _simulated_annealing(rows, k, deadline, rng):
    """
    Simulated annealing to improve the solution.
    Moves: pick a random row and position, try to change the value.
    """
    best_rows = [list(r) for r in rows]
    best_sc = _scope(best_rows)
    current_rows = [list(r) for r in rows]
    current_sc = _scope(current_rows)
    
    T = 10.0
    cooling = 0.9995
    t0 = time.time()
    
    while time.time() < deadline and T > 0.1:
        # Pick a random move
        ri = rng.randrange(len(current_rows))
        row = current_rows[ri]
        pos = rng.randrange(1, len(row))  # positions 1..k
        
        # Determine range for new value
        lower = row[pos - 1] + 1
        upper = min(row[pos] + 5, 200)
        
        if upper <= lower:
            T *= cooling
            continue
        
        old_val = row[pos]
        new_val = rng.randint(lower, upper)
        if new_val == old_val:
            continue
        
        row[pos] = new_val
        
        # Check validity
        rd = _row_diffs(row)
        valid = len(rd) == len(row) * (len(row) - 1) // 2
        if valid:
            for rj in range(len(current_rows)):
                if rj == ri:
                    continue
                other = _row_diffs(current_rows[rj])
                if rd & other:
                    valid = False
                    break
            # Check increasing order
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    valid = False
                    break
        
        new_sc = _scope(current_rows)
        
        if valid and new_sc <= current_sc:
            # Accept
            if new_sc < best_sc:
                best_sc = new_sc
                best_rows = [list(r) for r in current_rows]
        else:
            # Reject or accept with probability
            delta = new_sc - current_sc
            if delta < 0 or rng.random() < math.exp(-delta / T):
                pass  # accept
            else:
                row[pos] = old_val  # revert
        
        T *= cooling
    
    return best_rows, best_sc


def _conflict_graph_search(n_rows, k, deadline, rng):
    """
    Search using conflict graph ideas: generate candidate rows, then select
    a set of n_rows rows with no shared differences.
    """
    # Generate a pool of candidate rows
    candidates = []
    used_diffs_pool = set()
    
    # Generate diverse candidates
    for attempt in range(200):
        row, x = [0], 0
        valid = True
        while len(row) < k + 1:
            # Try a few candidates
            found = False
            for cand in range(x + 1, x + 1 + 10):
                new = {cand - a for a in row}
                if not new & used_diffs_pool and len(new) == len(row):
                    row.append(cand)
                    used_diffs_pool |= new
                    x = cand
                    found = True
                    break
            if not found:
                x += 1
                new = {x - a for a in row}
                if not new & used_diffs_pool and len(new) == len(row):
                    row.append(x)
                    used_diffs_pool |= new
                    found = True
                else:
                    valid = False
                    break
            if not found:
                valid = False
                break
        if valid:
            candidates.append(row)
        used_diffs_pool = set()  # Reset for next candidate
    
    if not candidates:
        return _greedy_fallback(n_rows, k), _scope(_greedy_fallback(n_rows, k))
    
    # Sort candidates by scope
    candidates.sort(key=lambda r: max(r))
    
    # Greedy selection with backtracking
    best_rows = None
    best_sc = float('inf')
    
    # Try different orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        # Shuffle candidates
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        
        selected = []
        used = set()
        for cand in shuffled:
            rd = _row_diffs(cand)
            if not rd & used:
                selected.append(cand)
                used |= rd
                if len(selected) == n_rows:
                    sc = _scope(selected)
                    if sc < best_sc:
                        best_sc = sc
                        best_rows = [list(r) for r in selected]
                    break
    
    if best_rows is None:
        # Fallback to greedy
        rows = _greedy_fallback(n_rows, k)
        return rows, _scope(rows)
    
    return best_rows, best_sc


def construct(n_rows, k):
    deadline = time.time() + 120.0  # safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy attempts
    best_rows = None
    best_sc = float('inf')
    
    t_phase1_end = deadline - 60.0
    while time.time() < t_phase1_end:
        bias = rng.random() * 0.5 + 0.3
        rows = _randomized_greedy(n_rows, k, deadline, rng, bias=bias)
        sc = _scope(rows)
        if _validate(rows, n_rows, k) and sc < best_sc:
            best_sc = sc
            best_rows = rows
    
    if best_rows is None:
        best_rows = _greedy_fallback(n_rows, k)
        best_sc = _scope(best_rows)
    
    # Phase 2: Conflict graph search
    t_phase2_end = deadline - 30.0
    if time.time() < t_phase2_end:
        rows2, sc2 = _conflict_graph_search(n_rows, k, t_phase2_end, rng)
        if sc2 < best_sc:
            best_sc = sc2
            best_rows = rows2
    
    # Phase 3: Local search
    t_phase3_end = deadline - 15.0
    if time.time() < t_phase3_end:
        best_rows, best_sc = _local_search(best_rows, k, t_phase3_end, rng)
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 5.0:
        best_rows, best_sc = _simulated_annealing(best_rows, k, deadline - 5.0, rng)
    
    # Final validation
    if not _validate(best_rows, n_rows, k):
        best_rows = _greedy_fallback(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
