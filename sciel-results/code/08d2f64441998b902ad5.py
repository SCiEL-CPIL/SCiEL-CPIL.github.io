"""Difference triangle set (7,5) - find 7 rows of 7 marks minimizing max entry."""
import time
import random
import copy

def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_row(used, k):
    """Build one row greedily: start at 0, pick smallest next mark with no diff conflicts."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _greedy_all(n_rows, k):
    """Build all rows greedily, each after the previous."""
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(used, k))
    return rows

def _random_greedy(n_rows, k, rng, max_x=200):
    """Build rows with some randomness: at each step, pick from a small candidate set."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            # Try candidates from current max+1 up to some window
            base = row[-1]
            candidates = []
            for x in range(base + 1, min(base + 20, max_x + 1)):
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    candidates.append(x)
            if not candidates:
                # Fallback: scan further
                x = base
                while x < max_x:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                        break
                if not candidates:
                    return None  # failed
            # Pick randomly from top few candidates
            pick = rng.choice(candidates[:min(5, len(candidates))])
            row.append(pick)
            used |= {pick - a for a in row[:-1]}
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, rng, max_iter=5000, max_x=200):
    """
    Local search: try to reduce max entry by replacing individual marks.
    State: list of rows. Objective: max entry (lower is better), subject to validity.
    """
    best_rows = copy.deepcopy(rows)
    best_score = _max_entry(best_rows)
    
    for it in range(max_iter):
        # Pick a random row and a random position to modify
        ri = rng.randrange(n_rows)
        # Don't change position 0 (must be 0)
        pi = rng.randrange(1, k + 1)
        
        old_val = rows[ri][pi]
        # Try a different value
        # Must maintain strict increase in the row
        lo = rows[ri][pi - 1] + 1
        hi = rows[ri][pi + 1] - 1 if pi < k else max_x
        
        if lo > hi:
            continue
        
        # Try to find a valid replacement
        # Collect used diffs from all rows except the ones involving position pi in row ri
        # Actually, we need to remove old diffs of row ri, try new, check global
        row = rows[ri]
        
        # Compute all diffs in row ri
        old_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                old_diffs.add(row[j] - row[i])
        
        # Try candidate values
        candidates = list(range(lo, hi + 1))
        rng.shuffle(candidates)
        
        found = False
        for new_val in candidates[:30]:  # limit candidates tried
            if new_val == old_val:
                continue
            new_row = row[:]
            new_row[pi] = new_val
            # Check strict increase
            if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                continue
            # Compute new diffs for this row
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            # Check: new_diffs minus old_diffs must not conflict with other rows' diffs
            added = new_diffs - old_diffs
            removed = old_diffs - new_diffs
            # Check added against all other rows' diffs
            conflict = False
            for oi in range(n_rows):
                if oi == ri:
                    continue
                orow = rows[oi]
                for i in range(len(orow)):
                    for j in range(i + 1, len(orow)):
                        d = orow[j] - orow[i]
                        if d in added:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            if not conflict:
                rows[ri] = new_row
                found = True
                new_score = _max_entry(rows)
                if new_score < best_score:
                    best_score = new_score
                    best_rows = copy.deepcopy(rows)
                break
        
        if not found:
            # No improvement found this iteration; could try random perturbation
            pass
    
    return best_rows, best_score

def _anneal(n_rows, k, rng, start_rows, max_iter=20000, max_x=200):
    """
    Simulated annealing: perturb a mark, accept if valid and score improves,
    or with some probability if it doesn't worsen too much.
    """
    rows = copy.deepcopy(start_rows)
    best_rows = copy.deepcopy(rows)
    best_score = _max_entry(rows)
    
    # Precompute: for efficiency, track used diffs globally
    # But for simplicity, validate locally
    
    current_score = best_score
    
    for it in range(max_iter):
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        row = rows[ri]
        old_val = row[pi]
        
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi < k else max_x
        
        if lo >= hi:
            continue
        
        # Try a random new value in range
        new_val = rng.randint(lo, hi)
        if new_val == old_val:
            continue
        
        new_row = row[:]
        new_row[pi] = new_val
        if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
            continue
        
        # Compute new diffs for this row
        new_diffs = set()
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                new_diffs.add(new_row[j] - new_row[i])
        
        # Compute old diffs for this row
        old_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                old_diffs.add(row[j] - row[i])
        
        added = new_diffs - old_diffs
        
        # Check conflict with other rows
        conflict = False
        for oi in range(n_rows):
            if oi == ri:
                continue
            orow = rows[oi]
            for i in range(len(orow)):
                for j in range(i + 1, len(orow)):
                    d = orow[j] - orow[i]
                    if d in added:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                break
        
        if conflict:
            continue
        
        # Valid move! Accept it (always, since we only want valid states)
        rows[ri] = new_row
        new_score = _max_entry(rows)
        
        if new_score < best_score:
            best_score = new_score
            best_rows = copy.deepcopy(rows)
    
    return best_rows, best_score

def construct(n_rows, k):
    deadline = time.time() + 110  # 110s safety margin (evaluator kills at 150s)
    
    # Start with greedy solution
    rng = random.Random(42)
    
    best_rows = _greedy_all(n_rows, k)
    best_score = _max_entry(best_rows)
    
    # Try multiple random greedy starts
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
        rows = _random_greedy(n_rows, k, rng)
        if rows is None:
            continue
        if not _validate(rows):
            continue
        score = _max_entry(rows)
        if score < best_score:
            best_score = score
            best_rows = copy.deepcopy(rows)
    
    # Local search / annealing from best
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        start = copy.deepcopy(best_rows)
        # Add some perturbation
        for _ in range(3):
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            row = start[ri]
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else best_score + 20
            if lo < hi:
                new_val = rng.randint(lo, hi)
                new_row = row[:]
                new_row[pi] = new_val
                if all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1)):
                    # Check validity
                    if _validate(start):
                        start[ri] = new_row
                    else:
                        start[ri] = row  # revert
        
        rows, score = _anneal(n_rows, k, rng, start, max_iter=5000)
        if score < best_score:
            best_score = score
            best_rows = copy.deepcopy(rows)
    
    # Final local search on best
    if time.time() < deadline - 10:
        rows, score = _local_search(best_rows, n_rows, k, rng, max_iter=10000)
        if score < best_score:
            best_score = score
            best_rows = copy.deepcopy(rows)
    
    return best_rows
