"""Rewritten program using simulated annealing / local search for difference triangle set."""
import random
import math

def _all_diffs(rows):
    """Return set of all positive differences within each row."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, k):
    """Check if all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _objective(rows):
    """Return the largest entry (scope). Lower is better."""
    return max(max(row) for row in rows)

def _random_row(k, max_val):
    """Generate a random strictly increasing row of k+1 integers starting at 0."""
    if max_val < k:
        max_val = k
    # Choose k distinct values from 1..max_val, sort them, prepend 0
    if max_val <= k:
        vals = list(range(1, max_val + 1))
        random.shuffle(vals)
        vals = sorted(vals[:k])
    else:
        vals = sorted(random.sample(range(1, max_val + 1), k))
    return [0] + vals

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all pairwise differences
    within rows are distinct across all rows.
    
    Uses simulated annealing with local search.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * n_rows] if k == 0 else []
    
    # Total number of differences needed: n_rows * C(k+1, 2)
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Counting bound: need at least total_diffs distinct positive integers
    # So scope must be at least total_diffs (since differences range from 1 to scope)
    counting_bound = total_diffs
    
    # Known best for (7,5) is 112. Let's start with a reasonable upper bound.
    # For general n_rows, k, use a heuristic upper bound.
    # Start with greedy-like initialization, then improve with SA.
    
    # Phase 1: Greedy initialization (similar to seed but as starting point)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    best_score = _objective(rows)
    best_rows = [r[:] for r in rows]
    
    # Phase 2: Simulated annealing
    # We'll try to reduce the scope by randomly modifying rows and accepting
    # improvements or occasionally worsening moves.
    
    current_rows = [r[:] for r in rows]
    current_score = best_score
    
    # Temperature schedule
    T0 = 10.0
    T_min = 0.01
    cooling = 0.9995
    T = T0
    
    # Number of iterations - need to be careful about time (120s limit)
    # Each iteration is cheap, so we can do many.
    max_iters = 200000
    rng = random.Random(42)
    
    for it in range(max_iters):
        if T < T_min:
            break
        
        # Pick a random row to modify
        ri = rng.randrange(n_rows)
        old_row = current_rows[ri][:]
        
        # Strategy: try to reduce the last element, or swap some elements
        # First, try to reduce the maximum of this row
        max_idx = len(old_row) - 1
        old_max = old_row[max_idx]
        
        # Try a new value for the last element (or some random element)
        # with probability, also try modifying other elements
        
        new_row = old_row[:]
        if rng.random() < 0.7:
            # Modify the last element
            if old_max > 1:
                # Try a smaller value
                new_val = rng.randint(1, old_max - 1)
                if new_val > new_row[-2]:  # must be increasing
                    new_row[-1] = new_val
                else:
                    # Try a different element
                    idx = rng.randrange(1, len(new_row))
                    new_val = rng.randint(new_row[idx-1] + 1, min(new_row[idx+1] - 1 if idx < len(new_row)-1 else old_max, old_max))
                    if new_val > new_row[idx-1] and (idx == len(new_row)-1 or new_val < new_row[idx+1]):
                        new_row[idx] = new_val
                    else:
                        continue
        else:
            # Randomly modify a random non-zero element
            idx = rng.randrange(1, len(new_row))
            lo = new_row[idx-1] + 1
            hi = new_row[idx+1] - 1 if idx < len(new_row)-1 else current_score
            if hi < lo:
                continue
            new_val = rng.randint(lo, hi)
            new_row[idx] = new_val
        
        # Check if new row is strictly increasing and starts at 0
        if new_row[0] != 0:
            continue
        for i in range(1, len(new_row)):
            if new_row[i] <= new_row[i-1]:
                continue
        
        # Check validity: all diffs across all rows must be distinct
        # Efficient check: compute diffs for new row, check against all other rows' diffs
        # and within new row
        new_diffs = set()
        valid = True
        for i in range(len(new_row)):
            for j in range(i+1, len(new_row)):
                d = new_row[j] - new_row[i]
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                break
        
        if not valid:
            continue
        
        # Check against all other rows
        for ri2 in range(n_rows):
            if ri2 == ri:
                continue
            row2 = current_rows[ri2]
            for i in range(len(row2)):
                for j in range(i+1, len(row2)):
                    d = row2[j] - row2[i]
                    if d in new_diffs:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Valid move! Compute new score
        new_score = max(max(r) for r in current_rows if r is not new_row) if False else 0
        # Actually compute properly
        new_max = 0
        for ri2 in range(n_rows):
            if ri2 == ri:
                m = max(new_row)
            else:
                m = max(current_rows[ri2])
            if m > new_max:
                new_max = m
        new_score = new_max
        
        # Accept or reject
        delta = new_score - current_score
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_rows[ri] = new_row
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_rows = [r[:] for r in current_rows]
        
        T *= cooling
    
    # Return the best found
    return best_rows
