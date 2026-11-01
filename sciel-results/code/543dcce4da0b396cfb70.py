"""Difference triangle set (7,5) - find 7 rows of 6 marks minimizing max entry.

Structural change: use a constraint-based backtracking search with strong
pruning (differential constraints), combined with iterative deepening on the
max-entry bound. This is a global search approach that explores the space of
valid configurations more systematically than local search/annealing.
"""
import time
import random
import copy
import sys

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

def _backtrack_solve(n_rows, k, max_entry, rng, deadline):
    """
    Constraint-based backtracking search for a valid set of rows with max entry <= max_entry.
    Uses strong pruning:
    - Each row must be strictly increasing, starting at 0, with all entries <= max_entry
    - All pairwise differences across all rows must be distinct
    - We build rows one at a time, and within each row, we build positions one at a time
    """
    # Total number of differences needed: n_rows * (k+1)*k/2
    total_diffs = n_rows * (k + 1) * k // 2
    # Available differences: 1 to max_entry
    if total_diffs > max_entry:
        return None  # Not enough room
    
    used_diffs = set()
    rows = []
    
    def solve_row(row_idx, pos, current_row):
        """Recursively build row row_idx, position pos."""
        if time.time() > deadline:
            return None
        
        if pos == k + 1:
            # Row complete
            rows.append(current_row[:])
            if row_idx + 1 == n_rows:
                return True
            return solve_row(row_idx + 1, 1, [0])
        
        if pos == 0:
            # First position is always 0
            return solve_row(row_idx, 1, [0])
        
        # Determine range for current position
        lo = current_row[-1] + 1 if pos > 0 else 1
        hi = max_entry
        
        # Additional constraint: remaining positions must fit
        remaining = k + 1 - pos
        if hi - lo + 1 < remaining:
            return None
        
        # Try values in a smart order: prefer values that leave room for remaining
        # Use a mix of ascending and random to avoid getting stuck
        candidates = list(range(lo, hi + 1))
        
        # Shuffle with bias toward smaller values (to keep max low)
        # But also try some larger values for diversity
        if rng.random() < 0.7:
            # Mostly ascending, with some randomization in the first half
            half = len(candidates) // 2
            candidates[:half] = sorted(candidates[:half], key=lambda x: rng.random())
            candidates[half:] = sorted(candidates[half:])
        else:
            rng.shuffle(candidates)
        
        for val in candidates:
            if time.time() > deadline:
                return None
            
            # Check if this value creates valid differences with existing row entries
            new_diffs = {val - a for a in current_row}
            
            # Check for internal duplicates in this row
            if len(new_diffs) != len(current_row):
                continue
            
            # Check against globally used differences
            if new_diffs & used_diffs:
                continue
            
            # Tentatively add
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            result = solve_row(row_idx, pos + 1, current_row)
            if result is not None:
                return result
            
            # Backtrack
            current_row.pop()
            used_diffs.difference_update(new_diffs)
        
        return None
    
    result = solve_row(0, 0, [])
    if result:
        return rows
    return None

def _backtrack_with_order(n_rows, k, max_entry, rng, deadline):
    """
    Backtracking with row ordering: try different orders of building rows.
    Also uses a smarter candidate ordering based on difference density.
    """
    total_diffs = n_rows * (k + 1) * k // 2
    if total_diffs > max_entry:
        return None
    
    used_diffs = set()
    rows = [None] * n_rows
    
    def try_build_row(row_idx, current_row, pos):
        """Build a single row given the used_diffs constraint."""
        if time.time() > deadline:
            return None
        
        if pos == k + 1:
            return current_row[:]
        
        if pos == 0:
            return try_build_row(row_idx, [0], 1)
        
        lo = current_row[-1] + 1
        hi = max_entry
        
        remaining = k + 1 - pos
        if hi - lo + 1 < remaining:
            return None
        
        candidates = list(range(lo, hi + 1))
        
        # Smart ordering: prefer values that minimize conflict potential
        # Sort by number of conflicts with used_diffs (ascending)
        def conflict_score(val):
            new_diffs = {val - a for a in current_row}
            # Check internal duplicates
            if len(new_diffs) != len(current_row):
                return float('inf')
            return len(new_diffs & used_diffs)
        
        scored = [(conflict_score(v), v) for v in candidates]
        scored.sort()
        
        # Take top candidates, with some randomness
        top_n = min(50, len(scored))
        top_candidates = [v for _, v in scored[:top_n]]
        
        if rng.random() < 0.3:
            rng.shuffle(top_candidates)
        
        for val in top_candidates:
            if time.time() > deadline:
                return None
            
            new_diffs = {val - a for a in current_row}
            if len(new_diffs) != len(current_row):
                continue
            if new_diffs & used_diffs:
                continue
            
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            result = try_build_row(row_idx, current_row, pos + 1)
            if result is not None:
                return result
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
        
        return None
    
    # Try building rows in different orders
    row_order = list(range(n_rows))
    rng.shuffle(row_order)
    
    for ri in row_order:
        if time.time() > deadline:
            return None
        
        row = try_build_row(ri, [], 0)
        if row is None:
            return None
        rows[ri] = row
        # Update used_diffs with all diffs in this row
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    return rows

def _local_search(rows, n_rows, k, rng, max_iter=3000, max_x=None):
    """
    Local search: try to reduce max entry by replacing individual marks.
    Uses incremental diff tracking for speed.
    """
    if max_x is None:
        max_x = _max_entry(rows) + 10
    
    best_rows = copy.deepcopy(rows)
    best_score = _max_entry(best_rows)
    
    # Precompute all diffs per row and global used set
    row_diffs = []
    for row in rows:
        dset = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                dset.add(row[j] - row[i])
        row_diffs.append(dset)
    
    all_diffs = set()
    for dset in row_diffs:
        all_diffs |= dset
    
    for it in range(max_iter):
        if time.time() > deadline_global:
            break
        
        # Pick a random row and position
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        
        row = rows[ri]
        old_val = row[pi]
        
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi < k else max_x
        
        if lo > hi:
            continue
        
        # Try candidate values, biased toward reducing max
        candidates = list(range(lo, hi + 1))
        # Bias: if old_val is the max in its row, prefer smaller values
        if old_val == max(row):
            candidates.sort(key=lambda v: abs(v - old_val) + rng.random())
        else:
            rng.shuffle(candidates)
        
        found = False
        for new_val in candidates[:20]:
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
                    new_diffs.add(new_row[j] - row[i])
            
            # Check internal duplicates
            expected = (len(new_row) * (len(new_row) - 1)) // 2
            if len(new_diffs) != expected:
                continue
            
            # Check conflicts with other rows
            added = new_diffs - row_diffs[ri]
            conflict = False
            for oi in range(n_rows):
                if oi == ri:
                    continue
                if added & row_diffs[oi]:
                    conflict = True
                    break
            
            if not conflict:
                # Accept the move
                rows[ri] = new_row
                row_diffs[ri] = new_diffs
                # Update global
                all_diffs = set()
                for dset in row_diffs:
                    all_diffs |= dset
                
                new_score = _max_entry(rows)
                if new_score < best_score:
                    best_score = new_score
                    best_rows = copy.deepcopy(rows)
                found = True
                break
        
        if not found:
            # Random restart-like perturbation occasionally
            if it % 500 == 0:
                # Try to rebuild one row from scratch
                ri = rng.randrange(n_rows)
                # Remove this row's diffs
                all_diffs = set()
                for oi in range(n_rows):
                    if oi != ri:
                        all_diffs |= row_diffs[oi]
                
                # Try to build a new row
                new_row = [0]
                success = True
                for pos in range(1, k + 1):
                    found_val = False
                    for x in range(new_row[-1] + 1, max_x + 1):
                        new_diffs = {x - a for a in new_row}
                        if len(new_diffs) == len(new_row) and not (new_diffs & all_diffs):
                            new_row.append(x)
                            all_diffs |= new_diffs
                            found_val = True
                            break
                    if not found_val:
                        success = False
                        break
                
                if success and len(new_row) == k + 1:
                    rows[ri] = new_row
                    row_diffs[ri] = set()
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            row_diffs[ri].add(new_row[j] - new_row[i])
    
    return best_rows, best_score

# Global deadline for local search
deadline_global = 0

def construct(n_rows, k):
    global deadline_global
    deadline_global = time.time() + 115  # Safety margin
    
    rng = random.Random(42)
    
    # Phase 1: Get a good initial solution with greedy
    best_rows = _greedy_all(n_rows, k)
    best_score = _max_entry(best_rows)
    
    # Phase 2: Try multiple random greedy starts
    for attempt in range(30):
        if time.time() > deadline_global - 40:
            break
        
        # Randomized greedy with different strategies
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                base = row[-1]
                # Try a window of candidates
                candidates = []
                for x in range(base + 1, min(base + 30, best_score + 1)):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        candidates.append(x)
                
                if not candidates:
                    # Extend window
                    for x in range(base + 1, best_score + 1):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            candidates.append(x)
                            break
                    if not candidates:
                        success = False
                        break
                
                # Pick with some randomness
                pick = rng.choice(candidates[:min(8, len(candidates))])
                row.append(pick)
                used |= {pick - a for a in row[:-1]}
            
            if not success:
                break
            rows.append(row)
        
        if success and len(rows) == n_rows and _validate(rows):
            score = _max_entry(rows)
            if score < best_score:
                best_score = score
                best_rows = copy.deepcopy(rows)
    
    # Phase 3: Constraint-based backtracking to find better solutions
    # Try to find solutions with lower max entry
    for target in range(best_score - 1, max(100, best_score - 20), -1):
        if time.time() > deadline_global - 20:
            break
        
        # Try a few times with different random seeds/orderings
        found = False
        for trial in range(5):
            if time.time() > deadline_global - 15:
                break
            
            rows = _backtrack_with_order(n_rows, k, target, rng, deadline_global - 10)
            if rows is not None and _validate(rows):
                score = _max_entry(rows)
                if score < best_score:
                    best_score = score
                    best_rows = copy.deepcopy(rows)
                found = True
                break
        
        if found:
            # Continue trying lower targets
            continue
        else:
            # If we couldn't find at this target, try slightly higher
            break
    
    # Phase 4: Local search to polish
    if time.time() < deadline_global - 10:
        start = copy.deepcopy(best_rows)
        rows, score = _local_search(start, n_rows, k, rng, max_iter=5000, max_x=best_score + 5)
        if score < best_score and _validate(rows):
            best_score = score
            best_rows = copy.deepcopy(rows)
    
    # Final validation
    if not _validate(best_rows):
        fresh_rows = _greedy_all(n_rows, k)
        if _validate(fresh_rows):
            best_rows = fresh_rows
    
    return best_rows
