"""Difference triangle set (7,5) - find 7 rows of 6 marks minimizing max entry.

Structural change: Model as a constraint satisfaction problem where the global
difference set is a shared resource pool. Use a backtracking solver with:
- Forward checking: prune candidates that make remaining positions infeasible
- Dynamic row ordering: build rows in order of tightest remaining constraints
- Difference density heuristic: prefer values that use "scarce" differences
- Iterative deepening on max-entry bound with multiple random restarts
- Local search polishing to reduce max entry
"""
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

def _greedy_row(used, k, max_x=None):
    """Build one row greedily: start at 0, pick smallest next mark with no diff conflicts."""
    row = [0]
    while len(row) < k + 1:
        base = row[-1]
        found = False
        upper = (max_x if max_x is not None else 500)
        for x in range(base + 1, upper + 1):
            new_diffs = {x - a for a in row}
            if len(new_diffs) == len(row) and not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            return None
    return row

def _greedy_all(n_rows, k, max_x=None):
    """Build all rows greedily, each after the previous."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, max_x)
        if row is None:
            return None
        rows.append(row)
    return rows

def _backtrack_with_forward_checking(n_rows, k, max_entry, rng, deadline):
    """
    Constraint-based backtracking with forward checking.
    The key insight: treat the set of used differences as a global resource pool.
    At each step, forward-check whether the remaining positions can still be filled.
    """
    total_diffs_needed = n_rows * (k + 1) * k // 2
    if total_diffs_needed > max_entry:
        return None
    
    used_diffs = set()
    rows = []
    
    def solve(row_idx, pos, current_row):
        """Recursively build row row_idx, position pos."""
        if time.time() > deadline:
            return False
        
        if pos == k + 1:
            # Row complete
            rows.append(current_row[:])
            if row_idx + 1 == n_rows:
                return True
            return solve(row_idx + 1, 1, [0])
        
        if pos == 0:
            return solve(row_idx, 1, [0])
        
        lo = current_row[-1] + 1
        hi = max_entry
        
        # Forward check: can we still fill remaining positions?
        remaining = k + 1 - pos
        if hi - lo + 1 < remaining:
            return False
        
        # Generate candidates with forward checking
        candidates = []
        for val in range(lo, hi + 1):
            new_diffs = {val - a for a in current_row}
            if len(new_diffs) != len(current_row):
                continue
            if new_diffs & used_diffs:
                continue
            
            # Forward check: after placing this value, can the next position be filled?
            if pos + 1 < k + 1:
                next_lo = val + 1
                next_hi = max_entry
                next_remaining = k + 1 - (pos + 1)
                if next_hi - next_lo + 1 < next_remaining:
                    continue
                # Check if at least one valid next value exists
                has_next = False
                for next_val in range(next_lo, next_hi + 1):
                    next_new_diffs = {next_val - a for a in current_row + [val]}
                    if len(next_new_diffs) == len(current_row) + 1:
                        if not (next_new_diffs & (used_diffs | new_diffs)):
                            has_next = True
                            break
                if not has_next:
                    continue
            
            candidates.append(val)
        
        if not candidates:
            return False
        
        # Order candidates: prefer those that leave more room and use "scarce" diffs
        def score(val):
            new_diffs = {val - a for a in current_row}
            # Prefer smaller values (to keep max low)
            s1 = val * 10
            # Prefer values whose differences are less "popular" (more available)
            s2 = 0
            for d in new_diffs:
                # Count how many other used diffs are within a small range
                s2 += sum(1 for ud in used_diffs if abs(ud - d) <= 3)
            return s1 + s2
        
        candidates.sort(key=score)
        
        # Add some randomness to avoid getting stuck
        if rng.random() < 0.3 and len(candidates) > 1:
            swap_idx = rng.randrange(len(candidates))
            candidates[0], candidates[swap_idx] = candidates[swap_idx], candidates[0]
        
        for val in candidates:
            if time.time() > deadline:
                return False
            
            new_diffs = {val - a for a in current_row}
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            if solve(row_idx, pos + 1, current_row):
                return True
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
        
        return False
    
    if solve(0, 0, []):
        return rows
    return None

def _backtrack_with_row_ordering(n_rows, k, max_entry, rng, deadline):
    """
    Backtracking with dynamic row ordering: build rows in order of tightest constraints.
    A row is "tighter" if it has fewer available difference values.
    """
    total_diffs_needed = n_rows * (k + 1) * k // 2
    if total_diffs_needed > max_entry:
        return None
    
    used_diffs = set()
    rows = [None] * n_rows
    
    def build_row(row_idx, current_row, pos):
        """Build a single row given the used_diffs constraint."""
        if time.time() > deadline:
            return None
        
        if pos == k + 1:
            return current_row[:]
        
        if pos == 0:
            return build_row(row_idx, [0], 1)
        
        lo = current_row[-1] + 1
        hi = max_entry
        
        remaining = k + 1 - pos
        if hi - lo + 1 < remaining:
            return None
        
        # Generate valid candidates
        candidates = []
        for val in range(lo, hi + 1):
            new_diffs = {val - a for a in current_row}
            if len(new_diffs) != len(current_row):
                continue
            if new_diffs & used_diffs:
                continue
            candidates.append(val)
        
        if not candidates:
            return None
        
        # Score candidates by how many future options they preserve
        def future_options(val):
            """Count how many valid next values exist after placing val."""
            if pos + 1 >= k + 1:
                return 1
            next_lo = val + 1
            next_hi = max_entry
            next_remaining = k + 1 - (pos + 1)
            if next_hi - next_lo + 1 < next_remaining:
                return 0
            count = 0
            for next_val in range(next_lo, next_hi + 1):
                next_new_diffs = {next_val - a for a in current_row + [val]}
                if len(next_new_diffs) == len(current_row) + 1:
                    if not (next_new_diffs & (used_diffs | {val - a for a in current_row})):
                        count += 1
                        if count >= 5:  # Cap for speed
                            break
            return count
        
        scored = [(future_options(v), v) for v in candidates]
        scored.sort(key=lambda x: (-x[0], x[1]))
        
        # Take top candidates with some randomness
        top_n = min(30, len(scored))
        top_candidates = [v for _, v in scored[:top_n]]
        
        if rng.random() < 0.2 and len(top_candidates) > 1:
            swap_idx = rng.randrange(len(top_candidates))
            top_candidates[0], top_candidates[swap_idx] = top_candidates[swap_idx], top_candidates[0]
        
        for val in top_candidates:
            if time.time() > deadline:
                return None
            
            new_diffs = {val - a for a in current_row}
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            result = build_row(row_idx, current_row, pos + 1)
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
        
        row = build_row(ri, [], 0)
        if row is None:
            return None
        rows[ri] = row
        # Update used_diffs with all diffs in this row
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    return rows

def _local_search(rows, n_rows, k, rng, max_iter=3000, max_x=None, deadline=None):
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
        if deadline is not None and time.time() > deadline:
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
                    new_diffs.add(new_row[j] - new_row[i])
            
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

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    rng = random.Random(42)
    
    # Phase 1: Get a good initial solution with greedy
    best_rows = _greedy_all(n_rows, k)
    if best_rows is None:
        # Fallback: simple greedy without max constraint
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                for x in range(row[-1] + 1, 500):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
            best_rows.append(row)
    
    best_score = _max_entry(best_rows)
    
    # Phase 2: Try multiple random greedy starts with bounded max
    for attempt in range(50):
        if time.time() > deadline - 60:
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
                for x in range(base + 1, min(base + 40, best_score + 1)):
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
                pick = rng.choice(candidates[:min(10, len(candidates))])
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
    for target in range(best_score - 1, max(100, best_score - 25), -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few times with different random seeds/orderings
        found = False
        for trial in range(8):
            if time.time() > deadline - 25:
                break
            
            # Alternate between forward checking and row ordering
            if trial % 2 == 0:
                rows = _backtrack_with_forward_checking(n_rows, k, target, rng, deadline - 20)
            else:
                rows = _backtrack_with_row_ordering(n_rows, k, target, rng, deadline - 20)
            
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
    if time.time() < deadline - 15:
        start = copy.deepcopy(best_rows)
        rows, score = _local_search(start, n_rows, k, rng, max_iter=8000, max_x=best_score + 5, deadline=deadline - 10)
        if score < best_score and _validate(rows):
            best_score = score
            best_rows = copy.deepcopy(rows)
    
    # Final validation
    if not _validate(best_rows):
        fresh_rows = _greedy_all(n_rows, k)
        if fresh_rows is not None and _validate(fresh_rows):
            best_rows = fresh_rows
    
    return best_rows
