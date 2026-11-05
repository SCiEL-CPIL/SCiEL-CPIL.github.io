import time
import random
import heapq
import bisect

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 145  # 5s margin from 150s limit

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_diffs(rows):
        used = set()
        for row in rows:
            used |= _row_diffs(row)
        return used

    def _valid(rows):
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
        return max(max(r) for r in rows)

    def scarcity_score(x, row):
        """
        Score a candidate x by the 'scarcity' of the differences it uses.
        Small differences are more scarce (harder to fit later), so we prefer
        candidates that use larger differences (lower sum of 1/d).
        """
        cost = 0.0
        for a in row:
            d = x - a
            if d > 0:
                cost += 1.0 / d
        return cost

    def greedy_construct_scarcity(seed_offset=0, max_candidates=100):
        """
        Greedy construction using scarcity heuristic.
        """
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                candidates = []
                x = last + 1
                checked = 0
                
                while checked < max_candidates:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        # Calculate scarcity cost
                        cost = scarcity_score(x, row)
                        # Add a small penalty for x itself to keep it compact
                        # but scarcity is the primary driver
                        score = cost + x * 0.001 
                        candidates.append((score, cost, x))
                    x += 1
                    checked += 1
                
                if not candidates:
                    # Fallback: keep going until we find one
                    x = last + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append((0, 0, x))
                            break
                        x += 1
                
                # Sort by score (lowest scarcity cost is best)
                candidates.sort(key=lambda t: t[0])
                
                # Pick the best, or randomly from top 3 for diversity
                best_x = candidates[0][2]
                if len(candidates) > 1 and rng.random() < 0.2:
                    idx = rng.randint(0, min(2, len(candidates) - 1))
                    best_x = candidates[idx][2]
                
                row.append(best_x)
                new_diffs = {best_x - a for a in row[:-1]}
                used |= new_diffs
            rows.append(row)
        
        return rows

    def backtrack_row_limited(used, max_val, row, depth, rng, deadline, beam_width=5):
        """
        Build one row via limited backtracking.
        Uses scarcity heuristic to order candidates.
        """
        if time.time() > deadline:
            return None
        if depth == k + 1:
            return row[:]
        
        last = row[-1] if row else 0
        start = last + 1
        
        # Generate candidates up to max_val
        candidates = []
        for x in range(start, max_val + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            cost = scarcity_score(x, row)
            candidates.append((cost, x, new_diffs))
            
            # Limit candidate generation to avoid explosion
            if len(candidates) > 200:
                break
        
        if not candidates:
            return None
        
        # Sort by scarcity cost
        candidates.sort(key=lambda t: t[0])
        
        # Try top beam_width candidates
        for cost, x, new_diffs in candidates[:beam_width]:
            row.append(x)
            used_copy = used | new_diffs
            
            result = backtrack_row_limited(used_copy, max_val, row, depth + 1, rng, deadline, beam_width)
            if result is not None:
                return result
            
            row.pop()
            
            # Check time
            if time.time() > deadline:
                return None

        return None

    def construct_with_backtracking(max_val, num_rows, rng, deadline, beam_width=3):
        """
        Try to construct all rows within max_val using limited backtracking per row.
        """
        used = set()
        rows = []
        
        for i in range(num_rows):
            if time.time() > deadline:
                return None
            
            row = [0]
            # We don't know the exact max_val for the row, but it's bounded by max_val
            # We pass max_val as the hard limit
            result = backtrack_row_limited(used, max_val, row, 1, rng, deadline, beam_width)
            if result is None:
                return None
            
            rows.append(result)
            used |= _row_diffs(result)
        
        return rows

    def local_search_shrink(rows, deadline, rng):
        """Local search: try to reduce the maximum entry by modifying individual elements."""
        rows = [list(r) for r in rows]
        best_scope = _scope(rows)
        used = _all_diffs(rows)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_scope = _scope(rows)
            
            # Try to reduce each element that is at the current maximum
            for ri in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    if old_val < current_scope:
                        continue
                    
                    low = row[pos - 1] + 1
                    high = old_val - 1
                    if pos + 1 < len(row):
                        high = min(high, row[pos + 1] - 1)
                    
                    if low > high:
                        continue
                    
                    candidates = list(range(high, low - 1, -1))
                    rng.shuffle(candidates)
                    
                    for new_val in candidates:
                        if time.time() > deadline:
                            break
                        if new_val >= old_val:
                            continue
                        
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(old_val - row[i])
                        for i in range(pos + 1, len(row)):
                            old_diffs.add(row[i] - old_val)
                        
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for i in range(pos + 1, len(row)):
                            new_diffs.add(row[i] - new_val)
                        
                        # Check conflict: new_diffs must not intersect (used - old_diffs)
                        other_diffs = used - old_diffs
                        conflict = new_diffs & other_diffs
                        if conflict:
                            continue
                        
                        row[pos] = new_val
                        used -= old_diffs
                        used |= new_diffs
                        improved = True
                        break
        
        return rows

    def local_search_swap(rows, deadline, rng, iterations=10000):
        """Local search: try to improve rows by moving individual elements."""
        best = [r[:] for r in rows]
        best_scope = _scope(best)
        used = _all_diffs(best)
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
                
            ri = rng.randint(0, len(best) - 1)
            pi = rng.randint(1, k)
            
            old_val = best[ri][pi]
            
            if pi < k:
                lo = best[ri][pi - 1] + 1
                hi = min(best_scope, best[ri][pi + 1] - 1)
            else:
                lo = best[ri][pi - 1] + 1
                hi = best_scope
            
            if lo > hi:
                continue
            
            new_val = rng.randint(lo, hi)
            
            if new_val == old_val:
                continue
            
            # Compute old and new diffs for position pi in row ri
            row = best[ri]
            old_diffs = set()
            for i in range(pi):
                old_diffs.add(old_val - row[i])
            for i in range(pi + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pi):
                new_diffs.add(new_val - row[i])
            for i in range(pi + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            # Check conflict with other rows' diffs
            other_diffs = used - old_diffs
            if new_diffs & other_diffs:
                continue
            
            # Check monotonicity
            if pi > 0 and new_val <= row[pi - 1]:
                continue
            if pi < k and new_val >= row[pi + 1]:
                continue
            
            # Accept the move
            best[ri][pi] = new_val
            used -= old_diffs
            used |= new_diffs
            
            new_scope = max(max(r) for r in best)
            if new_scope < best_scope:
                best_scope = new_scope
        
        return best, best_scope

    def simulated_annealing(rows, deadline, rng, max_iter=50000):
        """Simulated annealing to minimize the scope (largest entry)."""
        current = [list(r) for r in rows]
        current_scope = _scope(current)
        best = [list(r) for r in current]
        best_scope = current_scope
        used = _all_diffs(current)
        
        temp = max(current_scope * 0.1, 5.0)
        cooling = 0.9995
        min_temp = 0.01
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if temp < min_temp:
                break
            
            # Pick a random row and position to perturb
            ri = rng.randint(0, len(current) - 1)
            pi = rng.randint(1, k)
            
            row = current[ri]
            old_val = row[pi]
            
            # Determine valid range for new value
            lo = row[pi - 1] + 1
            hi = min(current_scope, row[pi + 1] - 1) if pi < k else current_scope
            
            if lo >= hi:
                continue
            
            # Random new value
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            
            # Compute old and new diffs
            old_diffs = set()
            for i in range(pi):
                old_diffs.add(old_val - row[i])
            for i in range(pi + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pi):
                new_diffs.add(new_val - row[i])
            for i in range(pi + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            # Check validity
            other_diffs = used - old_diffs
            if new_diffs & other_diffs:
                continue
            
            # Check monotonicity
            if pi > 0 and new_val <= row[pi - 1]:
                continue
            if pi < k and new_val >= row[pi + 1]:
                continue
            
            # Apply the move
            row[pi] = new_val
            used -= old_diffs
            used |= new_diffs
            
            new_scope = max(max(r) for r in current)
            
            # Accept if better or with probability based on temperature
            if new_scope <= current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [list(r) for r in current]
            else:
                delta = new_scope - current_scope
                prob = 1.0 - min(1.0, delta / (temp + 1e-9))
                if rng.random() > prob:
                    # Reject: revert
                    row[pi] = old_val
                    used -= new_diffs
                    used |= old_diffs
                else:
                    current_scope = new_scope
            
            temp *= cooling
        
        return best, best_scope

    # Phase 1: Generate initial solutions using Scarcity Greedy
    best_rows = greedy_construct_scarcity(seed_offset=0)
    best_scope = _scope(best_rows)
    
    # Try more scarcity greedy variants
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        rows = greedy_construct_scarcity(seed_offset=seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope

    # Phase 2: Try limited backtracking to potentially find better initial structures
    # We try to construct with a target scope slightly lower than current best
    # This is a heuristic attempt to jump out of local minima
    current_max = best_scope
    # Try a few backtracking attempts with tighter constraints
    for bt_seed in range(5):
        if time.time() > deadline - 30:
            break
        # Target scope: try to fit in best_scope - 1 or similar
        # But backtracking is expensive. Let's just use it to verify or find slightly better.
        # Actually, let's use backtracking to construct from scratch with a target scope
        target_scope = best_scope - 1
        if target_scope < 50:
            target_scope = 50
            
        rng_bt = random.Random(100 + bt_seed)
        # Use a smaller beam width for speed
        result = construct_with_backtracking(target_scope, n_rows, rng_bt, deadline - 30, beam_width=2)
        if result is not None:
            scope = _scope(result)
            if scope < best_scope:
                best_rows = result
                best_scope = scope
                current_max = scope

    # Phase 3: Local Search Shrink
    rng = random.Random(123)
    if time.time() < deadline - 15:
        polished = local_search_shrink(best_rows, deadline - 10, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Phase 4: Swap-based Local Search
    if time.time() < deadline - 15:
        polished, polished_scope = local_search_swap(best_rows, deadline - 10, rng, iterations=8000)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Phase 5: Simulated Annealing
    if time.time() < deadline - 15:
        sa_rows, sa_scope = simulated_annealing(best_rows, deadline - 10, rng, max_iter=30000)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope

    # Final Polish
    if time.time() < deadline - 3:
        rng = random.Random(999)
        polished = local_search_shrink(best_rows, deadline - 3, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    return best_rows
