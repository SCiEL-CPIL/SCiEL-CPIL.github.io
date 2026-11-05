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
    
    def greedy_construct(seed_offset=0):
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        
        return rows
    
    def greedy_construct_randomized(seed_offset=0):
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                max_step = max(1, int((k + 1) * 2.5))
                found = False
                for _attempt in range(30):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
                row.append(x)
                used |= new_diffs
            rows.append(row)
        
        return rows
    
    def backtrack_row(used, max_val, row, depth, rng, deadline):
        """Build one row via backtracking within max_val."""
        if time.time() > deadline:
            return None
        if depth == k + 1:
            return row[:]
        
        last = row[-1] if row else 0
        start = last + 1
        
        candidates = list(range(start, max_val + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            
            row.append(x)
            used_copy = used | new_diffs
            
            result = backtrack_row(used_copy, max_val, row, depth + 1, rng, deadline)
            if result is not None:
                return result
            
            row.pop()
        
        return None
    
    def backtrack_construct(max_val, num_rows, rng, deadline):
        """Try to construct all rows within max_val using backtracking."""
        used = set()
        rows = []
        
        for i in range(num_rows):
            if time.time() > deadline:
                return None
            
            row = [0]
            result = backtrack_row(used, max_val, row, 1, rng, deadline)
            if result is None:
                return None
            
            rows.append(result)
            used |= _row_diffs(result)
        
        return rows
    
    def local_search_shrink(rows, deadline, rng):
        """Local search: try to reduce the maximum entry by modifying individual elements.
        Works at the difference level: for each element position, try alternative values.
        Precomputes per-row diff sets for faster conflict checking."""
        rows = [list(r) for r in rows]
        best_scope = _scope(rows)
        used = _all_diffs(rows)
        row_diffs = [_row_diffs(r) for r in rows]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_scope = _scope(rows)
            
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
                        
                        other_diffs = used - old_diffs
                        conflict = new_diffs & other_diffs
                        if conflict:
                            continue
                        
                        row[pos] = new_val
                        used -= old_diffs
                        used |= new_diffs
                        row_diffs[ri] = new_diffs | (row_diffs[ri] - old_diffs)
                        improved = True
                        break
        
        return rows
    
    def local_search_swap(rows, deadline, rng, iterations=10000):
        """Local search: try to improve rows by moving individual elements.
        Uses precomputed diff sets for fast conflict checking."""
        best = [r[:] for r in rows]
        best_scope = _scope(best)
        used = _all_diffs(best)
        row_diffs = [_row_diffs(r) for r in best]
        
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
            
            other_diffs = used - old_diffs
            if new_diffs & other_diffs:
                continue
            
            if pi > 0 and new_val <= row[pi - 1]:
                continue
            if pi < k and new_val >= row[pi + 1]:
                continue
            
            best[ri][pi] = new_val
            used -= old_diffs
            used |= new_diffs
            row_diffs[ri] = new_diffs | (row_diffs[ri] - old_diffs)
            
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
        row_diffs = [_row_diffs(r) for r in current]
        
        temp = max(current_scope * 0.1, 5.0)
        cooling = 0.9995
        min_temp = 0.01
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if temp < min_temp:
                break
            
            ri = rng.randint(0, len(current) - 1)
            pi = rng.randint(1, k)
            
            row = current[ri]
            old_val = row[pi]
            
            lo = row[pi - 1] + 1
            hi = min(current_scope, row[pi + 1] - 1) if pi < k else current_scope
            
            if lo >= hi:
                continue
            
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            
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
            
            other_diffs = used - old_diffs
            if new_diffs & other_diffs:
                continue
            
            if pi > 0 and new_val <= row[pi - 1]:
                continue
            if pi < k and new_val >= row[pi + 1]:
                continue
            
            row[pi] = new_val
            used -= old_diffs
            used |= new_diffs
            row_diffs[ri] = new_diffs | (row_diffs[ri] - old_diffs)
            
            new_scope = max(max(r) for r in current)
            
            if new_scope <= current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [list(r) for r in current]
            else:
                delta = new_scope - current_scope
                prob = 1.0 - min(1.0, delta / (temp + 1e-9))
                if rng.random() > prob:
                    row[pi] = old_val
                    used -= new_diffs
                    used |= old_diffs
                    row_diffs[ri] = old_diffs | (row_diffs[ri] - new_diffs)
                else:
                    current_scope = new_scope
            
            temp *= cooling
        
        return best, best_scope
    
    # Start with greedy constructions
    best_rows = greedy_construct(0)
    best_scope = _scope(best_rows)
    
    # Try more greedy variants with different seeds
    for seed in range(15):
        if time.time() > deadline - 30:
            break
        rows = greedy_construct(seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
    
    # Try randomized greedy as fallback
    for seed in range(8):
        if time.time() > deadline - 30:
            break
        rows = greedy_construct_randomized(seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
    
    # Apply local search shrink
    rng = random.Random(123)
    if time.time() < deadline - 15:
        polished = local_search_shrink(best_rows, deadline - 10, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Apply swap-based local search
    if time.time() < deadline - 15:
        polished, polished_scope = local_search_swap(best_rows, deadline - 10, rng, iterations=8000)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Simulated annealing
    if time.time() < deadline - 15:
        sa_rows, sa_scope = simulated_annealing(best_rows, deadline - 10, rng, max_iter=30000)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Try backtracking at progressively lower scopes
    current_max = best_scope
    lower_bound = max(n_rows * k * (k + 1) // 2, 50)
    
    while time.time() < deadline - 8:
        target_scope = current_max - 1
        
        if target_scope < lower_bound:
            break
        
        rng = random.Random(int(time.time() * 1000) % 10000)
        result = backtrack_construct(target_scope, n_rows, rng, deadline - 8)
        if result is not None:
            best_rows = result
            best_scope = target_scope
            current_max = target_scope
            
            if time.time() < deadline - 8:
                polished = local_search_shrink(best_rows, deadline - 8, rng)
                polished_scope = _scope(polished)
                if polished_scope < best_scope:
                    best_rows = polished
                    best_scope = polished_scope
                    current_max = polished_scope
        else:
            current_max = target_scope
    
    # Final polish
    if time.time() < deadline - 3:
        rng = random.Random(999)
        polished = local_search_shrink(best_rows, deadline - 3, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    return best_rows
