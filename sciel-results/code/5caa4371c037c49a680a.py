import time
import random
import heapq
import bisect
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 142.0  # 8s margin from 150s limit
    
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
        if not rows:
            return 0
        return max(r[-1] for r in rows)
    
    def greedy_construct(seed_offset=0, use_scoring=False, max_search=200):
        """Greedy construction with optional best-fit scoring."""
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                best_x = None
                best_score = float('inf')
                candidates = []
                
                search_limit = max_search
                while x <= row[-1] + search_limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        if use_scoring:
                            # Score based on how well it packs into used diffs
                            score = _packing_score(new_diffs, used, x)
                        else:
                            score = x
                        candidates.append((score, x))
                        if score < best_score:
                            best_score = score
                            best_x = x
                    x += 1
                
                if best_x is None:
                    # Fallback: linear scan
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            best_x = x
                            break
                        x += 1
                
                # Pick from top candidates with randomness for diversity
                if use_scoring and len(candidates) > 1:
                    candidates.sort(key=lambda t: t[0])
                    top_n = min(5, len(candidates))
                    best_x = candidates[rng.randint(0, top_n - 1)][1]
                
                row.append(best_x)
                used |= {best_x - a for a in row[:-1]}
            
            rows.append(row)
        
        return rows
    
    def _packing_score(new_diffs, used, x):
        """Score candidate x by how well its differences pack into used space."""
        if not used:
            return x * 1.0
        
        # For each new diff, find distance to nearest used diff
        total_dist = 0.0
        for d in new_diffs:
            min_dist = float('inf')
            # Check nearby used diffs
            for delta in range(1, 6):
                if (d - delta) in used:
                    min_dist = min(min_dist, delta)
                    break
                if (d + delta) in used:
                    min_dist = min(min_dist, delta)
                    break
            if min_dist == float('inf'):
                min_dist = 10.0  # Large penalty for isolated diff
        
        return total_dist / max(len(new_diffs), 1) + 0.02 * x
    
    def backtrack_row(used, max_val, row, depth, rng, deadline):
        """Build one row via backtracking within max_val."""
        if time.time() > deadline:
            return None
        if depth == k + 1:
            return row[:]
        
        last = row[-1] if row else 0
        start = last + 1
        
        # Generate candidates in order, but shuffle for diversity
        candidates = list(range(start, min(max_val + 1, start + 50)))
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
    
    def local_search_shrink(rows, deadline, rng, max_iterations=20000):
        """Local search: try to reduce the maximum entry by modifying individual elements."""
        rows = [list(r) for r in rows]
        best_scope = _scope(rows)
        used = _all_diffs(rows)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            current_scope = _scope(rows)
            
            # Find a row and position to try to reduce
            ri = rng.randint(0, len(rows) - 1)
            row = rows[ri]
            
            # Prioritize positions with large values
            if rng.random() < 0.5:
                pi = k  # Last position
            else:
                pi = rng.randint(1, k)
            
            old_val = row[pi]
            if old_val < current_scope:
                continue
            
            lo = row[pi - 1] + 1
            hi = old_val - 1
            if pi + 1 < len(row):
                hi = min(hi, row[pi + 1] - 1)
            
            if lo > hi:
                continue
            
            # Try to find a smaller valid value
            for new_val in range(lo, hi + 1):
                if new_val >= old_val:
                    break
                
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
                
                # Valid move
                row[pi] = new_val
                used -= old_diffs
                used |= new_diffs
                break
        
        return rows
    
    def simulated_annealing(rows, deadline, rng, max_iter=50000):
        """Simulated annealing to minimize the scope (largest entry)."""
        current = [list(r) for r in rows]
        current_scope = _scope(current)
        best = [list(r) for r in current]
        best_scope = current_scope
        used = _all_diffs(current)
        
        temp = max(current_scope * 0.1, 5.0)
        cooling = 0.9997
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
                else:
                    current_scope = new_scope
            
            temp *= cooling
        
        return best, best_scope
    
    # Phase 1: Generate diverse initial solutions
    best_rows = None
    best_scope = float('inf')
    
    # Plain greedy
    for seed in range(10):
        if time.time() > deadline - 40:
            break
        rows = greedy_construct(seed, use_scoring=False)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    # Scoring-based greedy
    for seed in range(10):
        if time.time() > deadline - 40:
            break
        rows = greedy_construct(seed, use_scoring=True)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    if best_rows is None:
        # Fallback
        best_rows = greedy_construct(0)
        best_scope = _scope(best_rows)
    
    # Phase 2: Local search shrink
    rng = random.Random(123)
    if time.time() < deadline - 30:
        polished = local_search_shrink(best_rows, deadline - 25, rng, max_iterations=15000)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 20:
        sa_rows, sa_scope = simulated_annealing(best_rows, deadline - 15, rng, max_iter=40000)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 4: Backtracking at progressively lower scopes
    current_max = best_scope
    lower_bound = max(n_rows * k * (k + 1) // 2, 50)
    
    while time.time() < deadline - 10:
        target_scope = current_max - 1
        
        if target_scope < lower_bound:
            break
        
        rng = random.Random(int(time.time() * 1000) % 10000)
        result = backtrack_construct(target_scope, n_rows, rng, deadline - 10)
        if result is not None:
            best_rows = result
            best_scope = target_scope
            current_max = target_scope
            
            # Quick polish after backtracking success
            if time.time() < deadline - 10:
                polished = local_search_shrink(best_rows, deadline - 8, rng, max_iterations=5000)
                polished_scope = _scope(polished)
                if polished_scope < best_scope:
                    best_rows = polished
                    best_scope = polished_scope
                    current_max = polished_scope
        else:
            current_max = target_scope
    
    # Phase 5: Final polish
    if time.time() < deadline - 5:
        rng = random.Random(999)
        polished = local_search_shrink(best_rows, deadline - 3, rng, max_iterations=10000)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Final validation
    if not _valid(best_rows):
        # This should not happen, but as a safety net
        best_rows = greedy_construct(0)
    
    return best_rows
