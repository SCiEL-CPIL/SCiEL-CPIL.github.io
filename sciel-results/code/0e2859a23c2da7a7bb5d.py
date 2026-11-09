"""ILP + backtracking approach for truncated perfect difference sets."""
import time
import random
import numpy as np

def _validate(a):
    """Check Sidon + all diffs 1..30 present. Return max element or inf."""
    a = sorted(set(a))
    if 0 not in a:
        return float('inf')
    n = len(a)
    if n > 32:
        return float('inf')
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return float('inf')
            diffs.add(d)
    if not all(d in diffs for d in range(1, 31)):
        return float('inf')
    return max(a)

def _backtrack_construct(deadline):
    """Backtracking with forward checking: build set element by element,
    ensuring we can still cover all required differences."""
    best = [float('inf')]
    
    def solve(current, used_diffs, remaining_diffs, max_elem_bound):
        if time.time() > deadline:
            return
        if len(current) > 32:
            return
        # Prune: if max element already exceeds best, stop
        if current and max(current) >= best[0]:
            return
        # Check if all required diffs are covered
        if not remaining_diffs:
            # Valid solution
            if max(current) < best[0]:
                best[0] = max(current)
                best_set[:] = list(current)
            return
        
        # If we've used all 32 elements but still have remaining diffs, fail
        if len(current) == 32 and remaining_diffs:
            return
        
        # Forward checking: can we still cover remaining_diffs with (32 - len(current)) more elements?
        # Each new element can create at most len(current) new differences (with existing elements)
        # plus differences with other new elements. Rough bound:
        slots_left = 32 - len(current)
        max_new_diffs = slots_left * len(current) + slots_left * (slots_left - 1) // 2
        if len(remaining_diffs) > max_new_diffs:
            return
        
        # Try adding a new element
        if not current:
            candidates = [0]
        else:
            # Generate candidate positions
            max_val = max(current)
            # Candidates: values that create useful differences
            candidates = set()
            for d in remaining_diffs:
                for v in current:
                    c1 = v + d
                    c2 = v - d
                    if 0 <= c1 < max_elem_bound:
                        candidates.add(c1)
                    if 0 <= c2 < max_elem_bound:
                        candidates.add(c2)
            # Also try some random candidates near existing elements
            for _ in range(20):
                base = random.choice(current)
                offset = random.randint(1, 50)
                c = base + offset
                if 0 <= c < max_elem_bound:
                    candidates.add(c)
            candidates = sorted(candidates)
        
        # Sort candidates by how many remaining diffs they cover (greedy heuristic)
        def coverage_count(cand):
            count = 0
            for d in remaining_diffs:
                for v in current:
                    if cand - v == d or v - cand == d:
                        count += 1
                        break
            return count
        
        candidates.sort(key=coverage_count, reverse=True)
        
        for cand in candidates:
            if cand in current:
                continue
            # Check Sidon constraint
            trial = current + [cand]
            trial.sort()
            valid = True
            seen = set()
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    d = trial[j] - trial[i]
                    if d in seen:
                        valid = False
                        break
                    seen.add(d)
                if not valid:
                    break
            if not valid:
                continue
            
            new_used = used_diffs | seen
            new_remaining = remaining_diffs - seen
            
            solve(trial, new_used, new_remaining, max_elem_bound)
            if best[0] < max_elem_bound:
                max_elem_bound = best[0]
            if time.time() > deadline:
                return
    
    best_set = []
    # Start with small bound, increase if needed
    for bound in [100, 200, 500, 1000, 5000, 100000]:
        solve([0], {0}, set(range(1, 31)), bound)
        if best[0] < float('inf'):
            break
        if time.time() > deadline:
            break
    
    if best[0] < float('inf'):
        return best_set
    return None

def _ilp_construct(deadline):
    """Use scipy MILP to find minimum diameter Sidon set covering diffs 1..30."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None
    
    # We'll use a simpler approach: binary search on max element M,
    # and for each M, check if a valid Sidon set exists in [0, M].
    # This is an NP-hard problem, so ILP may not solve it directly.
    # Instead, use ILP to find a set of positions.
    
    # Alternative: use the fact that we need at most 32 elements.
    # Encode: x[i][v] = 1 if element i is at position v.
    # This is too large for direct ILP.
    
    # Better: use a heuristic ILP where we fix the number of elements n
    # and find positions that minimize max element while satisfying Sidon.
    
    # For now, fall back to a smarter backtracking with better pruning.
    return None

def _greedy_with_backtrack(deadline):
    """Hybrid: greedy construction with backtracking when stuck."""
    best = [float('inf')]
    best_set = []
    
    def build(current, remaining_diffs):
        if time.time() > deadline:
            return
        if len(current) > 32:
            return
        if max(current) >= best[0]:
            return
        if not remaining_diffs:
            if max(current) < best[0]:
                best[0] = max(current)
                best_set[:] = list(current)
            return
        
        slots_left = 32 - len(current)
        # Each new element adds at most len(current) new diffs with existing
        if len(remaining_diffs) > slots_left * len(current) + slots_left * (slots_left - 1) // 2:
            return
        
        # Generate candidates
        max_val = max(current) if current else 0
        candidates = set()
        for d in remaining_diffs:
            for v in current:
                c1 = v + d
                c2 = v - d
                if 0 <= c1 < best[0]:
                    candidates.add(c1)
                if 0 <= c2 < best[0]:
                    candidates.add(c2)
        
        # Add some random candidates
        for _ in range(30):
            base = random.choice(current) if current else 0
            offset = random.randint(1, max(1, int(best[0] * 0.3)))
            c = base + offset
            if 0 <= c < best[0]:
                candidates.add(c)
        
        candidates = sorted(candidates)
        
        # Score by coverage
        def coverage(cand):
            count = 0
            for d in remaining_diffs:
                for v in current:
                    if abs(cand - v) == d:
                        count += 1
                        break
            return count
        
        candidates.sort(key=coverage, reverse=True)
        
        for cand in candidates[:50]:  # Limit branching
            if cand in current:
                continue
            trial = current + [cand]
            trial.sort()
            # Check Sidon
            seen = set()
            valid = True
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    d = trial[j] - trial[i]
                    if d in seen:
                        valid = False
                        break
                    seen.add(d)
                if not valid:
                    break
            if not valid:
                continue
            
            new_remaining = remaining_diffs - seen
            build(trial, new_remaining)
            if best[0] < max_val:
                # Tighten bound
                pass
            if time.time() > deadline:
                return
    
    # Initial bound
    initial_bound = 200
    build([0], set(range(1, 31)))
    
    if best[0] < float('inf'):
        return best_set
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_set = None
    best_score = float('inf')
    
    # Try backtracking-based construction
    result = _greedy_with_backtrack(deadline)
    if result is not None:
        s = _validate(result)
        if s < best_score:
            best_score = s
            best_set = result
    
    # If time remains, try more random restarts
    if best_set is not None and time.time() < deadline - 10:
        for seed in range(20):
            if time.time() > deadline:
                break
            random.seed(seed)
            result = _greedy_with_backtrack(deadline)
            if result is not None:
                s = _validate(result)
                if s < best_score:
                    best_score = s
                    best_set = result
    
    # Fallback: simple greedy if nothing found
    if best_set is None:
        a = [0]
        for d in range(1, 31):
            diffs = set()
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    diffs.add(a[j] - a[i])
            if d in diffs:
                continue
            found = False
            for v in a:
                for x in [v + d, v - d]:
                    if x < 0 or x in a:
                        continue
                    trial = sorted(a + [x])
                    seen = set()
                    valid = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            dd = trial[j] - trial[i]
                            if dd in seen:
                                valid = False
                                break
                            seen.add(dd)
                        if not valid:
                            break
                    if valid:
                        a = trial
                        found = True
                        break
                if found:
                    break
            if not found:
                # Add two elements
                y = max(a) + d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    seen = set()
                    valid = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            dd = trial[j] - trial[i]
                            if dd in seen:
                                valid = False
                                break
                            seen.add(dd)
                        if not valid:
                            break
                    if valid:
                        a = trial
                        break
                    y += 1
        s = _validate(a)
        if s < best_score:
            best_score = s
            best_set = a
    
    return best_set
