import time
import random
import sys

# EVOLVE-BLOCK-START

TARGET = 30
TARGET_MASK = sum(1 << d for d in range(1, TARGET + 1))

def _check_sidon_mask(a, used_mask):
    """Check if adding elements to set a keeps Sidon property. Returns (ok, new_used_mask, new_covered_mask)."""
    n = len(a)
    new_used = used_mask
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            bit = 1 << d
            if new_used & bit:
                return False, 0, 0
            new_used |= bit
    covered = new_used & TARGET_MASK
    return True, new_used, covered

def _add_valid(a, x, used_mask):
    """Check if x can be added to sorted set a with given used_mask. Returns (ok, new_used_mask)."""
    new_used = used_mask
    for v in a:
        d = x - v
        if d <= 0:
            continue
        bit = 1 << d
        if new_used & bit:
            return False, 0
        new_used |= bit
    return True, new_used

def _covers_target(covered_mask):
    return (covered_mask & TARGET_MASK) == TARGET_MASK

def _backtrack_search(deadline, max_diam=120):
    """Backtracking search for minimal diameter Sidon set covering diffs 1..30."""
    best_diam = [float('inf')]
    best_set = [None]
    
    def rec(S, used_mask, current_max, depth):
        if time.time() > deadline:
            return
        if current_max >= best_diam[0]:
            return
        if _covers_target(used_mask):
            if current_max < best_diam[0]:
                best_diam[0] = current_max
                best_set[0] = list(S)
            return
        n = len(S)
        if n >= 32:
            return
        
        # Pruning: check if we can still cover all required diffs
        missing = TARGET_MASK & ~used_mask
        missing_count = bin(missing).count('1')
        remaining_slots = 32 - n
        # Each new element can create at most n new diffs
        if remaining_slots * n < missing_count:
            return
        
        start = S[-1] + 1 if S else 1
        upper = min(best_diam[0] - 1, max_diam)
        if upper < start:
            return
        
        # Try candidates in order of usefulness
        candidates = list(range(start, upper + 1))
        # Sort by how many new target diffs they cover
        def sort_key(x):
            new_covered = 0
            for v in S:
                d = x - v
                if d > 0 and d <= TARGET and not (used_mask & (1 << d)):
                    new_covered += 1
            return -new_covered  # More coverage first
        
        candidates.sort(key=sort_key)
        
        for x in candidates:
            if time.time() > deadline:
                return
            ok, new_used = _add_valid(S, x, used_mask)
            if not ok:
                continue
            S.append(x)
            rec(S, new_used, x, depth + 1)
            S.pop()
    
    rec([0], 0, 0, 0)
    return best_set[0], best_diam[0]

def _local_search_fast(a, deadline):
    """Fast local search using incremental Sidon checks."""
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    rng = random.Random(123)
    
    # Precompute used mask
    used_mask = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            used_mask |= 1 << (a[j] - a[i])
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        found_improvement = False
        
        # Try reducing the largest element
        last_idx = n - 1
        old_val = a[last_idx]
        
        for new_val in range(1, old_val):
            if new_val in a:
                continue
            # Check if replacing a[last_idx] with new_val is valid
            # Remove old diffs involving last_idx
            temp_used = 0
            valid = True
            for i in range(last_idx):
                d1 = a[i] - 0 if i == 0 else a[i] - a[0]
                # This is getting complex, let's just do full check
                pass
            
            # Simpler: build trial and check
            trial = a[:last_idx] + [new_val]
            trial = sorted(trial)
            tm = 0
            ok = True
            tn = len(trial)
            for i in range(tn):
                for j in range(i + 1, tn):
                    d = trial[j] - trial[i]
                    bit = 1 << d
                    if tm & bit:
                        ok = False
                        break
                    tm |= bit
                if not ok:
                    break
            if ok and _covers_target(tm):
                a = trial
                best = list(a)
                best_diam = a[-1]
                used_mask = tm
                found_improvement = True
                break
        
        if not found_improvement:
            # Try replacing other elements
            for i in range(n - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = a[i]
                for new_val in range(1, old_val):
                    if new_val in a:
                        continue
                    trial = a[:i] + [new_val] + a[i+1:]
                    trial = sorted(trial)
                    tm = 0
                    ok = True
                    tn = len(trial)
                    for ii in range(tn):
                        for jj in range(ii + 1, tn):
                            d = trial[jj] - trial[ii]
                            bit = 1 << d
                            if tm & bit:
                                ok = False
                                break
                            tm |= bit
                        if not ok:
                            break
                    if ok and _covers_target(tm) and trial[-1] < best_diam:
                        a = trial
                        best = list(a)
                        best_diam = a[-1]
                        used_mask = tm
                        found_improvement = True
                        break
                if found_improvement:
                    break
        
        if not found_improvement and rng.random() < 0.02:
            # Random perturbation
            idx = rng.randint(1, n - 1)
            new_val = rng.randint(1, best_diam)
            if new_val not in a:
                trial = list(a)
                trial[idx] = new_val
                trial = sorted(trial)
                tm = 0
                ok = True
                tn = len(trial)
                for ii in range(tn):
                    for jj in range(ii + 1, tn):
                        d = trial[jj] - trial[ii]
                        bit = 1 << d
                        if tm & bit:
                            ok = False
                            break
                        tm |= bit
                    if not ok:
                        break
                if ok and _covers_target(tm) and trial[-1] < best_diam:
                    a = trial
                    best = list(a)
                    best_diam = a[-1]
                    used_mask = tm
    
    return best, best_diam

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: Backtracking search with increasing diameter bounds
    for max_d in [40, 50, 60, 70, 80, 100, 120]:
        if time.time() > deadline - 10:
            break
        result, diam = _backtrack_search(deadline - 5, max_d)
        if result is not None and diam < best_diam:
            best_set = result
            best_diam = diam
        if best_diam <= max_d:
            break
    
    # Phase 2: Local search to improve
    if best_set is not None and time.time() < deadline - 5:
        improved, improved_diam = _local_search_fast(best_set, deadline - 3)
        if improved_diam < best_diam:
            best_set = improved
            best_diam = improved_diam
    
    # Fallback: simple greedy
    if best_set is None:
        a = [0]
        used_mask = 0
        for _ in range(30):
            if time.time() > deadline - 1:
                break
            best_x = None
            best_cover = -1
            for x in range(1, 200):
                ok, new_used = _add_valid(a, x, used_mask)
                if not ok:
                    continue
                cover = bin(new_used & TARGET_MASK).count('1')
                if cover > best_cover:
                    best_cover = cover
                    best_x = x
            if best_x is None:
                break
            a.append(best_x)
            a.sort()
            ok, new_used = _add_valid(a[:-1], a[-1], 0)
            # Recompute used mask
            used_mask = 0
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    used_mask |= 1 << (a[j] - a[i])
            if _covers_target(used_mask):
                break
        best_set = a
        best_diam = max(a)
    
    return best_set

# EVOLVE-BLOCK-END
