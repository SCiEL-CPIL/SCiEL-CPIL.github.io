import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_sidon(a):
    """Check if sorted list a is a Sidon set. Returns (bool, set_of_diffs)."""
    n = len(a)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False, diffs
            diffs.add(d)
    return True, diffs

def _covers_1_to_30(diffs):
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def _known_constructions():
    """Generate candidate sets from known perfect difference set constructions."""
    results = []
    
    # Known perfect difference sets of size 9
    # {0, 1, 4, 10, 12, 28, 32, 45, 63} - diameter 63
    results.append([0, 1, 4, 10, 12, 28, 32, 45, 63])
    
    # Try to find better ones by local search
    return results

def _local_search(a, deadline, max_iter=10000):
    """Local search to minimize diameter while maintaining Sidon property and coverage."""
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    rng = random.Random(42)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(a)
        if n < 2:
            break
        
        # Try to reduce the largest element first
        found = False
        last_idx = n - 1
        old_val = a[last_idx]
        
        for new_val in range(1, old_val):
            if new_val in a:
                continue
            trial = a[:last_idx] + [new_val]
            trial = sorted(trial)
            ok, diffs = _is_sidon(trial)
            if ok and _covers_1_to_30(diffs):
                a = trial
                best = list(a)
                best_diam = a[-1]
                found = True
                break
        
        if not found:
            # Try replacing any non-zero element
            for i in range(n - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = a[i]
                for new_val in range(1, old_val):
                    if new_val in a:
                        continue
                    trial = a[:i] + [new_val] + a[i+1:]
                    trial = sorted(trial)
                    ok, diffs = _is_sidon(trial)
                    if ok and _covers_1_to_30(diffs):
                        diam = trial[-1]
                        if diam < best_diam:
                            a = trial
                            best = list(a)
                            best_diam = diam
                            found = True
                            break
                if found:
                    break
        
        if not found:
            # Random perturbation
            if rng.random() < 0.05:
                new_a = list(a)
                idx = rng.randint(1, n - 1)
                new_val = rng.randint(1, best_diam)
                if new_val not in new_a:
                    new_a[idx] = new_val
                    new_a = sorted(new_a)
                    ok, diffs = _is_sidon(new_a)
                    if ok and _covers_1_to_30(diffs):
                        if new_a[-1] < best_diam:
                            a = new_a
                            best = list(a)
                            best_diam = a[-1]
    
    return best, best_diam

def _backtrack_minimal(deadline):
    """Backtracking to find minimal diameter Sidon set covering 1..30."""
    best = [None]
    best_diam = [float('inf')]
    
    def rec(a, used_diffs, current_max):
        if time.time() > deadline:
            return
        
        if _covers_1_to_30(used_diffs):
            if current_max < best_diam[0]:
                best_diam[0] = current_max
                best[0] = list(a)
            return
        
        if current_max >= best_diam[0]:
            return
        
        n = len(a)
        start = a[-1] + 1 if a else 1
        upper = min(best_diam[0] - 1, 80)
        
        for x in range(start, upper + 1):
            if time.time() > deadline:
                return
            
            new_diffs = set()
            ok = True
            for v in a:
                d = x - v
                if d <= 0:
                    continue
                if d in used_diffs or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            
            if not ok:
                continue
            
            new_used = used_diffs | new_diffs
            a.append(x)
            rec(a, new_used, x)
            a.pop()
    
    rec([0], set(), 0)
    return best[0], best_diam[0]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: Try known constructions
    for candidate in _known_constructions():
        if time.time() > deadline:
            break
        ok, diffs = _is_sidon(candidate)
        if ok and _covers_1_to_30(diffs):
            diam = candidate[-1]
            if diam < best_diam:
                best_diam = diam
                best_set = list(candidate)
    
    # Phase 2: Backtracking search
    if time.time() < deadline - 40:
        result, diam = _backtrack_minimal(deadline - 40)
        if result is not None and diam < best_diam:
            best_set = result
            best_diam = diam
    
    # Phase 3: Local search to improve
    if best_set is not None and time.time() < deadline - 5:
        improved, improved_diam = _local_search(best_set, deadline - 5)
        if improved_diam < best_diam:
            best_set = improved
            best_diam = improved_diam
    
    # Fallback: greedy construction
    if best_set is None:
        a = [0]
        used = set()
        for d in range(1, 31):
            if d in used:
                continue
            found = False
            for x in range(1, 200):
                if x in a:
                    continue
                new_diffs = set()
                ok = True
                for v in a:
                    diff = abs(x - v)
                    if diff in used or diff in new_diffs:
                        ok = False
                        break
                    new_diffs.add(diff)
                if ok:
                    a.append(x)
                    a.sort()
                    used.update(new_diffs)
                    found = True
                    break
            if not found:
                x = max(a) + d + 1
                while True:
                    new_diffs = set()
                    ok = True
                    for v in a:
                        diff = x - v
                        if diff in used or diff in new_diffs:
                            ok = False
                            break
                        new_diffs.add(diff)
                    if ok:
                        a.append(x)
                        a.sort()
                        used.update(new_diffs)
                        break
                    x += 1
        best_set = a
        best_diam = max(a)
    
    return best_set

# EVOLVE-BLOCK-END
