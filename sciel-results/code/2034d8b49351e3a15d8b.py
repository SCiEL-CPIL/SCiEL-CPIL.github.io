import time
import random
import sys

# EVOLVE-BLOCK-START

def _sidon_check(a):
    """Check if sorted list a is a Sidon set (all positive differences distinct)."""
    n = len(a)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True, diffs

def _covers_1_to_30(diffs):
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def _backtrack_search(deadline):
    """Backtracking search for minimal-diameter Sidon set covering 1..30."""
    best = [None]
    best_diam = [float('inf')]
    
    def rec(a, used_diffs, missing, idx):
        if time.time() > deadline:
            return
        # Check if we already cover 1..30
        if _covers_1_to_30(used_diffs):
            diam = a[-1]
            if diam < best_diam[0]:
                best_diam[0] = diam
                best[0] = list(a)
            return
        # Prune: if current max already exceeds best, stop
        if a[-1] >= best_diam[0]:
            return
        # Try to add elements
        # We need to cover missing differences
        # Next element must be > a[-1]
        # Lower bound: we need at least len(missing) more diffs from new elements
        # With k remaining elements, max new diffs = k*(k+1)/2 + k*len(a)
        # Simple: try candidate positions
        
        start = a[-1] + 1
        # Upper bound for next element
        upper = min(best_diam[0] - 1, 200)  # search up to 200 initially
        for x in range(start, upper + 1):
            if time.time() > deadline:
                return
            # Check if adding x creates duplicate differences
            new_diffs = set()
            ok = True
            for v in a:
                d = x - v
                if d in used_diffs or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            # Also check differences among existing elements are fine (they are)
            new_used = used_diffs | new_diffs
            a.append(x)
            rec(a, new_used, missing, idx + 1)
            a.pop()
    
    # Start with [0]
    rec([0], set(), set(range(1, 31)), 0)
    return best[0], best_diam[0]

def _randomized_search(deadline):
    """Randomized local search for Sidon set covering 1..30."""
    best = [None]
    best_diam = [float('inf')]
    rng = random.Random(42)
    
    def try_set(a):
        a = sorted(set(a))
        if 0 not in a:
            return
        if len(a) > 32:
            return
        n = len(a)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = a[j] - a[i]
                if d in diffs:
                    return
                diffs.add(d)
        if _covers_1_to_30(diffs):
            diam = a[-1]
            if diam < best_diam[0]:
                best_diam[0] = diam
                best[0] = list(a)
    
    # Start with random sets and improve
    for restart in range(200):
        if time.time() > deadline:
            break
        # Generate a random Sidon-like set
        # Start with 0, add elements greedily
        a = [0]
        used = set()
        for _ in range(8):  # try to get 9 elements
            # Pick a random small number
            candidates = []
            for x in range(1, 100):
                ok = True
                for v in a:
                    d = x - v
                    if d <= 0:
                        continue
                    if d in used:
                        ok = False
                        break
                if ok:
                    candidates.append(x)
            if not candidates:
                break
            # Pick randomly from candidates, weighted toward small
            x = candidates[rng.randint(0, min(len(candidates)-1, 20))]
            a.append(x)
            for v in a[:-1]:
                used.add(x - v)
        if len(a) < 9:
            continue
        try_set(a)
        
        # Local improvement: try to reduce max element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            a_sorted = sorted(a)
            # Try removing the largest element and replacing with something smaller
            if len(a_sorted) > 9:
                last = a_sorted[-1]
                for x in range(1, last):
                    if x in a_sorted:
                        continue
                    trial = sorted(a_sorted[:-1] + [x])
                    # Check Sidon
                    n = len(trial)
                    diffs = set()
                    ok = True
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = trial[j] - trial[i]
                            if d in diffs:
                                ok = False
                                break
                            diffs.add(d)
                    if ok and _covers_1_to_30(diffs):
                        a = trial
                        improved = True
                        break
            # Try swapping any element for a smaller one
            for i in range(len(a_sorted) - 1, -1, -1):
                if i == 0:
                    continue  # keep 0
                old = a_sorted[i]
                for x in range(1, old):
                    if x in a_sorted:
                        continue
                    trial = sorted(a_sorted[:i] + [x] + a_sorted[i+1:])
                    n = len(trial)
                    diffs = set()
                    ok = True
                    for j in range(n):
                        for k in range(j + 1, n):
                            d = trial[k] - trial[j]
                            if d in diffs:
                                ok = False
                                break
                            diffs.add(d)
                    if ok and _covers_1_to_30(diffs):
                        a = trial
                        improved = True
                        break
                if improved:
                    break
    
    return best[0], best_diam[0]

def _try_known_constructions():
    """Try known perfect difference set constructions."""
    results = []
    
    # Try to find size-9 Sidon sets with small diameter
    # Use the fact that perfect difference sets mod n^2-n+1 exist
    # For n=9, v=73
    # Try various shifts and reflections
    
    for n in range(9, 13):
        v = n * n - n + 1
        # Try to construct a perfect difference set mod v
        # Simple construction: {0, 1, n, n+1, 2n, ...} doesn't work directly
        # Use quadratic residues or other methods
        pass
    
    return results

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: Backtracking search with tight bounds
    result, diam = _backtrack_search(deadline - 60)
    if result is not None:
        best_set = result
        best_diam = diam
    
    # Phase 2: Randomized search
    if time.time() < deadline - 30:
        result, diam = _randomized_search(deadline - 30)
        if result is not None and diam < best_diam:
            best_set = result
            best_diam = diam
    
    # Phase 3: If we have a good result, try to improve it further
    if best_set is not None and time.time() < deadline - 10:
        # Local search on the best set found
        a = sorted(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            a_sorted = sorted(a)
            # Try removing and re-adding elements
            for i in range(len(a_sorted) - 1, 0, -1):
                if time.time() > deadline - 5:
                    break
                old = a_sorted[i]
                # Try replacing with smaller value
                for x in range(1, old):
                    if x in a_sorted:
                        continue
                    trial = sorted(a_sorted[:i] + [x] + a_sorted[i+1:])
                    n = len(trial)
                    diffs = set()
                    ok = True
                    for j in range(n):
                        for k in range(j + 1, n):
                            d = trial[k] - trial[j]
                            if d in diffs:
                                ok = False
                                break
                            diffs.add(d)
                    if ok and _covers_1_to_30(diffs):
                        diam = trial[-1]
                        if diam < best_diam:
                            a = trial
                            best_set = trial
                            best_diam = diam
                            improved = True
                            break
                if improved:
                    break
    
    if best_set is None:
        # Fallback: return a valid set even if not optimal
        # Use the greedy approach from the original
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        best_set = a
        best_diam = max(a)
    
    return best_set

# EVOLVE-BLOCK-END
