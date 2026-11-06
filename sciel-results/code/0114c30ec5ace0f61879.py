"""Construct a Sidon set containing 0 with all differences 1..30, minimizing max element."""
import time
import random

def _all_diffs(a):
    """Return list of all positive differences from sorted list a."""
    n = len(a)
    return [a[j] - a[i] for i in range(n) for j in range(i + 1, n)]

def _diff_set(a):
    return set(_all_diffs(a))

def _is_sidon(a):
    diffs = _all_diffs(a)
    return len(set(diffs)) == len(diffs)

def _covers_1_to_30(a):
    ds = _diff_set(a)
    return all(d in ds for d in range(1, 31))

def _valid(a):
    return len(a) <= 32 and max(a) <= 10**6 and 0 in a and _is_sidon(a) and _covers_1_to_30(a)

def _greedy_construct():
    """Build a Sidon set covering 1..30 via greedy addition."""
    a = [0]
    for d in range(1, 31):
        ds = _diff_set(a)
        if d in ds:
            continue
        # Try to add a single element x such that x - some existing element = d
        # and the Sidon property is preserved.
        found = False
        candidates = set()
        for v in a:
            candidates.add(v + d)
            if v - d >= 0:
                candidates.add(v - d)
        candidates -= set(a)
        for x in sorted(candidates):
            trial = sorted(a + [x])
            if _is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            # Add two elements to realize difference d
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if _is_sidon(trial):
                    a = trial
                    break
                y += 1
    return a

def _local_search(a, deadline):
    """Improve the set by replacing large elements with smaller ones."""
    a = sorted(a)
    best = a
    best_max = max(a)
    
    while time.time() < deadline:
        improved = False
        # Try to remove an element if coverage is still maintained
        for i in range(len(a) - 1, -1, -1):
            if time.time() >= deadline:
                break
            candidate = a[:i] + a[i+1:]
            if len(candidate) < 2:
                continue
            if _is_sidon(candidate) and _covers_1_to_30(candidate):
                a = candidate
                improved = True
                best_max = max(a)
                break
        if improved:
            continue
        
        # Try to replace a large element with a smaller one
        for i in range(len(a) - 1, 0, -1):
            if time.time() >= deadline:
                break
            old = a[i]
            # Try replacing with smaller values
            for new_val in range(1, old):
                if new_val in a:
                    continue
                candidate = a[:i] + [new_val]
                candidate.sort()
                if _is_sidon(candidate) and _covers_1_to_30(candidate):
                    if max(candidate) < best_max:
                        a = candidate
                        best_max = max(a)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try swapping two elements
            swapped = False
            for i in range(len(a)):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, len(a)):
                    # Try replacing a[j] with a smaller value
                    old_j = a[j]
                    for new_val in range(1, old_j):
                        if new_val in a:
                            continue
                        candidate = a[:]
                        candidate[j] = new_val
                        candidate.sort()
                        if _is_sidon(candidate) and _covers_1_to_30(candidate):
                            if max(candidate) < best_max:
                                a = candidate
                                best_max = max(a)
                                swapped = True
                                break
                    if swapped:
                        break
                if swapped:
                    break
            if not swapped:
                break
    
    return a

def _random_restart_construct(deadline):
    """Try multiple random initializations and local search."""
    best = None
    best_max = float('inf')
    
    # Start with greedy
    a = _greedy_construct()
    a = _local_search(a, deadline)
    if max(a) < best_max:
        best = a
        best_max = max(a)
    
    # Try random perturbations
    rng = random.Random(42)
    while time.time() < deadline:
        # Start from best, perturb
        if best is None:
            break
        a = list(best)
        # Randomly remove a non-essential element
        if len(a) > 8:
            removable = [i for i in range(1, len(a))]
            if removable:
                idx = rng.choice(removable)
                a.pop(idx)
                a.sort()
        # Try to repair and improve
        a = _local_search(a, deadline)
        if _valid(a) and max(a) < best_max:
            best = a
            best_max = max(a)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Phase 1: Greedy construction
    a = _greedy_construct()
    
    # Phase 2: Local search to tighten
    a = _local_search(a, deadline)
    
    # Phase 3: If time remains, try more aggressive optimization
    if time.time() < deadline - 5:
        result = _random_restart_construct(deadline)
        if result is not None and max(result) < max(a):
            a = result
    
    return a
