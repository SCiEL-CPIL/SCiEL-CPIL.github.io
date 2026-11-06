"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _all_diffs(s):
    """Return list of all positive differences in sorted set s."""
    diffs = []
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(s[j] - s[i])
    return diffs

def _is_sidon(s):
    """Check if all positive differences are distinct."""
    diffs = _all_diffs(s)
    return len(set(diffs)) == len(diffs)

def _covers_1_to_30(s):
    """Check if all d in 1..30 appear as a difference."""
    diffs = set(_all_diffs(s))
    return all(d in diffs for d in range(1, 31))

def _greedy_construct():
    """Greedy construction to get a valid starting set."""
    a = [0]
    for d in range(1, 31):
        if d in _all_diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if _is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if _is_sidon(trial):
                    a = trial
                    break
                y += 1
    return a

def _local_search(a, deadline):
    """Try to reduce max element by replacing large elements with smaller ones."""
    a = sorted(a)
    best_max = max(a)
    
    while time.time() < deadline:
        improved = False
        # Try removing each non-zero element and re-adding a smaller one
        for idx in range(len(a) - 1, 0, -1):  # Try to remove larger elements first
            elem = a[idx]
            remaining = a[:idx] + a[idx+1:]
            if len(remaining) < 2:
                continue
            # Check if remaining still covers 1..30 and is Sidon
            if not _is_sidon(remaining) or not _covers_1_to_30(remaining):
                continue
            # Try to add back a smaller element
            for new_val in range(1, elem):
                if new_val in remaining:
                    continue
                trial = sorted(remaining + [new_val])
                if _is_sidon(trial) and _covers_1_to_30(trial):
                    a = trial
                    best_max = max(a)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try pairwise swaps: remove one, add another
            improved = False
            for idx in range(len(a) - 1, 0, -1):
                elem = a[idx]
                remaining = a[:idx] + a[idx+1:]
                if len(remaining) < 2:
                    continue
                # We need to add exactly one element to keep size
                # But we removed one, so we add one
                for new_val in range(1, elem):
                    if new_val in remaining:
                        continue
                    trial = sorted(remaining + [new_val])
                    if _is_sidon(trial) and _covers_1_to_30(trial):
                        a = trial
                        best_max = max(a)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try adding an element then removing a larger one (net size same or +1)
                # Actually, let's try: for each element, try replacing with a smaller one
                # that keeps validity. If none works, try different order.
                # Try random restarts of the local search
                if random.random() < 0.1:
                    # Perturb: pick a random element and try to move it
                    idx = random.randint(1, len(a) - 1)
                    elem = a[idx]
                    remaining = a[:idx] + a[idx+1:]
                    for new_val in range(1, elem):
                        if new_val in remaining:
                            continue
                        trial = sorted(remaining + [new_val])
                        if _is_sidon(trial) and _covers_1_to_30(trial):
                            a = trial
                            best_max = max(a)
                            improved = True
                            break
                if not improved:
                    break
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Get initial greedy solution
    a = _greedy_construct()
    
    # Local search to minimize max element
    a = _local_search(a, deadline)
    
    # Safety check
    if not _is_sidon(a) or not _covers_1_to_30(a) or 0 not in a:
        # Fallback to greedy
        a = _greedy_construct()
    
    return a
# EVOLVE-BLOCK-END
