"""Improved 4-AP-free set construction using bitset-based checks and local search."""
import time
import random
import sys

N = 512
DEADLINE = 85.0  # safety margin before 90s kill

def _has_4ap_bitset(selected_bits):
    """Check if any 4-AP exists in the set represented by a bitset (list of bools or int).
    Uses the fact that a 4-AP a, a+d, a+2d, a+3d exists iff for some a,d all four are present.
    We iterate over possible (a, d) pairs."""
    # selected_bits is a Python int where bit i means (i+1) is in the set
    # 4-AP: a, a+d, a+2d, a+3d with a>=1, d>=1, a+3d<=512
    # So a ranges 1..512, d ranges 1..(512-a)//3
    # Bit for value v is at position v-1
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        if max_d < 1:
            continue
        for d in range(1, max_d + 1):
            b1 = (a - 1)
            b2 = (a + d - 1)
            b3 = (a + 2*d - 1)
            b4 = (a + 3*d - 1)
            if (selected_bits >> b1) & 1 and \
               (selected_bits >> b2) & 1 and \
               (selected_bits >> b3) & 1 and \
               (selected_bits >> b4) & 1:
                return True
    return False

def _has_4ap_fast(selected_set, a, d):
    """Check if a, a+d, a+2d, a+3d are all in selected_set."""
    return (a in selected_set and (a+d) in selected_set and 
            (a+2*d) in selected_set and (a+3*d) in selected_set)

def _greedy_randomized(seed=None):
    """Randomized greedy: shuffle candidates, add if no 4-AP is formed."""
    if seed is not None:
        random.seed(seed)
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    selected = set()
    # Precompute for each x, the list of (a, d) such that x completes a 4-AP
    # x can be at position 1,2,3,4 in the AP
    # If x is 1st: x, x+d, x+2d, x+3d -> need x+d, x+2d, x+3d in selected
    # If x is 2nd: x-d, x, x+d, x+2d -> need x-d, x+d, x+2d in selected
    # If x is 3rd: x-2d, x-d, x, x+d -> need x-2d, x-d, x+d in selected
    # If x is 4th: x-3d, x-2d, x-d, x -> need x-3d, x-2d, x-d in selected
    
    # For speed, precompute for each x the list of (a,d) pairs where x is involved
    # But that's complex. Instead, for each x, check all possible APs containing x.
    
    for x in candidates:
        # Check if adding x creates any 4-AP
        conflict = False
        # x as 1st element: x, x+d, x+2d, x+3d
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x+d) in selected and (x+2*d) in selected and (x+3*d) in selected:
                conflict = True
                break
        if conflict:
            continue
        # x as 2nd element: x-d, x, x+d, x+2d
        if not conflict:
            max_d2 = min(x - 1, (N - x) // 2)
            for d in range(1, max_d2 + 1):
                if (x-d) in selected and (x+d) in selected and (x+2*d) in selected:
                    conflict = True
                    break
        if conflict:
            continue
        # x as 3rd element: x-2d, x-d, x, x+d
        if not conflict:
            max_d3 = min((x - 1) // 2, N - x)
            for d in range(1, max_d3 + 1):
                if (x-2*d) in selected and (x-d) in selected and (x+d) in selected:
                    conflict = True
                    break
        if conflict:
            continue
        # x as 4th element: x-3d, x-2d, x-d, x
        if not conflict:
            max_d4 = (x - 1) // 3
            for d in range(1, max_d4 + 1):
                if (x-3*d) in selected and (x-2*d) in selected and (x-d) in selected:
                    conflict = True
                    break
        if not conflict:
            selected.add(x)
    return selected

def _local_search_add(selected, start_time, deadline):
    """Try to add elements from outside the set that don't create 4-APs."""
    selected = set(selected)
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline - 2.0:
            break
        added = False
        for x in range(1, N + 1):
            if x in selected:
                continue
            # Check if x can be added
            conflict = False
            max_d1 = (N - x) // 3
            for d in range(1, max_d1 + 1):
                if (x+d) in selected and (x+2*d) in selected and (x+3*d) in selected:
                    conflict = True
                    break
            if not conflict:
                max_d2 = min(x - 1, (N - x) // 2)
                for d in range(1, max_d2 + 1):
                    if (x-d) in selected and (x+d) in selected and (x+2*d) in selected:
                        conflict = True
                        break
            if not conflict:
                max_d3 = min((x - 1) // 2, N - x)
                for d in range(1, max_d3 + 1):
                    if (x-2*d) in selected and (x-d) in selected and (x+d) in selected:
                        conflict = True
                        break
            if not conflict:
                max_d4 = (x - 1) // 3
                for d in range(1, max_d4 + 1):
                    if (x-3*d) in selected and (x-2*d) in selected and (x-d) in selected:
                        conflict = True
                        break
            if not conflict:
                selected.add(x)
                added = True
        if not added:
            break
    return selected

def _local_search_swap(selected, start_time, deadline):
    """Try to improve by removing one element and adding another (or more)."""
    selected = set(selected)
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start_time
        if elapsed > deadline - 2.0:
            break
        # Try removing each element and see if we can add more
        for x in list(selected):
            if time.time() - start_time > deadline - 2.0:
                break
            trial = selected - {x}
            # Try to add elements back
            added_count = 0
            for y in range(1, N + 1):
                if y in trial:
                    continue
                # Check if y can be added to trial
                conflict = False
                max_d1 = (N - y) // 3
                for d in range(1, max_d1 + 1):
                    if (y+d) in trial and (y+2*d) in trial and (y+3*d) in trial:
                        conflict = True
                        break
                if not conflict:
                    max_d2 = min(y - 1, (N - y) // 2)
                    for d in range(1, max_d2 + 1):
                        if (y-d) in trial and (y+d) in trial and (y+2*d) in trial:
                            conflict = True
                            break
                if not conflict:
                    max_d3 = min((y - 1) // 2, N - y)
                    for d in range(1, max_d3 + 1):
                        if (y-2*d) in trial and (y-d) in trial and (y+d) in trial:
                            conflict = True
                            break
                if not conflict:
                    max_d4 = (y - 1) // 3
                    for d in range(1, max_d4 + 1):
                        if (y-3*d) in trial and (y-2*d) in trial and (y-d) in trial:
                            conflict = True
                            break
                if not conflict:
                    trial.add(y)
                    added_count += 1
            if len(trial) > len(selected):
                selected = trial
                improved = True
                break
    return selected

def construct():
    start_time = time.time()
    
    # Start with a good initial greedy
    best_set = _greedy_randomized(seed=42)
    best_size = len(best_set)
    
    # Try multiple random seeds
    for seed in range(42, 42 + 50):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5.0:
            break
        s = _greedy_randomized(seed=seed)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Local search to add more elements
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 3.0:
        best_set = _local_search_add(best_set, start_time, DEADLINE)
        best_size = len(best_set)
    
    # Try local search swap
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 3.0:
        best_set = _local_search_swap(best_set, start_time, DEADLINE)
        best_size = len(best_set)
    
    # Final verification
    selected_bits = 0
    for x in best_set:
        selected_bits |= (1 << (x - 1))
    if _has_4ap_bitset(selected_bits):
        # Fallback to a safe set
        safe_set = set()
        for x in range(1, N + 1):
            # Simple check
            ok = True
            max_d1 = (N - x) // 3
            for d in range(1, max_d1 + 1):
                if (x+d) in safe_set and (x+2*d) in safe_set and (x+3*d) in safe_set:
                    ok = False
                    break
            if ok:
                max_d2 = min(x - 1, (N - x) // 2)
                for d in range(1, max_d2 + 1):
                    if (x-d) in safe_set and (x+d) in safe_set and (x+2*d) in safe_set:
                        ok = False
                        break
            if ok:
                max_d3 = min((x - 1) // 2, N - x)
                for d in range(1, max_d3 + 1):
                    if (x-2*d) in safe_set and (x-d) in safe_set and (x+d) in safe_set:
                        ok = False
                        break
            if ok:
                max_d4 = (x - 1) // 3
                for d in range(1, max_d4 + 1):
                    if (x-3*d) in safe_set and (x-2*d) in safe_set and (x-d) in safe_set:
                        ok = False
                        break
            if ok:
                safe_set.add(x)
        best_set = safe_set
    
    return sorted(best_set)
