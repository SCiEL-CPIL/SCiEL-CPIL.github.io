"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills at 90s wall-clock. Use 80s safety margin.
_DEADLINE = 80.0

def _is_valid_full(selected):
    """Check that no element divides the sum of any 3 distinct OTHER elements."""
    n = len(selected)
    if n < 4:
        return True
    sel = list(selected)
    for i, a in enumerate(sel):
        others = [v for j, v in enumerate(sel) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy_build(order):
    """Greedy: add elements in given order if valid."""
    selected = []
    for x in order:
        candidate = selected + [x]
        ok = True
        # Incremental check: only need to check new combinations involving x
        # 1) x as the tested element: sum of 3 others divisible by x
        if len(selected) >= 3:
            for t in combinations(selected, 3):
                if sum(t) % x == 0:
                    ok = False
                    break
        if ok:
            # 2) Each existing element a: check if x participates in a 3-sum divisible by a
            for a in selected:
                # Need 2 others from selected (excluding a) such that x + b + c ≡ 0 mod a
                others_excl_a = [v for v in selected if v != a]
                if len(others_excl_a) >= 2:
                    for b, c in combinations(others_excl_a, 2):
                        if (x + b + c) % a == 0:
                            ok = False
                            break
                if not ok:
                    break
        if ok:
            selected.append(x)
    return selected

def _local_search(selected, time_limit=75.0):
    """Local search via swaps: try replacing one element with another to increase size."""
    best = list(selected)
    best_len = len(best)
    start = time.time()
    
    # Precompute for speed
    def _try_add_swap(current, remove_idx, add_val):
        """Try removing current[remove_idx] and adding add_val."""
        if add_val in current:
            return None
        candidate = [v for i, v in enumerate(current) if i != remove_idx]
        candidate.append(add_val)
        # Full validation (small n, acceptable)
        if _is_valid_full(candidate):
            return candidate
        return None
    
    # All candidate values
    all_vals = set(range(1, 193))
    
    while time.time() - start < time_limit:
        improved = False
        cur = list(best)
        # Try all pairs (remove i, add v)
        for i in range(len(cur)):
            if time.time() - start > time_limit:
                break
            removed = cur[i]
            for v in sorted(all_vals - set(cur), key=lambda x: -x):
                if time.time() - start > time_limit:
                    break
                candidate = _try_add_swap(cur, i, v)
                if candidate is not None and len(candidate) > best_len:
                    best = candidate
                    best_len = len(best)
                    improved = True
                    cur = list(best)
                    break
            if improved:
                break
        if not improved:
            # Try 2-opt: remove two, add two (limited)
            found = False
            for i in range(len(cur)):
                if time.time() - start > time_limit:
                    break
                for j in range(i+1, len(cur)):
                    if time.time() - start > time_limit:
                        break
                    candidate = [v for k, v in enumerate(cur) if k != i and k != j]
                    # Try adding two new values
                    remaining = all_vals - set(candidate)
                    # Try top candidates
                    for v1 in sorted(remaining, key=lambda x: -x)[:20]:
                        if time.time() - start > time_limit:
                            break
                        for v2 in sorted(remaining - {v1}, key=lambda x: -x)[:20]:
                            if time.time() - start > time_limit:
                                break
                            test = candidate + [v1, v2]
                            if _is_valid_full(test) and len(test) > best_len:
                                best = test
                                best_len = len(best)
                                cur = list(best)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
    
    return best

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy with different orderings
    # Start with descending (larger numbers are less likely to divide sums)
    greedy_desc = _greedy_build(range(192, 0, -1))
    
    # Also try ascending
    greedy_asc = _greedy_build(range(1, 193))
    
    # Try some random orderings
    best = greedy_desc
    if len(greedy_asc) > len(best):
        best = greedy_asc
    
    # Random restarts
    rng = random.Random(42)
    for _ in range(20):
        if time.time() - start_time > _DEADLINE - 30:
            break
        order = list(range(1, 193))
        rng.shuffle(order)
        result = _greedy_build(order)
        if len(result) > len(best):
            best = result
    
    # Phase 2: Local search
    if time.time() - start_time < _DEADLINE - 5:
        best = _local_search(best, time_limit=_DEADLINE - 5 - (time.time() - start_time))
    
    # Safety: ensure we return within deadline
    if time.time() - start_time > _DEADLINE - 1:
        pass  # Just return what we have
    
    return best
# EVOLVE-BLOCK-END
