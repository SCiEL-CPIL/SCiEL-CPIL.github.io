"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 192
MAX_K = 48
DEADLINE_MARGIN = 5.0  # seconds to keep before the 60s kill (or 90s)


def _valid_addition(x, selected):
    """Check if adding x to selected keeps the set valid.
    Two conditions:
      1) x does not divide the sum of any 3 distinct other selected elements.
      2) For each a in selected, a does not divide the sum of any 3 distinct
         elements from (selected + {x}) excluding a.
    """
    sel = selected
    # Condition 1: x vs triples from sel
    if len(sel) >= 3:
        for t in combinations(sel, 3):
            if sum(t) % x == 0:
                return False
    # Condition 2: for each a in sel, check triples from sel+{x} minus a
    for a in sel:
        # Triples from sel that don't include a and don't include x (already checked in cond1 for a? No)
        # Actually we need: for each a, no triple of 3 distinct elements from (sel + {x}) \ {a} sums to multiple of a.
        # The triples from sel \ {a} were already valid. New triples include x.
        # So check: for each pair (b, c) in sel \ {a}, b != c, sum = x + b + c; check a | (x+b+c)? No wait.
        # Triple is {x, b, c} where b, c in sel, b != c, b != a, c != a.
        # Also triple could be {x, b} + another? No, exactly 3 elements.
        # So new triples involving x and a: {x, b, c} with b, c in sel\{a}, b != c.
        # Check: (x + b + c) % a != 0 for all such pairs.
        others = [v for v in sel if v != a]
        if len(others) >= 2:
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    if (x + others[i] + others[j]) % a == 0:
                        return False
    return True


def _is_valid_set(selected):
    """Full validation of a set."""
    for a in selected:
        others = [v for v in selected if v != a]
        if len(others) >= 3:
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
    return True


def _greedy_large_first():
    selected = []
    for x in range(N, 0, -1):
        if _valid_addition(x, selected):
            selected.append(x)
    return selected


def _greedy_random(seed):
    rng = random.Random(seed)
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    selected = []
    for x in candidates:
        if _valid_addition(x, selected):
            selected.append(x)
    return selected


def _local_search_swap(selected, time_limit):
    """Try to improve cardinality by swapping one element out for another.
    Also try removing an element to allow adding two new ones (net +1)."""
    selected = list(selected)
    current_size = len(selected)
    deadline = time.time() + time_limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try single swap: remove i, add x not in set
        for i in range(len(selected)):
            if time.time() >= deadline:
                break
            sel_without_i = [v for j, v in enumerate(selected) if j != i]
            for x in range(N, 0, -1):
                if x in sel_without_i:
                    continue
                if _valid_addition(x, sel_without_i):
                    new_set = sel_without_i + [x]
                    if len(new_set) > current_size or True:  # same size but maybe enables further swaps
                        # Only accept if it doesn't reduce size; prefer larger
                        if len(new_set) >= current_size:
                            selected = new_set
                            current_size = len(selected)
                            improved = True
                            break
            if time.time() >= deadline:
                break
        # Try remove one, add two
        for i in range(len(selected)):
            if time.time() >= deadline:
                break
            sel_without_i = [v for j, v in enumerate(selected) if j != i]
            # Try to add two elements
            added = []
            for x in range(N, 0, -1):
                if x in sel_without_i or x in added:
                    continue
                if _valid_addition(x, sel_without_i + added):
                    added.append(x)
                    if len(added) == 2:
                        break
            if len(added) == 2:
                new_set = sel_without_i + added
                if len(new_set) > current_size:
                    selected = new_set
                    current_size = len(selected)
                    improved = True
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + (DEADLINE_MARGIN if DEADLINE_MARGIN < 55 else 55)
    
    # Phase 1: Greedy large-first (fast, good baseline)
    best = _greedy_large_first()
    best_size = len(best)
    
    # Phase 2: Try multiple random greedy orders
    rng = random.Random(42)
    for seed in range(200):
        if time.time() >= deadline:
            break
        candidate = _greedy_random(seed)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Phase 3: Local search on best
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        best = _local_search_swap(best, remaining)
        best_size = len(best)
    
    # Final validation
    if not _is_valid_set(best):
        # Fallback to greedy
        best = _greedy_large_first()
    
    return best
# EVOLVE-BLOCK-END
