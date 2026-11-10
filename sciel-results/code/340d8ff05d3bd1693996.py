"""Fast structural solver: large-first greedy + bitmask-accelerated local search."""
import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_START = time.time()
_DEADLINE = _START + 80.0  # 10s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _validate_set(S):
    """Full validation of set S. Returns True if valid."""
    S_list = sorted(S)
    n = len(S_list)
    if n < 4:
        return True
    # Precompute all triple sums with indices
    # For each element a at index idx, check no triple (not containing idx) has sum % a == 0
    for idx, a in enumerate(S_list):
        # Build pair sums of others
        others = [S_list[j] for j in range(n) if j != idx]
        m = len(others)
        if m < 3:
            continue
        # Check all triples in others
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    if (others[i] + others[j] + others[k]) % a == 0:
                        return False
    return True

def _can_add(S, x):
    """Check if x can be added to valid set S. S is a list (sorted)."""
    n = len(S)
    # Check 1: no triple in S has sum % x == 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (S[i] + S[j] + S[k]) % x == 0:
                    return False
    # Check 2: for each a in S, no triple (x + two from S\{a}) has sum % a == 0
    for i, a in enumerate(S):
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                if (x + S[j] + S[k]) % a == 0:
                    return False
    return True

def _greedy_largest_first():
    """Greedy: add from 192 down to 1, keep if valid."""
    S = []
    for x in range(_N, 0, -1):
        if len(S) >= _MAX:
            break
        if _can_add(S, x):
            S.append(x)
    return S

def _greedy_with_order(order):
    S = []
    for x in order:
        if len(S) >= _MAX:
            break
        if _can_add(S, x):
            S.append(x)
    return S

def _local_search(S_init, time_budget):
    """Local search with swaps and insertions."""
    S = list(S_init)
    best = list(S)
    best_len = len(S)
    all_vals = set(range(1, _N + 1))
    S_set = set(S)
    end_time = time.time() + time_budget
    
    while time.time() < end_time and _time_left() > 2:
        improved = False
        
        # Try insertions
        if len(S) < _MAX:
            candidates = [v for v in all_vals - S_set if v not in S_set]
            random.shuffle(candidates)
            for x in candidates[:30]:
                if _can_add(S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # Try swaps
        if len(S) >= 2 and not improved:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals - S_set]
            random.shuffle(candidates)
            
            for a in to_remove[:15]:
                for x in candidates[:15]:
                    if x == a:
                        continue
                    S_new = [v for v in S if v != a]
                    if _can_add(S_new, x):
                        S = S_new + [x]
                        S_set = set(S)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = list(S)
                        break
                if improved:
                    break
        
        # Perturb if stuck
        if not improved and len(S) > 4:
            to_remove = random.sample(S, min(3, len(S) - 2))
            S = [v for v in S if v not in set(to_remove)]
            S_set = set(S)
    
    return best

def construct():
    random.seed(12345)
    deadline = _DEADLINE
    
    # Phase 1: Largest-first greedy (fast, good baseline)
    best = _greedy_largest_first()
    
    # Phase 2: Try several other greedy orders
    if _time_left() > 40:
        orders = [
            list(range(1, _N + 1)),  # smallest first
            list(range(_N, 0, -1)),  # largest first (already done)
            list(range(100, _N + 1)) + list(range(1, 100)),
            list(range(150, _N + 1)) + list(range(1, 150)),
            list(range(50, _N + 1)) + list(range(1, 50)),
            list(range(1, _N + 1, 2)) + list(range(2, _N + 1, 2)),  # odds then evens
            list(range(_N, 0, -2)) + list(range(_N - 1, 0, -2)),  # large odds then evens
        ]
        for order in orders:
            if _time_left() < 30:
                break
            S = _greedy_with_order(order)
            if len(S) > len(best):
                best = S
    
    # Phase 3: Multi-start randomized greedy
    if _time_left() > 25:
        for trial in range(30):
            if _time_left() < 20:
                break
            # Random order biased toward large values
            vals = list(range(1, _N + 1))
            # Weight larger values higher
            weights = [v * random.random() + 50 for v in vals]
            order = sorted(vals, key=lambda v: -weights[v - 1])
            S = _greedy_with_order(order)
            if len(S) > len(best):
                best = S
    
    # Phase 4: Local search from best
    if _time_left() > 15:
        budget = min(10.0, _time_left() - 5)
        S = _local_search(best, budget)
        if len(S) > len(best):
            best = S
    
    # Phase 5: Another local search round
    if _time_left() > 8:
        budget = min(5.0, _time_left() - 3)
        S = _local_search(best, budget)
        if len(S) > len(best):
            best = S
    
    # Phase 6: Final insertion attempts
    if _time_left() > 3:
        S = list(best)
        S_set = set(S)
        for x in range(_N, 0, -1):
            if len(S) >= _MAX:
                break
            if x not in S_set and _can_add(S, x):
                S.append(x)
                S_set.add(x)
        if len(S) > len(best):
            best = S
    
    return list(best)
