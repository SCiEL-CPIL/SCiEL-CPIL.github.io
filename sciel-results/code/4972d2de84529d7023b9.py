# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

_N = 192
_MAX = 48
_START_TIME = time.time()
_DEADLINE = _START_TIME + 82.0  # Safety margin before 90s kill

def _time_left():
    return _DEADLINE - time.time()

# Precompute all triples (i,j,k) of distinct values in 1..N with their sum
# triples_by_sum[val] = list of (a,b,c) where a<b<c and a+b+c=val
_triples_by_sum = defaultdict(list)
for a in range(1, _N + 1):
    for b in range(a + 1, _N + 1):
        for c in range(b + 1, _N + 1):
            s = a + b + c
            _triples_by_sum[s].append((a, b, c))

# Precompute for each value x, which sums are divisible by x
# For x in 1..192, the possible sums are 1..576 (max 3*192)
# divisible_sums[x] = set of sums s where s % x == 0
_divisible_sums = [set() for _ in range(_N + 1)]
for x in range(1, _N + 1):
    for m in range(1, 577 // x + 1):
        s = x * m
        if s <= 3 * _N:
            _divisible_sums[x].add(s)

def _is_valid_addition_fast(S_set, x):
    """Check if adding x to valid set S_set keeps it valid.
    
    Uses precomputed triples for faster checking.
    """
    S_list = list(S_set)
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S_set
    # For each triple in S_set, check if sum % x == 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                s = S_list[i] + S_list[j] + S_list[k]
                if s % x == 0:
                    return False
    
    # Check 2: for each a in S_set, a does not divide sum of x + 2 others from S_set
    for i, a in enumerate(S_list):
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                s = x + S_list[j] + S_list[k]
                if s % a == 0:
                    return False
    
    return True

def _is_valid_addition_bitset(S_set, x):
    """Check validity using bitset-based approach for speed."""
    S_list = list(S_set)
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S_set
    # Build a set of all triple sums from S_set
    triple_sums = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triple_sums.add(S_list[i] + S_list[j] + S_list[k])
    
    # Check if any triple sum is divisible by x
    for s in triple_sums:
        if s % x == 0:
            return False
    
    # Check 2: for each a in S_set, a does not divide sum of x + 2 others
    # For each a, check all pairs (j,k) where j != i, k != i
    for i, a in enumerate(S_list):
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                s = x + S_list[j] + S_list[k]
                if s % a == 0:
                    return False
    
    return True

def _greedy_optimized(order):
    """Greedy with given order."""
    S = set()
    for x in order:
        if x in S:
            continue
        if _is_valid_addition_fast(S, x):
            S.add(x)
            if len(S) >= _MAX:
                break
    return S

def _local_search_model(S_init, time_budget=20.0):
    """Local search: try swaps, insertions, and 1-out-2-in moves."""
    S = set(S_init)
    start_time = time.time()
    best = set(S)
    best_len = len(S)
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget and _time_left() > 5:
        improved = False
        
        # Try insertions
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            for x in candidates[:50]:
                if _is_valid_addition_fast(S, x):
                    S.add(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = set(S)
                    break
        
        # Try 1-out-2-in moves (net gain +1)
        if len(S) < _MAX and len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            
            found_move = False
            for a in to_remove[:10]:
                if found_move:
                    break
                S_temp = S - {a}
                for x in candidates[:20]:
                    if found_move:
                        break
                    if not _is_valid_addition_fast(S_temp, x):
                        continue
                    S_temp_x = S_temp | {x}
                    for y in candidates[:20]:
                        if y == x:
                            continue
                        if _is_valid_addition_fast(S_temp_x, y):
                            S = S_temp_x | {y}
                            improved = True
                            found_move = True
                            if len(S) > best_len:
                                best_len = len(S)
                                best = set(S)
                            break
        
        # Try swaps (1-out-1-in)
        if not improved and len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            
            for a in to_remove[:20]:
                if improved:
                    break
                for x in candidates[:20]:
                    S_new = S - {a}
                    if x not in S_new and _is_valid_addition_fast(S_new, x):
                        S = S_new
                        S.add(x)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = set(S)
                        break
        
        if not improved:
            # Perturb: remove 2 random elements to escape local optima
            if len(S) > 2:
                to_remove = random.sample(list(S), 2)
                S -= set(to_remove)
    
    return best

def _band_optimization():
    """Optimize band parameters using model-and-optimize approach."""
    best = set()
    best_len = 0
    
    configs = [
        [(100, 192)],
        [(80, 192)],
        [(60, 192)],
        [(50, 192)],
        [(40, 192)],
        [(100, 192), (20, 39)],
        [(100, 192), (30, 49)],
        [(90, 192), (20, 50)],
        [(80, 192), (15, 40)],
        [(70, 192), (10, 35)],
        [(120, 192), (40, 70)],
        [(110, 192), (30, 60)],
        [(100, 192), (25, 50)],
        [(95, 192), (20, 45)],
        [(90, 192), (15, 40)],
        [(100, 192), (1, 30)],
        [(110, 192), (1, 40)],
        [(90, 192), (1, 50)],
        [(130, 192), (30, 60)],
        [(140, 192), (20, 50)],
        [(150, 192), (10, 40)],
    ]
    
    for config in configs:
        for _ in range(8):
            S = set()
            for start, end in config:
                candidates = list(range(start, min(end, _N) + 1))
                random.shuffle(candidates)
                for x in candidates:
                    if len(S) >= _MAX:
                        break
                    if x not in S and _is_valid_addition_fast(S, x):
                        S.add(x)
            if len(S) > best_len:
                best_len = len(S)
                best = set(S)
    
    return best

def _multi_start_greedy(num_starts=30):
    """Multi-start randomized greedy."""
    best = set()
    best_len = 0
    
    for _ in range(num_starts):
        if _time_left() < 10:
            break
        order = list(range(1, _N + 1))
        # Weighted random ordering
        weights = [random.random() * 100 + v for v in order]
        order.sort(key=lambda v: -weights[v - 1])
        S = _greedy_optimized(order)
        if len(S) > best_len:
            best_len = len(S)
            best = set(S)
    
    return best

def _refinement_pass(S_init, time_budget=15.0):
    """High-precision polish pass: try all possible single-element swaps
    and small neighborhood moves with tighter tolerance."""
    S = set(S_init)
    start_time = time.time()
    best = set(S)
    best_len = len(S)
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget and _time_left() > 5:
        improved = False
        
        # Exhaustive single-element insertion
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S]
            for x in candidates:
                if _is_valid_addition_fast(S, x):
                    S.add(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = set(S)
                    break
        
        # Exhaustive single-element swap
        if len(S) >= 1:
            to_remove = list(S)
            candidates = [v for v in all_vals if v not in S]
            
            for a in to_remove:
                if improved:
                    break
                S_without_a = S - {a}
                for x in candidates:
                    if x in S_without_a:
                        continue
                    if _is_valid_addition_fast(S_without_a, x):
                        S = S_without_a | {x}
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = set(S)
                        break
        
        # 1-out-2-in moves
        if len(S) < _MAX and len(S) >= 1 and not improved:
            to_remove = list(S)
            candidates = [v for v in all_vals if v not in S]
            
            for a in to_remove:
                if improved:
                    break
                S_temp = S - {a}
                for x in candidates:
                    if improved:
                        break
                    if not _is_valid_addition_fast(S_temp, x):
                        continue
                    S_temp_x = S_temp | {x}
                    for y in candidates:
                        if y == x or y in S_temp_x:
                            continue
                        if _is_valid_addition_fast(S_temp_x, y):
                            S = S_temp_x | {y}
                            improved = True
                            if len(S) > best_len:
                                best_len = len(S)
                                best = set(S)
                            break
        
        if not improved:
            # Small perturbation
            if len(S) > 3:
                to_remove = random.sample(list(S), 2)
                S -= set(to_remove)
    
    return best

def construct():
    """Main entry point: find maximum valid set."""
    deadline = _DEADLINE
    random.seed(42)  # Deterministic
    
    # Phase 1: Band optimization (model-and-optimize)
    best = _band_optimization()
    
    # Phase 2: Greedy with various orders
    if _time_left() > 30:
        orders = [
            range(_N, 0, -1),
            range(1, _N + 1),
            sorted(range(1, _N + 1), key=lambda x: -x),
            list(range(50, _N + 1)) + list(range(1, 50)),
            list(range(80, _N + 1)) + list(range(1, 80)),
            list(range(100, _N + 1)) + list(range(1, 100)),
            list(range(120, _N + 1)) + list(range(1, 120)),
            list(range(130, _N + 1)) + list(range(1, 130)),
            list(range(140, _N + 1)) + list(range(1, 140)),
        ]
        for order in orders:
            if _time_left() < 30:
                break
            S = _greedy_optimized(order)
            if len(S) > len(best):
                best = set(S)
    
    # Phase 3: Multi-start randomized greedy
    if _time_left() > 25:
        S = _multi_start_greedy(num_starts=30)
        if len(S) > len(best):
            best = set(S)
    
    # Phase 4: Local search from best
    if _time_left() > 20:
        budget = min(18.0, _time_left() - 10)
        S = _local_search_model(best, time_budget=budget)
        if len(S) > len(best):
            best = set(S)
    
    # Phase 5: Refinement pass (high-precision polish)
    if _time_left() > 12:
        budget = min(10.0, _time_left() - 6)
        S = _refinement_pass(best, time_budget=budget)
        if len(S) > len(best):
            best = set(S)
    
    # Phase 6: Final local search if time permits
    if _time_left() > 8:
        budget = min(6.0, _time_left() - 4)
        S = _local_search_model(best, time_budget=budget)
        if len(S) > len(best):
            best = set(S)
    
    return list(best)
# EVOLVE-BLOCK-END
