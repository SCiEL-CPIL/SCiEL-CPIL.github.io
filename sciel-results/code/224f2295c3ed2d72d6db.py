# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

_N = 192
_MAX = 48
_DEADLINE = time.time() + 85.0  # Safety margin before 90s kill

def _precompute_pair_sums(S_list):
    """Precompute all pair sums and their residues for faster validity checking."""
    n = len(S_list)
    pair_sums = []  # list of (sum, i, j)
    for i in range(n):
        for j in range(i + 1, n):
            pair_sums.append((S_list[i] + S_list[j], i, j))
    return pair_sums

def _is_valid_addition_v3(S_list, x):
    """Optimized validity check using precomputed pair sums."""
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S_list
    # Precompute pair sums
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            ab = a + S_list[j]
            for k in range(j + 1, n):
                s = ab + S_list[k]
                if s % x == 0:
                    return False
    
    # Check 2: for each a in S_list, a does not divide sum of x + 2 others from S_list
    for i, a in enumerate(S_list):
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            for k in range(j + 1, n):
                if k == i:
                    continue
                s = x + b + S_list[k]
                if s % a == 0:
                    return False
    
    return True

def _is_valid_addition_fast_v2(S_list, x):
    """Faster validity check using list instead of set."""
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S_list
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            ab = a + b
            for k in range(j + 1, n):
                s = ab + S_list[k]
                if s % x == 0:
                    return False
    
    # Check 2: for each a in S_list, a does not divide sum of x + 2 others from S_list
    for i, a in enumerate(S_list):
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            for k in range(j + 1, n):
                if k == i:
                    continue
                s = x + b + S_list[k]
                if s % a == 0:
                    return False
    
    return True

def _count_conflicts(S_list, x):
    """Count how many constraints x would violate if added to S_list."""
    n = len(S_list)
    conflicts = 0
    
    # Check 1: x does not divide sum of any 3 in S_list
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            ab = a + S_list[j]
            for k in range(j + 1, n):
                s = ab + S_list[k]
                if s % x == 0:
                    conflicts += 1
                    if conflicts > 10:  # Early exit
                        return conflicts
    
    # Check 2: for each a in S_list, a does not divide sum of x + 2 others from S_list
    for i, a in enumerate(S_list):
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            for k in range(j + 1, n):
                if k == i:
                    continue
                s = x + b + S_list[k]
                if s % a == 0:
                    conflicts += 1
                    if conflicts > 10:
                        return conflicts
    
    return conflicts

def _greedy_optimized(order, use_v2=True):
    """Greedy with given order."""
    S = []
    S_set = set()
    for x in order:
        if x in S_set:
            continue
        if use_v2:
            valid = _is_valid_addition_fast_v2(S, x)
        else:
            valid = _is_valid_addition_v3(S, x)
        if valid:
            S.append(x)
            S_set.add(x)
            if len(S) >= _MAX:
                break
    return S_set

def _local_search_model(S_init, time_budget=20.0):
    """Local search: try swaps, insertions, and 1-out-2-in moves."""
    S = set(S_init)
    S_list = list(S)
    start_time = time.time()
    best = set(S)
    best_len = len(S)
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try insertions
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            for x in candidates[:50]:
                S_list = list(S)
                if _is_valid_addition_fast_v2(S_list, x):
                    S.add(x)
                    S_list.append(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = set(S)
                    break
        
        # Try 1-out-2-in moves (Grow set by 1)
        if len(S) < _MAX and len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            # Sort candidates by value descending (larger values are safer)
            candidates.sort(reverse=True)
            
            found_move = False
            for a in to_remove[:8]:
                if found_move:
                    break
                S_temp = S - {a}
                S_temp_list = list(S_temp)
                # Try to add 2 elements from candidates
                for i, x in enumerate(candidates[:40]):
                    if found_move:
                        break
                    # Check if x can be added to S_temp
                    if not _is_valid_addition_fast_v2(S_temp_list, x):
                        continue
                    S_temp_x = S_temp | {x}
                    S_temp_x_list = S_temp_list + [x]
                    # Now try to add y to S_temp_x
                    for y in candidates[i+1:i+40]:
                        if y == x:
                            continue
                        if _is_valid_addition_fast_v2(S_temp_x_list, y):
                            # Move is valid: remove a, add x, add y
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
            
            for a in to_remove[:25]:
                if improved:
                    break
                S_new = S - {a}
                S_new_list = list(S_new)
                for x in candidates[:25]:
                    if x not in S_new and _is_valid_addition_fast_v2(S_new_list, x):
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

def _multi_start_greedy(time_budget=15.0):
    """Multiple greedy starts with different strategies."""
    start_time = time.time()
    best = set()
    best_len = 0
    
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget:
        # Different ordering strategies
        strategy = random.choice(['desc', 'asc', 'random', 'band_high', 'band_mid', 'band_low'])
        
        if strategy == 'desc':
            order = sorted(all_vals, reverse=True)
        elif strategy == 'asc':
            order = sorted(all_vals)
        elif strategy == 'random':
            order = all_vals[:]
            random.shuffle(order)
        elif strategy == 'band_high':
            # Prefer high values
            weights = [v for v in all_vals]
            order = sorted(all_vals, key=lambda v: -random.random() * 20 - v)
        elif strategy == 'band_mid':
            # Prefer mid-range values
            order = sorted(all_vals, key=lambda v: -abs(v - 96) + random.random() * 30)
        else:  # band_low
            # Prefer low values
            order = sorted(all_vals, key=lambda v: v + random.random() * 50)
        
        S = _greedy_optimized(order, use_v2=True)
        if len(S) > best_len:
            best_len = len(S)
            best = set(S)
    
    return best

def _band_optimization():
    """Optimize band parameters with more configurations."""
    best = set()
    best_len = 0
    
    configs = [
        [(100, 192)],
        [(80, 192)],
        [(60, 192)],
        [(50, 192)],
        [(40, 192)],
        [(30, 192)],
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
        [(85, 192), (10, 30)],
        [(75, 192), (5, 25)],
        [(130, 192), (50, 80)],
        [(120, 192), (30, 50)],
    ]
    
    for config in configs:
        for _ in range(3):
            S = set()
            S_list = []
            for start, end in config:
                candidates = list(range(start, min(end, _N) + 1))
                random.shuffle(candidates)
                for x in candidates:
                    if len(S) >= _MAX:
                        break
                    if x not in S and _is_valid_addition_fast_v2(S_list, x):
                        S.add(x)
                        S_list.append(x)
            if len(S) > best_len:
                best_len = len(S)
                best = set(S)
    
    return best

def _conflict_aware_greedy(time_budget=10.0):
    """Greedy that considers conflict counts to pick best candidates."""
    start_time = time.time()
    best = set()
    best_len = 0
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget:
        S = set()
        S_list = []
        remaining = set(all_vals)
        
        while len(S) < _MAX and remaining:
            # Find candidate with minimum conflicts
            best_x = None
            min_conflicts = float('inf')
            
            # Sample a subset for speed
            sample = list(remaining)
            random.shuffle(sample)
            sample = sample[:30]
            
            for x in sample:
                if x in S:
                    continue
                conflicts = _count_conflicts(S_list, x)
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_x = x
            
            if best_x is not None and min_conflicts == 0:
                S.add(best_x)
                S_list.append(best_x)
                remaining.discard(best_x)
            elif best_x is not None and min_conflicts <= 2:
                # Allow small number of conflicts for exploration
                if random.random() < 0.3:
                    S.add(best_x)
                    S_list.append(best_x)
                    remaining.discard(best_x)
                else:
                    break
            else:
                break
        
        if len(S) > best_len:
            best_len = len(S)
            best = set(S)
    
    return best

def _refinement_pass(S_init, time_budget=10.0):
    """Refinement: try to improve by removing problematic elements and adding better ones."""
    S = set(S_init)
    start_time = time.time()
    best = set(S)
    best_len = len(S)
    
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try to find elements that are causing many conflicts and remove them
        # Then try to add back different elements
        if len(S) > 10:
            # Score each element by how many constraints it violates when added
            # (approximate by checking how many potential additions it blocks)
            S_list = list(S)
            n = len(S_list)
            
            # For each element, count how many pairs it creates that would block additions
            block_counts = defaultdict(int)
            for i, a in enumerate(S_list):
                for j in range(i + 1, n):
                    b = S_list[j]
                    # a blocks x if there exists c in S such that a | (x + b + c)
                    # This is complex, so use a simpler heuristic
                    block_counts[a] += 1
                    block_counts[b] += 1
            
            # Remove the element with highest block count (most restrictive)
            if block_counts:
                worst = max(block_counts.keys(), key=lambda x: block_counts[x])
                S_temp = S - {worst}
                S_temp_list = list(S_temp)
                
                # Try to add a different element
                candidates = [v for v in all_vals if v not in S_temp]
                random.shuffle(candidates)
                
                for x in candidates[:30]:
                    if _is_valid_addition_fast_v2(S_temp_list, x):
                        S = S_temp | {x}
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = set(S)
                        break
        
        # Try standard local search moves
        if not improved:
            # Try insertions
            if len(S) < _MAX:
                candidates = [v for v in all_vals if v not in S]
                random.shuffle(candidates)
                S_list = list(S)
                for x in candidates[:40]:
                    if _is_valid_addition_fast_v2(S_list, x):
                        S.add(x)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = set(S)
                        break
            
            # Try swaps
            if not improved and len(S) >= 1:
                to_remove = list(S)
                random.shuffle(to_remove)
                candidates = [v for v in all_vals if v not in S]
                random.shuffle(candidates)
                
                for a in to_remove[:20]:
                    if improved:
                        break
                    S_new = S - {a}
                    S_new_list = list(S_new)
                    for x in candidates[:20]:
                        if x not in S_new and _is_valid_addition_fast_v2(S_new_list, x):
                            S = S_new
                            S.add(x)
                            improved = True
                            if len(S) > best_len:
                                best_len = len(S)
                                best = set(S)
                            break
        
        if not improved:
            # Perturb
            if len(S) > 2:
                to_remove = random.sample(list(S), 2)
                S -= set(to_remove)
    
    return best

def construct():
    """Main entry point: find maximum valid set."""
    deadline = _DEADLINE
    random.seed(42)  # Deterministic
    
    # Phase 1: Band optimization
    best = _band_optimization()
    
    # Phase 2: Conflict-aware greedy
    if time.time() < deadline - 40:
        budget = min(10.0, deadline - time.time() - 30)
        S = _conflict_aware_greedy(time_budget=budget)
        if len(S) > len(best):
            best = S
    
    # Phase 3: Multi-start greedy
    if time.time() < deadline - 25:
        budget = min(10.0, deadline - time.time() - 15)
        S = _multi_start_greedy(time_budget=budget)
        if len(S) > len(best):
            best = S
    
    # Phase 4: Local search from best
    if time.time() < deadline - 10:
        budget = min(10.0, deadline - time.time() - 5)
        S = _local_search_model(best, time_budget=budget)
        if len(S) > len(best):
            best = S
    
    # Phase 5: Refinement pass
    if time.time() < deadline - 5:
        budget = min(5.0, deadline - time.time() - 2)
        S = _refinement_pass(best, time_budget=budget)
        if len(S) > len(best):
            best = S
    
    return list(best)
# EVOLVE-BLOCK-END
