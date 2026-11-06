import time
import random
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
DEADLINE = 80.0  # safety margin under 90s wall-clock

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _can_form_target(target, items):
    """
    Check if target can be formed as a sum of a subset of items.
    items: list of Fraction values
    Returns True if possible, False otherwise.
    """
    if target == 0:
        return True
    if not items:
        return False
    
    # Filter items that are <= target (positive values only)
    valid_items = [v for v in items if 0 < v <= target]
    if not valid_items:
        return False
    
    # If any single item equals target
    for v in valid_items:
        if v == target:
            return True
    
    # Sort descending for better pruning
    valid_items.sort(reverse=True)
    
    # Quick check: if sum of all valid items < target, impossible
    if sum(valid_items) < target:
        return False
    
    n = len(valid_items)
    
    # Use meet-in-the-middle for moderate n
    if n <= 20:
        # Split into two halves
        mid = n // 2
        left = valid_items[:mid]
        right = valid_items[mid:]
        
        # Generate all subset sums for left half
        left_sums = set([Fraction(0)])
        for v in left:
            new_sums = set()
            for s in left_sums:
                new_sums.add(s + v)
            left_sums = left_sums | new_sums
        
        # For each sum in right, check if target - sum is in left_sums
        right_sums = set([Fraction(0)])
        for v in right:
            new_sums = set()
            for s in right_sums:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            right_sums = right_sums | new_sums
        
        for rs in right_sums:
            if (target - rs) in left_sums:
                return True
        
        return False
    else:
        # For larger sets, use DFS with pruning
        suffix_sums = [Fraction(0)] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + valid_items[i]
        
        def dfs(idx, current_sum):
            if current_sum == target:
                return True
            if idx >= n:
                return False
            if current_sum > target:
                return False
            if current_sum + suffix_sums[idx] < target:
                return False
            
            # Include
            if current_sum + valid_items[idx] <= target:
                if dfs(idx + 1, current_sum + valid_items[idx]):
                    return True
            
            # Exclude
            if dfs(idx + 1, current_sum):
                return True
            
            return False
        
        return dfs(0, Fraction(0))

def _check_violation(S, a):
    """
    Check if 1/a can be formed by a subset of {1/x for x in S, x != a}.
    Returns True if violation exists, False otherwise.
    """
    target = Fraction(1, a)
    items = []
    for x in S:
        if x != a:
            val = Fraction(1, x)
            if val < target:
                items.append(val)
            elif val == target:
                return True
    
    if not items:
        return False
    
    return _can_form_target(target, items)

def _is_valid_set(S):
    """Check if S is subsum-free."""
    for a in S:
        if _check_violation(S, a):
            return False
    return True

def _count_violations(S):
    """Count the number of elements in S that have a violation."""
    cnt = 0
    for a in S:
        if _check_violation(S, a):
            cnt += 1
    return cnt

def _greedy_construct(order):
    """
    Greedily construct a valid set by trying elements in the given order.
    """
    S = []
    for x in order:
        if _is_valid_set(S + [x]):
            S.append(x)
    return S

def _greedy_remove():
    """
    Start from [1..24] and remove elements to make it valid.
    Prefer removing elements that cause the most violations.
    """
    cur = list(range(1, 25))
    while not _is_valid_set(cur) and _time_left() > 5.0:
        # Find element whose removal reduces violations the most
        base_v = _count_violations(cur)
        if base_v == 0:
            break
        
        best_el = None
        best_after = base_v
        for v in sorted(cur, reverse=True):
            trial = [x for x in cur if x != v]
            a = _count_violations(trial)
            if a < best_after:
                best_after = a
                best_el = v
                if a == 0:
                    break
        
        if best_el is None:
            break
        cur = [x for x in cur if x != best_el]
    
    return cur

def _simulated_annealing(initial_set, max_iter=500):
    """
    Simulated annealing to find a larger valid set.
    State: a set of elements (represented as a frozenset or list)
    Objective: maximize size subject to validity
    Moves: add an element, remove an element, swap an element
    """
    S = list(initial_set)
    current_size = len(S)
    best_S = list(S)
    best_size = current_size
    
    if not _is_valid_set(S):
        # If initial set is invalid, try to fix it
        while not _is_valid_set(S) and _time_left() > 5.0:
            # Remove a random element that has a violation
            violating = [a for a in S if _check_violation(S, a)]
            if not violating:
                break
            v = random.choice(violating)
            S.remove(v)
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = list(S)
    
    T = 10.0
    cooling = 0.995
    rng = random.Random(42)
    
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Generate a random move
        move_type = rng.random()
        if move_type < 0.4:
            # Add an element
            candidates = [x for x in range(1, 25) if x not in S]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_S = S + [x]
            new_size = len(new_S)
            # Energy: penalty for invalidity + bonus for size
            if _is_valid_set(new_S):
                new_energy = -new_size
            else:
                new_energy = -new_size + 10 * _count_violations(new_S)
            
            if _is_valid_set(S):
                old_energy = -current_size
            else:
                old_energy = -current_size + 10 * _count_violations(S)
            
            delta = new_energy - old_energy
            if delta < 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
                S = new_S
                current_size = new_size
                if _is_valid_set(S) and current_size > best_size:
                    best_size = current_size
                    best_S = list(S)
        elif move_type < 0.7:
            # Remove an element
            if not S:
                continue
            x = rng.choice(S)
            new_S = [v for v in S if v != x]
            new_size = len(new_S)
            if _is_valid_set(new_S):
                new_energy = -new_size
            else:
                new_energy = -new_size + 10 * _count_violations(new_S)
            
            if _is_valid_set(S):
                old_energy = -current_size
            else:
                old_energy = -current_size + 10 * _count_violations(S)
            
            delta = new_energy - old_energy
            if delta < 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
                S = new_S
                current_size = new_size
                if _is_valid_set(S) and current_size > best_size:
                    best_size = current_size
                    best_S = list(S)
        else:
            # Swap: remove one, add another
            if not S:
                continue
            x = rng.choice(S)
            candidates = [v for v in range(1, 25) if v not in S]
            if not candidates:
                continue
            y = rng.choice(candidates)
            new_S = [v for v in S if v != x] + [y]
            new_size = len(new_S)
            if _is_valid_set(new_S):
                new_energy = -new_size
            else:
                new_energy = -new_size + 10 * _count_violations(new_S)
            
            if _is_valid_set(S):
                old_energy = -current_size
            else:
                old_energy = -current_size + 10 * _count_violations(S)
            
            delta = new_energy - old_energy
            if delta < 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
                S = new_S
                current_size = new_size
                if _is_valid_set(S) and current_size > best_size:
                    best_size = current_size
                    best_S = list(S)
        
        T *= cooling
    
    return best_S

def _local_search(S):
    """
    Try to improve S by removing one element and trying to add others.
    """
    S = list(S)
    improved = True
    while improved and _time_left() > 3.0:
        improved = False
        for i, a in enumerate(S):
            # Try removing a
            S_without_a = S[:i] + S[i+1:]
            # Try adding elements not in S
            for x in range(1, 25):
                if x not in S_without_a and x != a:
                    new_S = S_without_a + [x]
                    if _is_valid_set(new_S) and len(new_S) > len(S):
                        S = new_S
                        improved = True
                        break
            if improved:
                break
    return S

def construct():
    best_set = None
    best_size = 0
    
    def consider(S):
        nonlocal best_set, best_size
        if not S:
            return
        if len(S) <= best_size:
            return
        if _is_valid_set(S):
            best_size = len(S)
            best_set = list(S)
    
    # 1) Structural anchors
    consider(list(range(13, 25)))  # Seed
    for start in range(2, 25):
        if _time_left() < 5.0:
            break
        consider(list(range(start, 25)))
    
    # 2) Greedy construction from various orders
    orders = [
        list(range(24, 0, -1)),  # Large to small
        list(range(1, 25)),      # Small to large
        list(range(1, 25, 2)) + list(range(2, 25, 2)),  # Odds then evens
        [24, 12, 23, 11, 22, 10, 21, 9, 20, 8, 19, 7, 18, 6, 17, 5, 16, 4, 15, 3, 14, 2, 13, 1],
    ]
    
    for order in orders:
        if _time_left() < 5.0:
            break
        S = _greedy_construct(order)
        consider(S)
        # Local search
        S = _local_search(S)
        consider(S)
    
    # 3) Random restarts with greedy
    rng = random.Random(12345)
    restart_count = 0
    while _time_left() > 10.0 and restart_count < 100:
        order = list(range(1, 25))
        rng.shuffle(order)
        S = _greedy_construct(order)
        consider(S)
        S = _local_search(S)
        consider(S)
        restart_count += 1
    
    # 4) Greedy removal from full set
    if _time_left() > 10.0:
        S = _greedy_remove()
        consider(S)
        S = _local_search(S)
        consider(S)
    
    # 5) Simulated annealing from best found so far
    if _time_left() > 15.0 and best_set:
        S = _simulated_annealing(best_set, max_iter=200)
        consider(S)
    
    # 6) More random restarts if time permits
    restart_count = 0
    while _time_left() > 5.0 and restart_count < 200:
        order = list(range(1, 25))
        rng.shuffle(order)
        S = _greedy_construct(order)
        consider(S)
        S = _local_search(S)
        consider(S)
        restart_count += 1
    
    if best_set is None:
        best_set = list(range(13, 25))
    
    return best_set
