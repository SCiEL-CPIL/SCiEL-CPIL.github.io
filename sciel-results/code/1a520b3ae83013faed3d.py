"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill (evaluator limit)
    N = 24
    universe = list(range(1, N + 1))
    
    # Precompute LCM of all numbers 1..24 for exact integer arithmetic
    total_lcm = 1
    for i in range(1, N + 1):
        total_lcm = lcm(total_lcm, i)
    
    # weight[i] = total_lcm // i
    weights = {i: total_lcm // i for i in universe}
    
    best_set = set()
    best_size = 0
    
    # Fast validity check using optimized subset sum generation
    def is_valid(s):
        s_list = list(s)
        n = len(s_list)
        if n <= 1:
            return True
        
        for i, a in enumerate(s_list):
            target = weights[a]
            others = [x for j, x in enumerate(s_list) if j != i]
            m = len(others)
            
            if m == 0:
                continue
            
            # Quick check: if target > sum of all others, impossible
            sum_others = sum(weights[x] for x in others)
            if target > sum_others:
                continue
            
            # Quick check: if target < min weight, impossible (non-empty subset)
            min_w = min(weights[x] for x in others)
            if target < min_w:
                continue
            
            # Meet in the middle with optimized subset sum generation
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Build left sums using iterative set expansion
            left_sums = {0}
            for x in left:
                w = weights[x]
                left_sums = left_sums | {v + w for v in left_sums}
            
            # Build right sums
            right_sums = {0}
            for x in right:
                w = weights[x]
                right_sums = right_sums | {v + w for v in right_sums}
                
            # Check if target can be formed
            for r in right_sums:
                if (target - r) in left_sums:
                    return False
            
        return True

    def greedy_construct(order, initial_set=None):
        current_set_set = set(initial_set) if initial_set else set()
        for elem in order:
            if elem in current_set_set:
                continue
            candidate_set = current_set_set | {elem}
            if is_valid(candidate_set):
                current_set_set.add(elem)
        return current_set_set

    def try_extend(base_set, remaining, order):
        """Try to extend a base set by adding elements from remaining in given order."""
        current_set = set(base_set)
        for elem in order:
            if elem in current_set or elem not in remaining:
                continue
            candidate_set = current_set | {elem}
            if is_valid(candidate_set):
                current_set.add(elem)
        return current_set

    def local_search_improve(s):
        """Try to improve set s by adding elements or swapping."""
        s = set(s)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # 1. Try to add any element
            for add_elem in universe:
                if add_elem in s:
                    continue
                candidate = s | {add_elem}
                if is_valid(candidate):
                    s = candidate
                    improved = True
                    break # Restart loop with new s
            
            # 2. If no simple add worked, try 1-for-2 swap
            if not improved:
                for remove_elem in list(s):
                    if time.time() > deadline - 2:
                        return s
                    reduced = s - {remove_elem}
                    # Try adding two elements
                    for add1 in universe:
                        if add1 in reduced:
                            continue
                        for add2 in universe:
                            if add2 in reduced or add2 == add1:
                                continue
                            candidate = reduced | {add1, add2}
                            if len(candidate) > len(s):
                                if is_valid(candidate):
                                    s = candidate
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        return s

    # 1. Deterministic Greedy: Descending
    order_desc = sorted(universe, reverse=True)
    set_desc = greedy_construct(order_desc)
    if len(set_desc) > best_size:
        best_size = len(set_desc)
        best_set = set(set_desc)
        
    # 2. Deterministic Greedy: Ascending
    order_asc = sorted(universe)
    set_asc = greedy_construct(order_asc)
    if len(set_asc) > best_size:
        best_size = len(set_asc)
        best_set = set(set_asc)
    
    # 3. Structured initialization
    core_starts = [
        list(range(12, 25)),
        list(range(10, 25)),
        list(range(8, 25)),
        list(range(7, 25)),
    ]
    
    for core in core_starts:
        if time.time() > deadline - 5:
            break
        core_set = set(core)
        if is_valid(core_set):
            if len(core_set) > best_size:
                best_size = len(core_set)
                best_set = set(core_set)
            remaining = [x for x in range(1, min(core)) if x not in core_set]
            for r_order in [sorted(remaining, reverse=True), sorted(remaining), remaining]:
                extended = try_extend(core_set, set(remaining), r_order)
                if len(extended) > best_size:
                    best_size = len(extended)
                    best_set = set(extended)
    
    # 4. Randomized Greedy Restart with diverse strategies
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        strategy = rng.random()
        
        if strategy < 0.3:
            order = universe[:]
            rng.shuffle(order)
        elif strategy < 0.6:
            order = sorted(universe, reverse=True)
            for _ in range(rng.randint(1, 5)):
                i, j = rng.sample(range(len(order)), 2)
                order[i], order[j] = order[j], order[i]
        elif strategy < 0.8:
            core_size = rng.randint(8, 14)
            core = rng.sample(range(core_size + 1, 25), min(core_size, 24 - core_size))
            rest = [x for x in universe if x not in core]
            rng.shuffle(rest)
            order = core + rest
        else:
            order = sorted(universe, key=lambda x: -x + rng.gauss(0, 5))
        
        s = greedy_construct(order)
        # Apply local search to the greedy result
        s = local_search_improve(s)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    # 5. Final Local Search on best_set
    best_set = local_search_improve(best_set)
    best_size = len(best_set)
    
    return list(best_set)
# EVOLVE-BLOCK-END
