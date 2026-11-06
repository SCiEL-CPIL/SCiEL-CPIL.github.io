"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _greedy_construct(order, deadline):
    """Greedy construction following a given order of subsets."""
    chosen_sets = []
    
    for s in order:
        if time.time() > deadline:
            break
        # Check if s conflicts with any pair in chosen_sets
        valid = True
        n_chosen = len(chosen_sets)
        for i in range(n_chosen):
            a = chosen_sets[i]
            as_inter = a & s
            for j in range(i + 1, n_chosen):
                b = chosen_sets[j]
                bs_inter = b & s
                if as_inter != bs_inter:
                    continue
                ab_inter = a & b
                if ab_inter == as_inter:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            chosen_sets.append(s)
    
    return [sorted(s) for s in chosen_sets]

def _sa_improve(initial_family, all_subsets, deadline, max_iters=30000):
    """Simulated annealing to improve the family."""
    current = [set(s) for s in initial_family]
    best = [set(s) for s in initial_family]
    best_size = len(best)
    
    current_set = set(frozenset(s) for s in current)
    not_in_current = [s for s in all_subsets if frozenset(s) not in current_set]
    
    T = 3.0
    T_min = 0.01
    cooling = 0.999
    
    rng = random.Random(777)
    
    iter_count = 0
    while iter_count < max_iters and time.time() < deadline:
        iter_count += 1
        if T < T_min:
            T = T_min
        
        move_type = rng.random()
        
        if move_type < 0.4 and len(not_in_current) > 0:
            # Add a random set
            s = rng.choice(not_in_current)
            s_set = set(s)
            
            conflict = False
            n_cur = len(current)
            for i in range(n_cur):
                a = current[i]
                as_inter = a & s_set
                for j in range(i + 1, n_cur):
                    b = current[j]
                    bs_inter = b & s_set
                    if as_inter != bs_inter:
                        continue
                    ab_inter = a & b
                    if ab_inter == as_inter:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                current.append(s_set)
                current_set.add(frozenset(s_set))
                not_in_current.remove(s)
                new_size = len(current)
                if new_size > best_size:
                    best_size = new_size
                    best = [set(x) for x in current]
        elif move_type < 0.6 and len(current) > 2:
            # Remove a random set (always valid, just reduces size)
            idx = rng.randrange(len(current))
            removed = current.pop(idx)
            current_set.discard(frozenset(removed))
            not_in_current.append(removed)
            # Accept with probability to allow exploration
            if rng.random() >= math.exp(-1.0 / T):
                current.insert(idx, removed)
                current_set.add(frozenset(removed))
                not_in_current.remove(removed)
        else:
            # Swap
            if len(current) == 0 or len(not_in_current) == 0:
                continue
            idx = rng.randrange(len(current))
            removed = current.pop(idx)
            current_set.discard(frozenset(removed))
            
            s = rng.choice(not_in_current)
            s_set = set(s)
            
            conflict = False
            n_cur = len(current)
            for i in range(n_cur):
                a = current[i]
                as_inter = a & s_set
                for j in range(i + 1, n_cur):
                    b = current[j]
                    bs_inter = b & s_set
                    if as_inter != bs_inter:
                        continue
                    ab_inter = a & b
                    if ab_inter == as_inter:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                current.append(s_set)
                current_set.add(frozenset(s_set))
                not_in_current.remove(s)
                not_in_current.append(removed)
                new_size = len(current)
                if new_size > best_size:
                    best_size = new_size
                    best = [set(x) for x in current]
            else:
                current.insert(idx, removed)
                current_set.add(frozenset(removed))
        
        T *= cooling
    
    return [sorted(s) for s in best]

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    all_subsets = [set(row) for row in combinations(range(1, 17), 4)]
    n_total = len(all_subsets)
    
    best_family = []
    
    # Multiple greedy passes
    # Pass 1: Natural order
    family = _greedy_construct(all_subsets, deadline)
    if len(family) > len(best_family):
        best_family = family
    
    if time.time() > deadline:
        return best_family
    
    # Pass 2: Random
    random.seed(42)
    shuffled = all_subsets[:]
    random.shuffle(shuffled)
    family = _greedy_construct(shuffled, deadline)
    if len(family) > len(best_family):
        best_family = family
    
    if time.time() > deadline:
        return best_family
    
    # Pass 3: Random
    random.seed(123)
    shuffled2 = all_subsets[:]
    random.shuffle(shuffled2)
    family = _greedy_construct(shuffled2, deadline)
    if len(family) > len(best_family):
        best_family = family
    
    if time.time() > deadline:
        return best_family
    
    # Pass 4: Order by pairwise intersection-2 count (descending)
    inter2_count = [0] * n_total
    for i in range(n_total):
        si = all_subsets[i]
        for j in range(i + 1, n_total):
            if len(si & all_subsets[j]) == 2:
                inter2_count[i] += 1
                inter2_count[j] += 1
    indexed = list(range(n_total))
    indexed.sort(key=lambda x: -inter2_count[x])
    order4 = [all_subsets[i] for i in indexed]
    family = _greedy_construct(order4, deadline)
    if len(family) > len(best_family):
        best_family = family
    
    if time.time() > deadline:
        return best_family
    
    # Pass 5: Reverse order
    order5 = [all_subsets[i] for i in reversed(indexed)]
    family = _greedy_construct(order5, deadline)
    if len(family) > len(best_family):
        best_family = family
    
    if time.time() > deadline:
        return best_family
    
    # Simulated annealing refinement
    if len(best_family) > 0:
        sa_deadline = deadline - 3
        sa_result = _sa_improve(best_family, all_subsets, sa_deadline, max_iters=30000)
        if len(sa_result) > len(best_family):
            best_family = sa_result
    
    return best_family
# EVOLVE-BLOCK-END
