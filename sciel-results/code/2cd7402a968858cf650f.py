import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _precompute_conflicts():
    """Precompute all conflict triples and build efficient conflict structures."""
    triple_set = set()
    
    for a in range(1, 181):
        a2 = a * a
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                for d_val in [d, a2 // d]:
                    if d_val == a:
                        continue
                    b = a + d_val
                    c = a + a2 // d_val
                    if b == c:
                        continue
                    if b < 1 or b > 180 or c < 1 or c > 180:
                        continue
                    if b == a or c == a:
                        continue
                    triple = tuple(sorted((a, b, c)))
                    triple_set.add(triple)
    
    # conflicts_with[x] = set of frozenset({y,z}) such that {x,y,z} is a conflict triple
    conflicts_with = {i: set() for i in range(1, 181)}
    for (a, b, c) in triple_set:
        conflicts_with[a].add(frozenset((b, c)))
        conflicts_with[b].add(frozenset((a, c)))
        conflicts_with[c].add(frozenset((a, b)))
    
    # incompatible_pairs[x] = set of y such that there exists z with {x,y,z} conflict
    # This helps with faster pair-based checks
    incompatible_pairs = {i: set() for i in range(1, 181)}
    for (a, b, c) in triple_set:
        incompatible_pairs[a].add(b)
        incompatible_pairs[a].add(c)
        incompatible_pairs[b].add(a)
        incompatible_pairs[b].add(c)
        incompatible_pairs[c].add(a)
        incompatible_pairs[c].add(b)
    
    return triple_set, conflicts_with, incompatible_pairs

def _can_add(element, current_set, conflicts_with):
    """Check if element can be added to current_set without creating a conflict triple."""
    for pair in conflicts_with[element]:
        y, z = tuple(pair)
        if y in current_set and z in current_set:
            return False
    return True

def _greedy_build(start_set, candidates, conflicts_with, deadline):
    """Greedily add candidates to start_set if validity is maintained."""
    current_set = set(start_set)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in current_set:
            continue
        if _can_add(cand, current_set, conflicts_with):
            current_set.add(cand)
    
    return current_set

def _local_search_1swap(current_set, all_elements, conflicts_with, deadline):
    """Try 1-vertex swaps: remove one element, try to add others."""
    current = set(current_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for remove_elem in list(current):
            if time.time() > deadline:
                break
            temp_set = current - {remove_elem}
            for add_elem in all_elements:
                if add_elem in temp_set:
                    continue
                if _can_add(add_elem, temp_set, conflicts_with):
                    new_set = temp_set | {add_elem}
                    if len(new_set) > len(current):
                        current = new_set
                        improved = True
                        break
    
    return current

def _local_search_2swap_fast(current_set, all_elements, conflicts_with, deadline):
    """Optimized 2-vertex swap local search with early termination."""
    current = set(current_set)
    current_list = list(current)
    n = len(current_list)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_list = list(current)
        n = len(current_list)
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                remove_pair = {current_list[i], current_list[j]}
                temp_set = current - remove_pair
                
                # Try adding pairs
                found = False
                for add_elem in all_elements:
                    if time.time() > deadline:
                        found = True  # break outer loops
                        break
                    if add_elem in temp_set:
                        continue
                    if not _can_add(add_elem, temp_set, conflicts_with):
                        continue
                    temp_with_one = temp_set | {add_elem}
                    for add_elem2 in all_elements:
                        if time.time() > deadline:
                            found = True
                            break
                        if add_elem2 in temp_with_one or add_elem2 == add_elem:
                            continue
                        if _can_add(add_elem2, temp_with_one, conflicts_with):
                            new_set = temp_with_one | {add_elem2}
                            if len(new_set) > len(current):
                                current = new_set
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if improved:
                break
    
    return current

def _simulated_annealing(current_set, all_elements, conflicts_with, deadline):
    """Simulated annealing to escape local optima."""
    current = set(current_set)
    current_size = len(current)
    best = set(current)
    best_size = current_size
    
    temp = 5.0
    cooling = 0.98
    
    while time.time() < deadline and temp > 0.1:
        # Pick a random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Try adding a random element
            candidates = [x for x in all_elements if x not in current]
            if not candidates:
                temp *= cooling
                continue
            add_elem = random.choice(candidates)
            if _can_add(add_elem, current, conflicts_with):
                new_set = current | {add_elem}
                delta = 1
            else:
                # Can't add directly, try removing one conflicting element
                new_set = None
                delta = -1
                for pair in conflicts_with[add_elem]:
                    y, z = tuple(pair)
                    if y in current and z in current:
                        # Remove one of y, z
                        for rem in [y, z]:
                            temp_set = current - {rem}
                            if _can_add(add_elem, temp_set, conflicts_with):
                                new_set = temp_set | {add_elem}
                                delta = 0  # same size
                                break
                        if new_set:
                            break
        elif move_type < 0.7:
            # Try removing a random element
            if not current:
                temp *= cooling
                continue
            rem_elem = random.choice(list(current))
            new_set = current - {rem_elem}
            delta = -1
        else:
            # Try a swap
            if not current:
                temp *= cooling
                continue
            rem_elem = random.choice(list(current))
            temp_set = current - {rem_elem}
            candidates = [x for x in all_elements if x not in temp_set]
            if not candidates:
                temp *= cooling
                continue
            add_elem = random.choice(candidates)
            if _can_add(add_elem, temp_set, conflicts_with):
                new_set = temp_set | {add_elem}
                delta = 0
            else:
                temp *= cooling
                continue
        
        if new_set is None:
            temp *= cooling
            continue
        
        new_size = len(new_set)
        delta = new_size - current_size
        
        # Accept if better or with probability based on temperature
        if delta >= 0:
            current = new_set
            if new_size > best_size:
                best = new_set
                best_size = new_size
        else:
            prob = math.exp(delta / temp)
            if random.random() < prob:
                current = new_set
        
        temp *= cooling
    
    return best

def construct():
    deadline = time.time() + 82  # 82s limit, 8s margin
    
    # Precompute conflicts
    triple_set, conflicts_with, incompatible_pairs = _precompute_conflicts()
    all_elements = list(range(1, 181))
    
    random.seed(42)
    
    # Start with high numbers (less likely to be in conflict triples)
    best = set(range(91, 181))
    best_size = len(best)
    
    # Greedy add from low numbers in various orders
    low_candidates = list(range(1, 91))
    
    # Multiple greedy passes with different orderings
    greedy_deadline = deadline - 50
    
    # Order 1: ascending
    result = _greedy_build(best, sorted(low_candidates), conflicts_with, greedy_deadline)
    if len(result) > best_size:
        best = result
        best_size = len(best)
    
    # Order 2: descending
    result = _greedy_build(best, sorted(low_candidates, reverse=True), conflicts_with, greedy_deadline)
    if len(result) > best_size:
        best = result
        best_size = len(best)
    
    # Random orders - more iterations
    for i in range(50):
        if time.time() > greedy_deadline:
            break
        shuffled = low_candidates[:]
        random.shuffle(shuffled)
        result = _greedy_build(best, shuffled, conflicts_with, greedy_deadline)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Local search: 1-swap
    ls1_deadline = deadline - 30
    improved = True
    while improved and time.time() < ls1_deadline:
        improved = False
        result = _local_search_1swap(best, all_elements, conflicts_with, min(time.time() + 3, ls1_deadline))
        if len(result) > best_size:
            best = result
            best_size = len(best)
            improved = True
    
    # Local search: 2-swap
    ls2_deadline = deadline - 15
    improved = True
    while improved and time.time() < ls2_deadline:
        improved = False
        result = _local_search_2swap_fast(best, all_elements, conflicts_with, min(time.time() + 4, ls2_deadline))
        if len(result) > best_size:
            best = result
            best_size = len(best)
            improved = True
    
    # Simulated annealing
    sa_deadline = deadline - 5
    if time.time() < sa_deadline:
        result = _simulated_annealing(best, all_elements, conflicts_with, sa_deadline)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Final polish: try all single additions
    if time.time() < deadline - 1:
        for elem in all_elements:
            if elem in best:
                continue
            if _can_add(elem, best, conflicts_with):
                best.add(elem)
                best_size = len(best)
    
    return list(best)

# EVOLVE-BLOCK-END
