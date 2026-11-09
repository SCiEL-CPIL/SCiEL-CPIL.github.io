import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _precompute_conflicts():
    """Precompute all conflict triples (a,b,c) where 1/a = 1/b + 1/c with distinct a,b,c in [1,180]."""
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
    
    # conflicts_with[x] = list of (y, z) such that {x,y,z} is a conflict triple
    conflicts_with = {i: [] for i in range(1, 181)}
    for (a, b, c) in triple_set:
        conflicts_with[a].append((b, c))
        conflicts_with[b].append((a, c))
        conflicts_with[c].append((a, b))
    
    # Also build: for each pair (x,y), the set of z such that {x,y,z} is a conflict triple
    # This helps with: given S, can we add x? Check if any (y,z) in conflicts_with[x] has y,z in S.
    # Alternative: precompute for each x, a bitmask or set-based check.
    
    return triple_set, conflicts_with

def _can_add(element, current_set, conflicts_with):
    """Check if element can be added to current_set without creating a conflict triple."""
    for y, z in conflicts_with[element]:
        if y in current_set and z in current_set:
            return False
    return True

def _greedy_build(start_set, candidates, conflicts_with, time_limit):
    """Greedily add candidates to start_set if validity is maintained."""
    current_set = set(start_set)
    deadline = time.time() + time_limit
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in current_set:
            continue
        if _can_add(cand, current_set, conflicts_with):
            current_set.add(cand)
    
    return current_set

def _local_search_1swap(current_set, all_elements, conflicts_with, time_limit):
    """Try 1-vertex swaps: remove one element, try to add others."""
    deadline = time.time() + time_limit
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
            if improved:
                break
    
    return current

def _simulated_annealing(current_set, all_elements, conflicts_with, time_limit, start_temp=1.0):
    """Simulated annealing over add/remove/swap moves."""
    deadline = time.time() + time_limit
    current = set(current_set)
    best = set(current)
    best_size = len(current)
    temp = start_temp
    
    # Precompute: for each element, count how many conflict pairs it has
    # to help with move selection
    conflict_count = {e: len(conflicts_with[e]) for e in range(1, 181)}
    
    while time.time() < deadline:
        temp *= 0.995  # cooling
        
        # Choose a random move type
        move_type = random.random()
        
        if move_type < 0.3 and len(current) > 5:
            # Remove a random element
            remove_elem = random.choice(list(current))
            new_set = current - {remove_elem}
            delta = -1  # size change
        elif move_type < 0.7:
            # Add a random element
            candidates = [e for e in all_elements if e not in current]
            if not candidates:
                continue
            add_elem = random.choice(candidates)
            if not _can_add(add_elem, current, conflicts_with):
                continue
            new_set = current | {add_elem}
            delta = 1
        else:
            # Swap: remove one, add one
            if not current:
                continue
            remove_elem = random.choice(list(current))
            candidates = [e for e in all_elements if e not in current and e != remove_elem]
            if not candidates:
                continue
            add_elem = random.choice(candidates)
            temp_set = current - {remove_elem}
            if not _can_add(add_elem, temp_set, conflicts_with):
                continue
            new_set = temp_set | {add_elem}
            delta = 0  # same size
        
        # Accept/reject
        if delta > 0:
            current = new_set
            if len(current) > best_size:
                best = set(current)
                best_size = len(current)
        elif delta == 0:
            # For same-size moves, accept with probability based on temperature
            # Use a heuristic: prefer moves that reduce "conflict potential"
            # Simple: accept with probability exp(1/temp) which is > 1, so always accept
            # Better: accept with some probability to explore
            if random.random() < math.exp(0.5 / max(temp, 0.01)):
                current = new_set
        else:
            # delta = -1, accept with probability exp(delta/temp)
            prob = math.exp(delta / max(temp, 0.01))
            if random.random() < prob:
                current = new_set
    
    return best

def _local_search_2swap_fast(current_set, all_elements, conflicts_with, time_limit):
    """Optimized 2-swap: remove two, add two. Uses early termination."""
    deadline = time.time() + time_limit
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
                remove_pair = (current_list[i], current_list[j])
                temp_set = current - {remove_pair[0], remove_pair[1]}
                
                # Find all elements that can be added to temp_set
                addable = []
                for add_elem in all_elements:
                    if add_elem in temp_set:
                        continue
                    if _can_add(add_elem, temp_set, conflicts_with):
                        addable.append(add_elem)
                
                if len(addable) < 2:
                    continue
                
                # Try pairs from addable
                for ai in range(len(addable)):
                    if time.time() > deadline:
                        break
                    a1 = addable[ai]
                    temp_with_one = temp_set | {a1}
                    for aj in range(ai + 1, len(addable)):
                        a2 = addable[aj]
                        if _can_add(a2, temp_with_one, conflicts_with):
                            current = temp_with_one | {a2}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return current

def _multi_greedy(all_elements, conflicts_with, time_limit):
    """Multiple greedy constructions with different orderings."""
    deadline = time.time() + time_limit
    best = set()
    best_size = 0
    
    # Start with high numbers (fewer conflicts)
    high = list(range(91, 181))
    low = list(range(1, 91))
    
    # Strategy 1: high first, then low ascending
    result = _greedy_build(set(high), sorted(low), conflicts_with, 2)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 2: high first, then low descending
    result = _greedy_build(set(high), sorted(low, reverse=True), conflicts_with, 2)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 3: all elements in random order
    for i in range(15):
        if time.time() > deadline:
            break
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        result = _greedy_build(set(), shuffled, conflicts_with, 1)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Strategy 4: degree-based ordering (elements with fewer conflicts first)
    conflict_counts = {e: len(conflicts_with[e]) for e in range(1, 181)}
    sorted_by_degree = sorted(all_elements, key=lambda x: conflict_counts[x])
    result = _greedy_build(set(), sorted_by_degree, conflicts_with, 2)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 5: reverse degree
    sorted_by_degree_desc = sorted(all_elements, key=lambda x: -conflict_counts[x])
    result = _greedy_build(set(), sorted_by_degree_desc, conflicts_with, 2)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    return best

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    # Precompute conflicts
    triple_set, conflicts_with = _precompute_conflicts()
    all_elements = list(range(1, 181))
    
    random.seed(42)
    
    # Phase 1: Multi-greedy initialization (10s)
    best = _multi_greedy(all_elements, conflicts_with, 10)
    best_size = len(best)
    
    # Phase 2: Local search 1-swap (10s)
    if time.time() < deadline - 60:
        for _ in range(5):
            if time.time() > deadline - 60:
                break
            result = _local_search_1swap(best, all_elements, conflicts_with, 2)
            if len(result) > best_size:
                best = result
                best_size = len(best)
    
    # Phase 3: Simulated annealing (20s)
    if time.time() < deadline - 30:
        sa_result = _simulated_annealing(best, all_elements, conflicts_with, 20, start_temp=2.0)
        if len(sa_result) > best_size:
            best = sa_result
            best_size = len(best)
    
    # Phase 4: Local search 2-swap fast (15s)
    if time.time() < deadline - 15:
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            result = _local_search_2swap_fast(best, all_elements, conflicts_with, 5)
            if len(result) > best_size:
                best = result
                best_size = len(best)
    
    # Phase 5: More SA with lower temperature (10s)
    if time.time() < deadline - 5:
        sa_result = _simulated_annealing(best, all_elements, conflicts_with, 10, start_temp=0.5)
        if len(sa_result) > best_size:
            best = sa_result
            best_size = len(best)
    
    # Phase 6: Final local search and polish (remaining time)
    remaining = deadline - time.time() - 2
    if remaining > 3:
        result = _local_search_1swap(best, all_elements, conflicts_with, min(3, remaining))
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
    
    # Verify validity
    if not _fast_check_validity(best):
        # Fallback: rebuild greedily
        best = _greedy_build(set(all_elements), sorted(all_elements, reverse=True), conflicts_with, 1)
    
    return list(best)

def _fast_check_validity(s):
    """Verify that set s has no conflict triple."""
    s_set = set(s)
    for a in s_set:
        a2 = a * a
        for b in s_set:
            if b <= a:
                continue
            d = b - a
            if a2 % d != 0:
                continue
            c = a + a2 // d
            if c == a or c == b:
                continue
            if c in s_set:
                return False
    return True

# EVOLVE-BLOCK-END
