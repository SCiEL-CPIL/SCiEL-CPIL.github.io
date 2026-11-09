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
    
    return triple_set, conflicts_with

def _can_add(element, current_set, conflicts_with):
    """Check if element can be added to current_set without creating a conflict triple."""
    for y, z in conflicts_with[element]:
        if y in current_set and z in current_set:
            return False
    return True

def _min_degree_greedy(all_elements, conflicts_with, time_limit):
    """Minimum-degree greedy: at each step pick the element with fewest conflicts among remaining."""
    deadline = time.time() + time_limit
    current_set = set()
    remaining = set(all_elements)
    
    while remaining and time.time() < deadline:
        # Find element with minimum "effective degree" (number of conflict pairs both in current_set or that would conflict)
        best_elem = None
        best_degree = float('inf')
        
        for elem in remaining:
            # Count how many conflict pairs of elem have both partners in current_set
            # This is the "blocking degree" - if 0, elem can be added
            # If > 0, it's blocked. We want elements that are least blocked.
            blocked = 0
            for y, z in conflicts_with[elem]:
                if y in current_set and z in current_set:
                    blocked += 1
                    if blocked > best_degree:
                        break
            if blocked < best_degree:
                best_degree = blocked
                best_elem = elem
                if blocked == 0:
                    break  # Found an unblocked element, take it
        
        if best_elem is None:
            break
        
        if best_degree == 0:
            current_set.add(best_elem)
            remaining.discard(best_elem)
        else:
            # All remaining elements are blocked. Try adding the least blocked one anyway?
            # No - that would create a conflict. Stop.
            break
    
    return current_set

def _min_degree_greedy_v2(all_elements, conflicts_with, time_limit):
    """Minimum-degree greedy v2: track dynamic degrees, recompute when set changes."""
    deadline = time.time() + time_limit
    current_set = set()
    remaining = list(all_elements)
    remaining_set = set(remaining)
    
    while remaining and time.time() < deadline:
        # For each remaining element, count how many of its conflict pairs are fully in current_set
        # Pick the one with minimum count (prefer 0)
        best_elem = None
        best_block = float('inf')
        
        for elem in remaining:
            block_count = 0
            for y, z in conflicts_with[elem]:
                if y in current_set and z in current_set:
                    block_count += 1
                    if block_count >= best_block:
                        break
            if block_count < best_block:
                best_block = block_count
                best_elem = elem
                if block_count == 0:
                    break
        
        if best_elem is None or best_block > 0:
            break
        
        current_set.add(best_elem)
        remaining.remove(best_elem)
        remaining_set.discard(best_elem)
    
    return current_set

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

def _local_search_remove1_add2(current_set, all_elements, conflicts_with, time_limit):
    """Remove one element, try to add two others (net +1)."""
    deadline = time.time() + time_limit
    current = set(current_set)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for remove_elem in list(current):
            if time.time() > deadline:
                break
            temp_set = current - {remove_elem}
            
            # Find all addable elements
            addable = []
            for add_elem in all_elements:
                if add_elem in temp_set:
                    continue
                if _can_add(add_elem, temp_set, conflicts_with):
                    addable.append(add_elem)
            
            if len(addable) < 2:
                continue
            
            # Try pairs from addable
            for i in range(len(addable)):
                if time.time() > deadline:
                    break
                a1 = addable[i]
                temp_with_one = temp_set | {a1}
                for j in range(i + 1, len(addable)):
                    a2 = addable[j]
                    if _can_add(a2, temp_with_one, conflicts_with):
                        current = temp_with_one | {a2}
                        improved = True
                        break
                if improved:
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
    
    while time.time() < deadline:
        temp *= 0.997  # slower cooling
        
        move_type = random.random()
        
        if move_type < 0.25 and len(current) > 5:
            # Remove a random element
            remove_elem = random.choice(list(current))
            new_set = current - {remove_elem}
            delta = -1
        elif move_type < 0.65:
            # Add a random element
            candidates = [e for e in all_elements if e not in current]
            if not candidates:
                continue
            add_elem = random.choice(candidates)
            if not _can_add(add_elem, current, conflicts_with):
                continue
            new_set = current | {add_elem}
            delta = 1
        elif move_type < 0.85:
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
            delta = 0
        else:
            # Remove one, add two (net +1)
            if len(current) < 3:
                continue
            remove_elem = random.choice(list(current))
            temp_set = current - {remove_elem}
            candidates = [e for e in all_elements if e not in temp_set]
            if len(candidates) < 2:
                continue
            
            # Pick two random candidates that can be added
            random.shuffle(candidates)
            added = []
            for cand in candidates:
                if len(added) >= 2:
                    break
                if _can_add(cand, temp_set | set(added), conflicts_with):
                    added.append(cand)
            
            if len(added) < 2:
                continue
            
            new_set = temp_set | set(added)
            delta = 1
        
        # Accept/reject
        if delta > 0:
            current = new_set
            if len(current) > best_size:
                best = set(current)
                best_size = len(current)
        elif delta == 0:
            # For same-size moves, accept with some probability to explore
            if random.random() < min(1.0, 0.3 + 0.7 * temp):
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
    
    # Strategy 0: Minimum-degree greedy
    result = _min_degree_greedy_v2(all_elements, conflicts_with, 3)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 1: high first, then low ascending
    high = list(range(91, 181))
    low = list(range(1, 91))
    
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
    for i in range(12):
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
    sorted_by_degree_desc = sorted(all_elements, key=lambda x: -conflicts_counts[x])
    result = _greedy_build(set(), sorted_by_degree_desc, conflicts_with, 2)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 6: Min-degree greedy with random tie-breaking
    for i in range(8):
        if time.time() > deadline:
            break
        # Perturb the degree ordering
        perturbed = all_elements[:]
        random.shuffle(perturbed)
        # Sort by (conflict_count, random_rank)
        random_ranks = {e: random.random() for e in all_elements}
        sorted_perturbed = sorted(all_elements, key=lambda x: (conflict_counts[x], random_ranks[x]))
        result = _greedy_build(set(), sorted_perturbed, conflicts_with, 1)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    return best

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

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    # Precompute conflicts
    triple_set, conflicts_with = _precompute_conflicts()
    all_elements = list(range(1, 181))
    
    random.seed(42)
    
    # Phase 1: Multi-greedy initialization (12s)
    best = _multi_greedy(all_elements, conflicts_with, 12)
    best_size = len(best)
    
    # Phase 2: Local search 1-swap (8s)
    if time.time() < deadline - 58:
        for _ in range(4):
            if time.time() > deadline - 58:
                break
            result = _local_search_1swap(best, all_elements, conflicts_with, 2)
            if len(result) > best_size:
                best = result
                best_size = len(best)
    
    # Phase 3: Remove-1-add-2 local search (10s)
    if time.time() < deadline - 48:
        for _ in range(3):
            if time.time() > deadline - 48:
                break
            result = _local_search_remove1_add2(best, all_elements, conflicts_with, 3)
            if len(result) > best_size:
                best = result
                best_size = len(best)
    
    # Phase 4: Simulated annealing (18s)
    if time.time() < deadline - 30:
        sa_result = _simulated_annealing(best, all_elements, conflicts_with, 18, start_temp=2.0)
        if len(sa_result) > best_size:
            best = sa_result
            best_size = len(best)
    
    # Phase 5: Local search 2-swap fast (12s)
    if time.time() < deadline - 18:
        for _ in range(3):
            if time.time() > deadline - 18:
                break
            result = _local_search_2swap_fast(best, all_elements, conflicts_with, 4)
            if len(result) > best_size:
                best = result
                best_size = len(best)
    
    # Phase 6: More SA with lower temperature (8s)
    if time.time() < deadline - 10:
        sa_result = _simulated_annealing(best, all_elements, conflicts_with, 8, start_temp=0.5)
        if len(sa_result) > best_size:
            best = sa_result
            best_size = len(best)
    
    # Phase 7: Final local search and polish (remaining time)
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

# EVOLVE-BLOCK-END
