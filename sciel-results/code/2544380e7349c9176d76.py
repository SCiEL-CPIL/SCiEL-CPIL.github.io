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
    
    # conflict_degree[x] = number of conflict pairs involving x
    conflict_degree = {i: len(conflicts_with[i]) for i in range(1, 181)}
    
    # incompatible_pairs[x] = set of y such that there exists z with {x,y,z} conflict
    incompatible_pairs = {i: set() for i in range(1, 181)}
    for (a, b, c) in triple_set:
        incompatible_pairs[a].add(b)
        incompatible_pairs[a].add(c)
        incompatible_pairs[b].add(a)
        incompatible_pairs[b].add(c)
        incompatible_pairs[c].add(a)
        incompatible_pairs[c].add(b)
    
    return triple_set, conflicts_with, incompatible_pairs, conflict_degree

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

def _repair_search(current_set, all_elements, conflicts_with, conflict_degree, deadline):
    """Targeted repair: for each missing element, find minimal removals to allow addition."""
    current = set(current_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort missing elements by how "easy" they are to add (fewer conflicts)
        missing = [x for x in all_elements if x not in current]
        missing.sort(key=lambda x: conflict_degree[x])
        
        for elem in missing:
            if time.time() > deadline:
                break
            if elem in current:
                continue
            
            # Find all blocking pairs
            blockers = []
            for pair in conflicts_with[elem]:
                y, z = tuple(pair)
                if y in current and z in current:
                    blockers.append((y, z))
            
            if not blockers:
                current.add(elem)
                improved = True
                continue
            
            # Try removing 1 element to unblock
            best_removal = None
            for (y, z) in blockers:
                for rem in [y, z]:
                    temp = current - {rem}
                    if _can_add(elem, temp, conflicts_with):
                        # Check if this removal doesn't create other issues
                        # (it won't since we're only removing)
                        if best_removal is None or len(temp | {elem}) > len(current):
                            best_removal = rem
                        break
            
            if best_removal is not None:
                current = current - {best_removal}
                current.add(elem)
                improved = True
        
        # Also try: remove one element, then add multiple
        for rem in list(current):
            if time.time() > deadline:
                break
            temp = current - {rem}
            added = 0
            for elem in missing:
                if time.time() > deadline:
                    break
                if elem in temp:
                    continue
                if _can_add(elem, temp, conflicts_with):
                    temp.add(elem)
                    added += 1
            if added > 1:
                current = temp
                improved = True
    
    return current

def _backtracking_improve(current_set, all_elements, conflicts_with, deadline):
    """CSP-style: try to add a batch of elements using backtracking with forward checking."""
    current = set(current_set)
    missing = [x for x in all_elements if x not in current]
    
    # Sort by conflict degree (fewest constraints first)
    missing.sort(key=lambda x: len(conflicts_with[x]))
    
    best = set(current)
    best_size = len(current)
    
    def backtrack(remaining, current_set, depth, deadline):
        if time.time() > deadline or depth > 30:
            return current_set
        
        if not remaining:
            return current_set
        
        # Try adding the first element
        elem = remaining[0]
        rest = remaining[1:]
        
        if _can_add(elem, current_set, conflicts_with):
            # Forward check: see which remaining elements are still addable
            new_set = current_set | {elem}
            new_remaining = [x for x in rest if _can_add(x, new_set, conflicts_with)]
            result = backtrack(new_remaining, new_set, depth + 1, deadline)
            if len(result) >= len(current_set):
                return result
        
        # Skip this element
        return backtrack(rest, current_set, depth + 1, deadline)
    
    # Try multiple batches
    for batch_size in [5, 8, 12]:
        if time.time() > deadline:
            break
        for start in range(0, len(missing) - batch_size, batch_size):
            if time.time() > deadline:
                break
            batch = missing[start:start + batch_size]
            result = backtrack(batch, current, 0, deadline)
            if len(result) > best_size:
                best = result
                best_size = len(result)
                current = result
                # Recompute missing
                missing = [x for x in all_elements if x not in current]
                missing.sort(key=lambda x: len(conflicts_with[x]))
    
    return best

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
            if improved:
                break
    
    return current

def _simulated_annealing(current_set, all_elements, conflicts_with, deadline):
    """Simulated annealing to escape local optima."""
    current = set(current_set)
    best = set(current)
    best_size = len(current)
    
    temp = 5.0
    cooling = 0.98
    
    while time.time() < deadline and temp > 0.1:
        move_type = random.random()
        
        if move_type < 0.4:
            candidates = [x for x in all_elements if x not in current]
            if not candidates:
                temp *= cooling
                continue
            add_elem = random.choice(candidates)
            if _can_add(add_elem, current, conflicts_with):
                new_set = current | {add_elem}
                delta = 1
            else:
                new_set = None
                delta = -1
                for pair in conflicts_with[add_elem]:
                    y, z = tuple(pair)
                    if y in current and z in current:
                        for rem in [y, z]:
                            temp_set = current - {rem}
                            if _can_add(add_elem, temp_set, conflicts_with):
                                new_set = temp_set | {add_elem}
                                delta = 0
                                break
                        if new_set:
                            break
        elif move_type < 0.7:
            if not current:
                temp *= cooling
                continue
            rem_elem = random.choice(list(current))
            new_set = current - {rem_elem}
            delta = -1
        else:
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
        delta = new_size - len(current)
        
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
    deadline = time.time() + 80  # 80s limit, 10s margin
    
    # Precompute conflicts
    triple_set, conflicts_with, incompatible_pairs, conflict_degree = _precompute_conflicts()
    all_elements = list(range(1, 181))
    
    random.seed(42)
    
    # Start with high numbers (less likely to be in conflict triples)
    best = set(range(91, 181))
    best_size = len(best)
    
    # Greedy add from low numbers sorted by conflict degree (ascending)
    low_candidates = list(range(1, 91))
    low_candidates.sort(key=lambda x: conflict_degree[x])
    
    greedy_deadline = deadline - 55
    
    # Order 1: by conflict degree ascending
    result = _greedy_build(best, low_candidates, conflicts_with, greedy_deadline)
    if len(result) > best_size:
        best = result
        best_size = len(best)
    
    # Order 2: by conflict degree descending
    result = _greedy_build(best, sorted(low_candidates, key=lambda x: -conflict_degree[x]), conflicts_with, greedy_deadline)
    if len(result) > best_size:
        best = result
        best_size = len(best)
    
    # Random orders
    for i in range(30):
        if time.time() > greedy_deadline:
            break
        shuffled = low_candidates[:]
        random.shuffle(shuffled)
        result = _greedy_build(best, shuffled, conflicts_with, greedy_deadline)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Repair search
    repair_deadline = deadline - 30
    if time.time() < repair_deadline:
        result = _repair_search(best, all_elements, conflicts_with, conflict_degree, repair_deadline)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Local search: 1-swap
    ls1_deadline = deadline - 20
    if time.time() < ls1_deadline:
        result = _local_search_1swap(best, all_elements, conflicts_with, min(time.time() + 5, ls1_deadline))
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Backtracking improvement
    bt_deadline = deadline - 10
    if time.time() < bt_deadline:
        result = _backtracking_improve(best, all_elements, conflicts_with, bt_deadline)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Simulated annealing
    sa_deadline = deadline - 5
    if time.time() < sa_deadline:
        result = _simulated_annealing(best, all_elements, conflicts_with, sa_deadline)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Final polish: try all single additions
    if time.time() < deadline - 2:
        for elem in all_elements:
            if elem in best:
                continue
            if _can_add(elem, best, conflicts_with):
                best.add(elem)
                best_size = len(best)
    
    return list(best)

# EVOLVE-BLOCK-END
