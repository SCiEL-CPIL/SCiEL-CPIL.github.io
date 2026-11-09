import time
import random
from itertools import combinations

def _check_validity(selected):
    """Check if the entire set is valid."""
    n = len(selected)
    # For each a in selected
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        # Check all combinations of 3 distinct others
        for u in range(len(others)):
            for v in range(u+1, len(others)):
                for w in range(v+1, len(others)):
                    s = others[u] + others[v] + others[w]
                    if s % a == 0:
                        return False
    return True

def _count_violations_for_element(a, selected):
    """Count violations involving a as the divisor."""
    count = 0
    others = [x for x in selected if x != a]
    for triplet in combinations(others, 3):
        if sum(triplet) % a == 0:
            count += 1
    return count

def _find_violation_triplets(selected):
    """Find all (a, triplet) violations."""
    violations = []
    n = len(selected)
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        for triplet in combinations(others, 3):
            if sum(triplet) % a == 0:
                violations.append((a, triplet))
    return violations

def _precompute_conflicts(pool, n=192):
    """
    Precompute for each element a in pool, the set of triplets (b,c,d) from pool
    such that a divides b+c+d. Returns dict: a -> set of frozensets({b,c,d})
    """
    conflicts = {a: set() for a in pool}
    pool_set = set(pool)
    for a in pool:
        others = [x for x in pool if x != a]
        for triplet in combinations(others, 3):
            if sum(triplet) % a == 0:
                conflicts[a].add(frozenset(triplet))
    return conflicts

def _can_add_element(a, current_set, pool):
    """Check if adding a to current_set keeps it valid."""
    # a must not divide sum of any 3 distinct elements in current_set
    for triplet in combinations(current_set, 3):
        if sum(triplet) % a == 0:
            return False
    # Each element in current_set must not divide sum of a + 2 others from current_set
    for b in current_set:
        for triplet in combinations([x for x in current_set if x != b], 2):
            if (a + triplet[0] + triplet[1]) % b == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Pool: 144..192 (49 elements)
    base_pool = list(range(144, 193))
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy construction with conflict-aware ordering
    # Precompute conflicts for the base pool
    conflicts = _precompute_conflicts(base_pool)
    
    for attempt in range(50):
        if time.time() > deadline - 2.0:
            break
        
        # Different random orderings
        pool_order = base_pool[:]
        random.shuffle(pool_order)
        
        current_set = []
        
        # Greedy: add elements in order, but prioritize those with fewer conflicts
        # with already-selected elements
        for a in pool_order:
            if time.time() > deadline - 2.0:
                break
            
            # Check if a can be added
            if not _can_add_element(a, current_set, base_pool):
                continue
            
            current_set.append(a)
            
            if len(current_set) > 48:
                current_set = current_set[:48]
                break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            if best_size == 48:
                return best_set
        
        # Local search: try to swap elements to increase size or fix issues
        if len(current_set) < 48:
            # Try adding elements from a wider pool
            extended_pool = list(range(100, 193))
            extended_pool = [x for x in extended_pool if x not in current_set]
            random.shuffle(extended_pool)
            
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for candidate in extended_pool:
                    if time.time() > deadline - 2.0:
                        break
                    if len(current_set) >= 48:
                        break
                    if candidate in current_set:
                        continue
                    if _can_add_element(candidate, current_set, extended_pool):
                        current_set.append(candidate)
                        improved = True
                        if len(current_set) >= 48:
                            break
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                
                if best_size == 48:
                    return best_set
    
    # If we didn't reach 48, try the removal-based approach as fallback
    if best_size < 48:
        for attempt in range(20):
            if time.time() > deadline - 1.0:
                break
            
            current_set = base_pool[:]
            
            # Removal Loop
            while True:
                if time.time() > deadline - 1.0:
                    break
                
                violations = _find_violation_triplets(current_set)
                
                if not violations:
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
                        if best_size == 48:
                            return best_set
                    break
                
                violation_counts = {}
                for a, triplet in violations:
                    violation_counts[a] = violation_counts.get(a, 0) + 1
                    for x in triplet:
                        violation_counts[x] = violation_counts.get(x, 0) + 1
                
                if not violation_counts:
                    break
                    
                max_count = max(violation_counts.values())
                candidates_to_remove = [k for k, v in violation_counts.items() if v == max_count]
                
                if random.random() < 0.3:
                    all_involved = set()
                    for a, triplet in violations:
                        all_involved.add(a)
                        all_involved.update(triplet)
                    elem_to_remove = random.choice(list(all_involved))
                else:
                    elem_to_remove = random.choice(candidates_to_remove)
                
                if elem_to_remove in current_set:
                    current_set.remove(elem_to_remove)
                else:
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Ensure we return at most 48 elements
    if len(best_set) > 48:
        best_set = best_set[:48]
    
    return best_set
