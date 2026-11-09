"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def _is_sunflower_triple(a, b, c):
    """Check if three sets form a sunflower (all pairwise intersections equal)."""
    return (a & b) == (a & c) == (b & c)

def _count_conflicts_with_family(new_set, family):
    """Count how many pairs in family would form a sunflower with new_set."""
    count = 0
    n = len(family)
    for i in range(n):
        for j in range(i + 1, n):
            if _is_sunflower_triple(family[i], family[j], new_set):
                count += 1
    return count

def _greedy_construct(order, deadline):
    """Greedy construction given an ordering of all 4-subsets."""
    family = []
    for s in order:
        if time.time() > deadline:
            break
        # Check if adding s creates a sunflower with any existing pair
        conflict = False
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if _is_sunflower_triple(family[i], family[j], s):
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            family.append(s)
    return family

def _local_improvement(family, all_sets, deadline, max_rounds=3):
    """Try to improve family by swapping out elements that block many candidates."""
    family_set = set(frozenset(s) for s in family)
    family_list = list(family)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        # For each element in family, compute how many candidates it blocks
        # A candidate is blocked if it forms a sunflower with two family members where one is this element
        block_counts = defaultdict(int)
        for s in family_list:
            fs = frozenset(s)
            for i in range(len(family_list)):
                for j in range(i + 1, len(family_list)):
                    a = family_list[i]
                    b = family_list[j]
                    if _is_sunflower_triple(a, b, s):
                        # s forms a sunflower with pair (a,b)
                        # This means removing either a or b might allow s
                        block_counts[frozenset(a)] += 1
                        block_counts[frozenset(b)] += 1
        
        # Try to find an element to remove that allows adding many new elements
        best_swap = None
        best_gain = 0
        
        # Sort family elements by block count (most blocking first)
        sorted_family = sorted(family_list, key=lambda x: block_counts.get(frozenset(x), 0), reverse=True)
        
        for elem_to_remove in sorted_family[:min(5, len(family_list))]:
            if time.time() > deadline:
                break
            
            # Temporarily remove this element
            temp_family = [s for s in family_list if s != elem_to_remove]
            temp_family_set = set(frozenset(s) for s in temp_family)
            
            # Find candidates that can now be added
            addable = []
            for s in all_sets:
                fs = frozenset(s)
                if fs in temp_family_set:
                    continue
                # Check if s can be added to temp_family
                can_add = True
                for i in range(len(temp_family)):
                    for j in range(i + 1, len(temp_family)):
                        if _is_sunflower_triple(temp_family[i], temp_family[j], s):
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    addable.append(s)
            
            # If we can add more than 1 element, this is a good swap
            gain = len(addable) - 1  # net gain
            if gain > best_gain and len(addable) > 0:
                best_gain = gain
                best_swap = (elem_to_remove, addable[:best_gain])
        
        if best_swap and best_gain > 0:
            elem_to_remove, addable = best_swap
            family_list = [s for s in family_list if s != elem_to_remove]
            for s in addable:
                family_list.append(s)
    
    return family_list

def _score_set_for_order(s, all_sets):
    """Score a set based on how "central" it is - sets that are in many potential sunflowers should be deprioritized."""
    # Count how many other sets have intersection size 2 with s
    # (this is the most common sunflower configuration)
    count = 0
    for t in all_sets:
        if s == t:
            continue
        inter = len(s & t)
        if inter == 2:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Precompute all 4-element subsets
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    best_family = []
    best_size = 0
    
    # Strategy 1: Greedy with heuristic ordering
    # Order sets by how "conflict-prone" they are (ascending - try less conflict-prone first)
    if time.time() < deadline:
        # Compute conflict scores
        conflict_scores = {}
        for s in all_sets:
            conflict_scores[frozenset(s)] = _score_set_for_order(s, all_sets)
        
        # Sort by conflict score (ascending) - try less conflict-prone sets first
        ordered = sorted(all_sets, key=lambda s: conflict_scores[frozenset(s)])
        family = _greedy_construct(ordered, deadline)
        if len(family) > best_size:
            best_family = family
            best_size = len(family)
    
    # Strategy 2: Multiple random orderings with local improvement
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline - 10:  # Leave 10s for final improvement
        attempt += 1
        shuffled = all_sets[:]
        rng.shuffle(shuffled)
        family = _greedy_construct(shuffled, deadline - 10)
        if len(family) > best_size:
            best_family = family
            best_size = len(family)
            if best_size >= 250:
                break
    
    # Final: Apply local improvement to the best family found
    if best_family and time.time() < deadline - 2:
        improved = _local_improvement(best_family, all_sets, deadline - 2, max_rounds=2)
        if len(improved) > len(best_family):
            best_family = improved
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
