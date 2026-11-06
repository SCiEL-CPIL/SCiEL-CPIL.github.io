"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict, deque

def _is_valid(S):
    """Check if set S satisfies the condition: for all a<b<=c in S, a does not divide b+c."""
    S = sorted(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            for k in range(j, n):
                c = S[k]
                if (b + c) % a == 0:
                    return False
    return True

def _find_min_cut_removals(candidates, n=256):
    """
    Find the minimum set of elements to remove from candidates to make the set valid.
    Uses a bipartite graph formulation where we try to maximize the number of kept elements.
    
    The key insight: if we include element a, we must exclude all pairs (b,c) with b,c > a
    such that a divides b+c. We can model this as: for each element a, if we keep it,
    we must remove at least one element from each conflicting pair.
    
    This is a hitting set problem, which is NP-hard in general, but we can use a greedy
    approximation or formulate it as a flow problem for small instances.
    
    For now, let's use a greedy approach: repeatedly find the element whose removal
    resolves the most conflicts, and remove it.
    """
    candidates = set(candidates)
    if not candidates:
        return set()
    
    # Precompute conflicts: for each a in candidates, find all pairs (b,c) with b,c > a
    # such that a divides b+c
    conflicts = defaultdict(set)  # conflicts[a] = set of pairs (b,c)
    
    sorted_cands = sorted(candidates)
    for i, a in enumerate(sorted_cands):
        for j in range(i+1, len(sorted_cands)):
            b = sorted_cands[j]
            for k in range(j, len(sorted_cands)):
                c = sorted_cands[k]
                if (b + c) % a == 0:
                    conflicts[a].add((b, c))
    
    # If no conflicts, we can keep all elements
    if not any(conflicts[a] for a in candidates):
        return set()
    
    # Greedy: repeatedly remove the element that resolves the most conflicts
    # An element x can be either a "divisor" (if x is in conflicts keys) or a "summand"
    # If we remove x as a divisor, we resolve all conflicts[x]
    # If we remove x as a summand, we resolve all conflicts[a] that contain pairs with x
    
    # Let's track which conflicts are still active
    active_conflicts = {a: set(pairs) for a, pairs in conflicts.items() if pairs}
    
    removed = set()
    
    while active_conflicts:
        # Find the element whose removal resolves the most active conflicts
        best_element = None
        best_resolution = 0
        
        # Option 1: Remove as a divisor
        for a, pairs in active_conflicts.items():
            if len(pairs) > best_resolution:
                best_resolution = len(pairs)
                best_element = a
        
        # Option 2: Remove as a summand
        # For each element x, count how many active conflicts it participates in
        conflict_count = defaultdict(int)
        for a, pairs in active_conflicts.items():
            for b, c in pairs:
                conflict_count[b] += 1
                conflict_count[c] += 1
        
        for x, count in conflict_count.items():
            if count > best_resolution:
                best_resolution = count
                best_element = x
        
        if best_element is None or best_resolution == 0:
            break
        
        # Remove best_element
        removed.add(best_element)
        
        # Update active conflicts
        new_active = {}
        for a, pairs in active_conflicts.items():
            if a == best_element:
                continue  # This divisor is removed
            new_pairs = set()
            for b, c in pairs:
                if b != best_element and c != best_element:
                    new_pairs.add((b, c))
            if new_pairs:
                new_active[a] = new_pairs
        active_conflicts = new_active
    
    return removed

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # The upper 1/3 construction is known to be valid and has size 86.
    S_upper_1_3 = list(range(171, 257))
    
    # Try to extend by adding elements from the lower range
    # Start with the upper 1/3 and try to add elements from 1 to 170
    # For each candidate, check if adding it creates conflicts, and if so,
    # use min-cut to find the minimum removals needed
    
    best_set = set(S_upper_1_3)
    best_size = len(best_set)
    
    # Try adding elements from 1 to 170 in increasing order of "cost"
    # An element is "cheap" if it creates few conflicts
    candidates = list(range(1, 171))
    
    # For each candidate, compute how many conflicts it would create
    def count_conflicts(x, current_set):
        """Count how many pairs (b,c) in current_set with b,c > x such that x divides b+c."""
        current_set = sorted(current_set)
        count = 0
        for i, b in enumerate(current_set):
            if b <= x:
                continue
            for j in range(i, len(current_set)):
                c = current_set[j]
                if (b + c) % x == 0:
                    count += 1
        return count
    
    # Sort candidates by conflict count (ascending) - try cheap ones first
    conflict_counts = []
    for x in candidates:
        cc = count_conflicts(x, best_set)
        conflict_counts.append((cc, x))
    conflict_counts.sort()
    
    # Try to add each candidate in order of increasing conflict count
    for cc, x in conflict_counts:
        if time.time() > deadline:
            break
        
        if x in best_set:
            continue
        
        # Try adding x to the set
        new_set = best_set | {x}
        
        # Check if new_set is valid
        if _is_valid(new_set):
            best_set = new_set
            best_size += 1
        else:
            # If not valid, try to find minimum removals
            # But this is expensive, so only do it if we have time
            if time.time() < deadline - 10:
                # Find conflicts caused by adding x
                # We need to remove some elements to resolve conflicts involving x
                # Let's use a simpler approach: try removing elements that conflict with x
                conflicts_with_x = set()
                for b in best_set:
                    if b > x:
                        for c in best_set:
                            if c >= b and (b + c) % x == 0:
                                conflicts_with_x.add(b)
                                conflicts_with_x.add(c)
                
                # Try removing the smallest number of elements to resolve all conflicts
                # Greedy: remove the element that resolves the most conflicts
                removed = set()
                remaining_conflicts = set()
                for b in best_set:
                    if b > x:
                        for c in best_set:
                            if c >= b and (b + c) % x == 0:
                                remaining_conflicts.add((b, c))
                
                while remaining_conflicts and len(removed) < 3:  # Limit removals to 3
                    # Find the element that appears in the most conflicts
                    elem_count = defaultdict(int)
                    for b, c in remaining_conflicts:
                        elem_count[b] += 1
                        elem_count[c] += 1
                    
                    if not elem_count:
                        break
                    
                    # Remove the element with highest count (excluding x)
                    best_elem = None
                    best_count = 0
                    for elem, count in elem_count.items():
                        if elem != x and count > best_count:
                            best_count = count
                            best_elem = elem
                    
                    if best_elem is None:
                        break
                    
                    removed.add(best_elem)
                    remaining_conflicts = set()
                    for b in best_set:
                        if b > x and b not in removed:
                            for c in best_set:
                                if c >= b and c not in removed and (b + c) % x == 0:
                                    remaining_conflicts.add((b, c))
                
                # Check if removing these elements makes the set valid
                candidate_set = (best_set | {x}) - removed
                if len(candidate_set) > best_size and _is_valid(candidate_set):
                    best_set = candidate_set
                    best_size = len(best_set)
    
    # Final local improvement: try swapping elements
    if time.time() < deadline - 5:
        # Try to improve by swapping elements
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in list(best_set):
                for y in range(1, 257):
                    if y in best_set:
                        continue
                    new_set = (best_set - {x}) | {y}
                    if _is_valid(new_set):
                        # Check if this is an improvement (it's not, since size is same)
                        # But it might enable further improvements, so skip for now
                        pass
    
    return list(best_set)
# EVOLVE-BLOCK-END
