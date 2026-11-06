import time
import random

def construct():
    n = 512
    deadline = time.time() + 50  # Safety margin for 90s limit
    
    # Precompute all 4-term APs as hyperedges
    # Each AP is a tuple of 4 indices (0-based)
    # Also precompute for each element, which APs contain it
    aps = []
    element_to_aps = [[] for _ in range(n)]
    
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e < n:
                ap_idx = len(aps)
                aps.append((a, b, c, e))
                element_to_aps[a].append(ap_idx)
                element_to_aps[b].append(ap_idx)
                element_to_aps[c].append(ap_idx)
                element_to_aps[e].append(ap_idx)
    
    num_aps = len(aps)
    
    # For each element, precompute a bitmask of which APs it belongs to
    # Since num_aps can be large, we'll use a different approach
    # Instead, for each element, store the list of AP indices
    
    # For efficient checking: when we add element x, we need to check
    # if any AP containing x already has 3 other elements selected
    
    # Alternative: for each element x, precompute which other elements
    # would complete an AP with x and two already-selected elements
    
    # Let's use a different approach: maintain for each AP a count of selected elements
    # When selecting element x, increment count for all APs containing x
    # If any AP reaches count 4, that's invalid
    
    # For branch and bound, we need:
    # 1. A way to quickly check if adding x is valid
    # 2. A way to compute an upper bound on additional elements we can add
    
    # Precompute for each element x, the set of elements that conflict with x
    # (i.e., there exists a 3-element subset T such that T ∪ {x} forms a 4-AP)
    # Actually, we need: x conflicts with y if there exist z, w such that {x,y,z,w} is a 4-AP
    
    # Let's precompute conflict sets
    # conflicts[x] = set of y such that x and y are in some 4-AP
    # But this is too coarse. We need to know if adding x would complete any AP.
    
    # Better approach: for each element x, and for each pair (y,z) already selected,
    # check if {x,y,z} + one more = 4-AP. This is expensive.
    
    # Let's use the AP count approach with a bitset representation
    
    # For each AP, store a bitmask of its 4 elements
    ap_masks = []
    for a, b, c, e in aps:
        mask = (1 << a) | (1 << b) | (1 << c) | (1 << e)
        ap_masks.append(mask)
    
    # For each element, store which APs contain it (as a list of AP indices)
    # Already have element_to_aps
    
    # For upper bound: use the fact that if we have k elements left to consider,
    # and each can be in at most m APs, we can derive bounds
    
    # Simpler upper bound: the number of unselected elements that don't immediately
    # conflict with the current selection
    
    def can_add(x, selected_mask, ap_counts):
        """Check if element x can be added without completing any 4-AP."""
        for ap_idx in element_to_aps[x]:
            if ap_counts[ap_idx] >= 3:
                # This AP already has 3 elements selected, adding x would make 4
                return False
        return True
    
    def compute_upper_bound(selected_mask, ap_counts, remaining):
        """
        Compute an upper bound on how many more elements we can add.
        remaining: list of candidate elements not yet decided
        """
        # Simple bound: count elements in remaining that can be added
        count = 0
        for x in remaining:
            if can_add(x, selected_mask, ap_counts):
                count += 1
        return count
    
    # Greedy lower bound: use the base-3 construction
    def get_base3_set():
        """Generate the base-3 3-AP-free set (digits 0, 2) in [1, n]."""
        S = 0
        for x in range(1, n + 1):
            valid = True
            num = x
            while num > 0:
                if num % 3 == 1:
                    valid = False
                    break
                num //= 3
            if valid:
                S |= (1 << (x - 1))
        return S
    
    # Convert bitset to list
    def bitset_to_list(bitset):
        result = []
        for i in range(n):
            if bitset & (1 << i):
                result.append(i)
        return result
    
    # Get initial solution from base-3 set
    base3_mask = get_base3_set()
    base3_list = bitset_to_list(base3_mask)
    
    # Compute ap_counts for the base3 set
    ap_counts = [0] * num_aps
    for x in base3_list:
        for ap_idx in element_to_aps[x]:
            ap_counts[ap_idx] += 1
    
    best_size = len(base3_list)
    best_solution = base3_list[:]
    
    # Branch and bound
    # State: (selected_mask, ap_counts, candidates, depth)
    # We'll use a recursive approach with time checking
    
    import sys
    sys.setrecursionlimit(10000)
    
    def branch_and_bound(selected_mask, ap_counts, candidates, current_size):
        nonlocal best_size, best_solution, deadline
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_solution = bitset_to_list(selected_mask)
            return
        
        # Upper bound: current_size + len(candidates)
        if current_size + len(candidates) <= best_size:
            return
        
        # Try to find a better upper bound
        # Count how many candidates can actually be added
        addable = []
        for x in candidates:
            if can_add(x, selected_mask, ap_counts):
                addable.append(x)
        
        if not addable:
            if current_size > best_size:
                best_size = current_size
                best_solution = bitset_to_list(selected_mask)
            return
        
        if current_size + len(addable) <= best_size:
            return
        
        # Choose the next element to branch on
        # Heuristic: choose the element that appears in the most APs with other addable elements
        # Or simply choose the first addable element
        x = addable[0]
        
        # Branch 1: include x
        new_ap_counts = ap_counts[:]
        for ap_idx in element_to_aps[x]:
            new_ap_counts[ap_idx] += 1
        
        # Check if including x is valid (no AP reaches 4)
        valid = True
        for ap_idx in element_to_aps[x]:
            if new_ap_counts[ap_idx] >= 4:
                valid = False
                break
        
        if valid:
            new_selected = selected_mask | (1 << x)
            # Filter candidates: remove x and any that conflict
            new_candidates = []
            for y in candidates:
                if y == x:
                    continue
                if can_add(y, new_selected, new_ap_counts):
                    new_candidates.append(y)
            branch_and_bound(new_selected, new_ap_counts, new_candidates, current_size + 1)
        
        if time.time() > deadline:
            return
        
        # Branch 2: exclude x
        new_candidates = [y for y in candidates if y != x]
        branch_and_bound(selected_mask, ap_counts, new_candidates, current_size)
    
    # Start branch and bound from the base3 solution
    # Candidates are all elements not in base3
    candidates = [x for x in range(n) if not (base3_mask & (1 << x))]
    
    # Shuffle candidates to get diversity
    random.shuffle(candidates)
    
    branch_and_bound(base3_mask, ap_counts, candidates, len(base3_list))
    
    # If we have time, try a different starting point
    if time.time() < deadline - 5:
        # Try greedy construction
        greedy_mask = 0
        greedy_ap_counts = [0] * num_aps
        greedy_list = []
        
        for x in range(n):
            if can_add(x, greedy_mask, greedy_ap_counts):
                greedy_mask |= (1 << x)
                for ap_idx in element_to_aps[x]:
                    greedy_ap_counts[ap_idx] += 1
                greedy_list.append(x)
        
        if len(greedy_list) > best_size:
            best_size = len(greedy_list)
            best_solution = greedy_list[:]
        
        # Branch and bound from greedy
        if time.time() < deadline - 5:
            candidates = [x for x in range(n) if not (greedy_mask & (1 << x))]
            random.shuffle(candidates)
            branch_and_bound(greedy_mask, greedy_ap_counts, candidates, len(greedy_list))
    
    # Convert to 1-based indexing and return
    return sorted([x + 1 for x in best_solution])
