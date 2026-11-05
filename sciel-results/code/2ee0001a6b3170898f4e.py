"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 55.0  # Safety margin before 90s limit
    best_sets = ([], [])
    best_size = 0
    
    def _can_add_to_set(x, S, set_sums, set_diffs, other_diffs):
        """Check if x can be added to set S without violating constraints."""
        # Sidon: all pair sums a+x and x+x must be unique in set_sums
        for a in S:
            if (a + x) in set_sums:
                return False
        if (x + x) in set_sums:
            return False
        # Difference disjointness: new diffs must not intersect other set's diffs
        for a in S:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
        return True
    
    def _add_to_set(x, S, set_sums, set_diffs):
        """Add x to set S, updating sums and diffs."""
        S.append(x)
        for a in S[:-1]:
            set_sums.add(a + x)
            set_diffs.add(x - a)
            set_diffs.add(a - x)
        set_sums.add(x + x)
    
    def _greedy_one_ordering(order, deadline):
        """
        Single forward pass over the given ordering of candidates.
        For each candidate x, try to add it to set 0 first, then set 1.
        """
        sets = [[], []]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            for side in (0, 1):
                S = sets[side]
                other_diffs = set_diffs[1 - side]
                
                if _can_add_to_set(x, S, set_sums[side], set_diffs[side], other_diffs):
                    _add_to_set(x, S, set_sums[side], set_diffs[side])
                    break
        
        return sets
    
    def _local_search(sets, deadline):
        """
        Local search: try to add unplaced elements, and try swaps to improve.
        Returns improved sets.
        """
        sets = [list(sets[0]), list(sets[1])]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        # Rebuild sums and diffs
        for side in (0, 1):
            S = sets[side]
            for i, a in enumerate(S):
                for b in S[i:]:
                    set_sums[side].add(a + b)
            for a in S:
                for b in S:
                    if a != b:
                        set_diffs[side].add(a - b)
        
        placed = set(sets[0]) | set(sets[1])
        unplaced = [x for x in range(1, 257) if x not in placed]
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # Try adding unplaced elements
            for x in unplaced[:]:
                if time.time() > deadline - 1.0:
                    break
                for side in (0, 1):
                    S = sets[side]
                    other_diffs = set_diffs[1 - side]
                    if _can_add_to_set(x, S, set_sums[side], set_diffs[side], other_diffs):
                        _add_to_set(x, S, set_sums[side], set_diffs[side])
                        unplaced.remove(x)
                        improved = True
                        break
            
            # Try swaps: remove an element from one set, try to add a better one
            if time.time() > deadline - 1.0:
                break
            for side in (0, 1):
                S = sets[side]
                if len(S) == 0:
                    continue
                for i in range(len(S)):
                    if time.time() > deadline - 1.0:
                        break
                    a = S[i]
                    # Temporarily remove a
                    S.pop(i)
                    # Rebuild sums and diffs for this set
                    old_sums = set_sums[side].copy()
                    old_diffs = set_diffs[side].copy()
                    set_sums[side] = set()
                    set_diffs[side] = set()
                    for j, u in enumerate(S):
                        for v in S[j:]:
                            set_sums[side].add(u + v)
                    for u in S:
                        for v in S:
                            if u != v:
                                set_diffs[side].add(u - v)
                    
                    # Try to add a different element
                    found_better = False
                    for x in unplaced[:]:
                        if time.time() > deadline - 1.0:
                            break
                        other_diffs = set_diffs[1 - side]
                        if _can_add_to_set(x, S, set_sums[side], set_diffs[side], other_diffs):
                            _add_to_set(x, S, set_sums[side], set_diffs[side])
                            unplaced.remove(x)
                            found_better = True
                            improved = True
                            break
                    
                    if not found_better:
                        # Put a back
                        S.append(a)
                        set_sums[side] = old_sums
                        set_diffs[side] = old_diffs
        
        return sets
    
    base_order = list(range(1, 257))
    
    # First try natural (ascending) order - deterministic baseline
    sets = _greedy_one_ordering(base_order, deadline)
    sets = _local_search(sets, deadline)
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = (list(sets[0]), list(sets[1]))
    
    # Try descending order
    sets = _greedy_one_ordering(base_order[::-1], deadline)
    sets = _local_search(sets, deadline)
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = (list(sets[0]), list(sets[1]))
    
    # Try multiple random permutations
    random.seed(42)
    while time.time() < deadline - 2.0:
        random_order = base_order[:]
        random.shuffle(random_order)
        
        sets = _greedy_one_ordering(random_order, deadline)
        sets = _local_search(sets, deadline)
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = (list(sets[0]), list(sets[1]))
        
        # Early stop if we've hit a good size
        if best_size >= 40:
            break
    
    # Ensure nonempty
    if not best_sets[0] and not best_sets[1]:
        best_sets = ([1], [2])
    elif not best_sets[0]:
        best_sets[0] = [1]
    elif not best_sets[1]:
        best_sets[1] = [1]
    
    return best_sets
# EVOLVE-BLOCK-END
