"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 52.0  # Safety margin before 90s limit
    best_sets = ([], [])
    best_size = 0

    def _can_add(S, sums, diffs, other_diffs, x):
        """Check if x can be added to set S."""
        # Sidon: all pair sums must be unique
        for a in S:
            s = a + x
            if s in sums:
                return False
        if (x + x) in sums:
            return False
        # Difference disjointness
        for a in S:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
        return True

    def _add(S, sums, diffs, x):
        """Add x to set S, updating sums and diffs."""
        S.append(x)
        for a in S[:-1]:
            sums.add(a + x)
            diffs.add(x - a)
            diffs.add(a - x)
        sums.add(x + x)

    def _greedy_one_ordering(order, deadline, side_bias=None):
        """
        Single forward pass over the given ordering.
        side_bias: if 'first', always try set 0 first; if 'second', try set 1 first;
                   if None, try set 0 first (default).
        """
        sets = [[], []]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Determine order of sides to try
            if side_bias == 'second':
                sides = (1, 0)
            else:
                sides = (0, 1)
            
            for side in sides:
                S = sets[side]
                other_diffs = set_diffs[1 - side]
                
                if _can_add(S, set_sums[side], set_diffs[side], other_diffs, x):
                    _add(S, set_sums[side], set_diffs[side], x)
                    break
        
        return sets

    def _rebuild_sums_diffs(S, sums, diffs):
        """Rebuild sums and diffs sets for a given set S."""
        sums.clear()
        diffs.clear()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        for i in range(n):
            for j in range(n):
                if i != j:
                    diffs.add(S[i] - S[j])
                    diffs.add(S[j] - S[i])

    def _local_search(sets, deadline, max_iter=500):
        """
        Local search: try to improve by removing elements and re-adding,
        or by swapping elements between sets.
        """
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        for side in (0, 1):
            _rebuild_sums_diffs(sets[side], set_sums[side], set_diffs[side])
        
        current_size = len(sets[0]) + len(sets[1])
        all_candidates = list(range(1, 257))
        in_set0 = set(sets[0])
        in_set1 = set(sets[1])
        
        stall_count = 0
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding unplaced elements - prioritize those with fewer conflicts
            unplaced = [x for x in all_candidates if x not in in_set0 and x not in in_set1]
            if unplaced:
                # Score unplaced elements by how "easy" they are to add
                def _conflict_score(x):
                    score = 0
                    for side in (0, 1):
                        S = sets[side]
                        other_diffs = set_diffs[1 - side]
                        for a in S:
                            if (a + x) in set_sums[side] or (x + x) in set_sums[side]:
                                score += 1
                            d1 = x - a
                            d2 = a - x
                            if d1 in other_diffs or d2 in other_diffs:
                                score += 1
                    return score
                
                # Sort by conflict score (fewer conflicts first), then randomize ties
                scored = [(x, _conflict_score(x)) for x in unplaced]
                scored.sort(key=lambda t: (t[1], random.random()))
                
                for x, _ in scored[:40]:
                    for side in (0, 1):
                        S = sets[side]
                        other_diffs = set_diffs[1 - side]
                        if _can_add(S, set_sums[side], set_diffs[side], other_diffs, x):
                            _add(S, set_sums[side], set_diffs[side], x)
                            if side == 0:
                                in_set0.add(x)
                            else:
                                in_set1.add(x)
                            current_size += 1
                            improved = True
                            break
                    if improved:
                        break
            
            # Try moving an element from one set to the other
            if not improved and current_size > 4:
                for side in (0, 1):
                    if len(sets[side]) > 1:
                        other_side = 1 - side
                        # Try each element in the set
                        for idx in range(len(sets[side])):
                            x = sets[side][idx]
                            S_other = sets[other_side]
                            if _can_add(S_other, set_sums[other_side], set_diffs[other_side], set_diffs[side], x):
                                # Remove x from current set
                                sets[side].pop(idx)
                                if side == 0:
                                    in_set0.discard(x)
                                else:
                                    in_set1.discard(x)
                                # Rebuild sums and diffs for current set
                                _rebuild_sums_diffs(sets[side], set_sums[side], set_diffs[side])
                                # Add x to other set
                                _add(S_other, set_sums[other_side], set_diffs[other_side], x)
                                if other_side == 0:
                                    in_set0.add(x)
                                else:
                                    in_set1.add(x)
                                improved = True
                                break
                    if improved:
                        break
            
            # Try 2-opt: remove one element, try to add two new ones
            if not improved and current_size > 6:
                for side in (0, 1):
                    if len(sets[side]) > 2:
                        idx = random.randint(0, len(sets[side]) - 1)
                        x = sets[side][idx]
                        # Temporarily remove x
                        sets[side].pop(idx)
                        if side == 0:
                            in_set0.discard(x)
                        else:
                            in_set1.discard(x)
                        _rebuild_sums_diffs(sets[side], set_sums[side], set_diffs[side])
                        
                        added_count = 0
                        unplaced_now = [y for y in all_candidates if y not in in_set0 and y not in in_set1 and y != x]
                        random.shuffle(unplaced_now)
                        for y in unplaced_now[:25]:
                            for s2 in (0, 1):
                                if _can_add(sets[s2], set_sums[s2], set_diffs[s2], set_diffs[1-s2], y):
                                    _add(sets[s2], set_sums[s2], set_diffs[s2], y)
                                    if s2 == 0:
                                        in_set0.add(y)
                                    else:
                                        in_set1.add(y)
                                    current_size += 1
                                    added_count += 1
                                    break
                            if added_count >= 2:
                                break
                        
                        if added_count > 0:
                            improved = True
                        else:
                            # Put x back
                            _add(sets[side], set_sums[side], set_diffs[side], x)
                            if side == 0:
                                in_set0.add(x)
                            else:
                                in_set1.add(x)
                        break
            
            # Try 3-opt: remove two elements from same set, try to add three new ones
            if not improved and current_size > 8:
                for side in (0, 1):
                    if len(sets[side]) > 3:
                        idx1 = random.randint(0, len(sets[side]) - 1)
                        idx2 = random.randint(0, len(sets[side]) - 1)
                        if idx1 == idx2:
                            idx2 = (idx2 + 1) % len(sets[side])
                        
                        x1 = sets[side][idx1]
                        x2 = sets[side][idx2]
                        
                        # Remove both (handle order carefully)
                        if idx1 > idx2:
                            sets[side].pop(idx1)
                            sets[side].pop(idx2)
                        else:
                            sets[side].pop(idx2)
                            sets[side].pop(idx1)
                        
                        if side == 0:
                            in_set0.discard(x1)
                            in_set0.discard(x2)
                        else:
                            in_set1.discard(x1)
                            in_set1.discard(x2)
                        
                        _rebuild_sums_diffs(sets[side], set_sums[side], set_diffs[side])
                        
                        added_count = 0
                        unplaced_now = [y for y in all_candidates if y not in in_set0 and y not in in_set1 and y != x1 and y != x2]
                        random.shuffle(unplaced_now)
                        for y in unplaced_now[:30]:
                            for s2 in (0, 1):
                                if _can_add(sets[s2], set_sums[s2], set_diffs[s2], set_diffs[1-s2], y):
                                    _add(sets[s2], set_sums[s2], set_diffs[s2], y)
                                    if s2 == 0:
                                        in_set0.add(y)
                                    else:
                                        in_set1.add(y)
                                    current_size += 1
                                    added_count += 1
                                    break
                            if added_count >= 3:
                                break
                        
                        if added_count > 0:
                            improved = True
                        else:
                            # Put both back
                            _add(sets[side], set_sums[side], set_diffs[side], x1)
                            _add(sets[side], set_sums[side], set_diffs[side], x2)
                            if side == 0:
                                in_set0.add(x1)
                                in_set0.add(x2)
                            else:
                                in_set1.add(x1)
                                in_set1.add(x2)
                        break
            
            if not improved:
                stall_count += 1
                if stall_count > 30:
                    break
            else:
                stall_count = 0
        
        return sets

    base_order = list(range(1, 257))
    
    # First try natural (ascending) order - deterministic baseline
    sets = _greedy_one_ordering(base_order, deadline)
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = (list(sets[0]), list(sets[1]))
    
    # Try descending order
    sets = _greedy_one_ordering(base_order[::-1], deadline)
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = (list(sets[0]), list(sets[1]))
    
    # Try with side_bias='second'
    sets = _greedy_one_ordering(base_order, deadline, side_bias='second')
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = (list(sets[0]), list(sets[1]))
    
    # Try multiple random permutations with varying biases
    random.seed(42)
    while time.time() < deadline - 8.0:
        random_order = base_order[:]
        random.shuffle(random_order)
        
        # Alternate between different side biases
        if random.random() < 0.5:
            sets = _greedy_one_ordering(random_order, deadline)
        else:
            sets = _greedy_one_ordering(random_order, deadline, side_bias='second')
        
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = (list(sets[0]), list(sets[1]))
    
    # Apply local search to the best solution
    if best_size > 0 and time.time() < deadline - 3.0:
        polished = _local_search([list(best_sets[0]), list(best_sets[1])], deadline, max_iter=1000)
        size = len(polished[0]) + len(polished[1])
        if size > best_size:
            best_size = size
            best_sets = (list(polished[0]), list(polished[1]))
        
        # If we still have time, do another round of local search
        if time.time() < deadline - 2.0:
            polished2 = _local_search([list(best_sets[0]), list(best_sets[1])], deadline, max_iter=600)
            size2 = len(polished2[0]) + len(polished2[1])
            if size2 > best_size:
                best_size = size2
                best_sets = (list(polished2[0]), list(polished2[1]))
    
    # Ensure nonempty
    if not best_sets[0] and not best_sets[1]:
        best_sets = ([1], [2])
    elif not best_sets[0]:
        best_sets[0] = [1]
    elif not best_sets[1]:
        best_sets[1] = [1]
    
    return best_sets
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
