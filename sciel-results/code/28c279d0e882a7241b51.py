"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_valid_set(selected):
    """
    Check if 'selected' is a valid set: no element is the arithmetic mean
    of any subset of at least 2 other elements.
    """
    n = len(selected)
    if n < 3:
        return True
    
    # For each element a, check if it's the mean of any subset of others
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        
        # Check all subset sizes from 2 to m
        # For efficiency, we can use the fact that sum must be k*a
        # We'll check smaller k first as they're more likely to be violated
        for k in range(2, m + 1):
            target_sum = k * a
            # Check if any subset of size k from others sums to target_sum
            # For small k, use combinations; for large k, this could be expensive
            # But since we're building incrementally and n<=48, m<=47
            # For k=2: O(m^2), k=3: O(m^3), etc.
            
            # Optimization: for k=2, just check pairs
            if k == 2:
                # Need two elements x,y from others such that x+y = 2*a
                # i.e., x + y = target_sum
                # Use a set for O(m) check
                seen = set()
                for x in others:
                    y = target_sum - x
                    if y in seen:
                        return False
                    seen.add(x)
            elif k == 3:
                # Need three elements x,y,z from others such that x+y+z = 3*a
                # O(m^2) check
                others_set = set(others)
                for idx_x, x in enumerate(others):
                    for y in others[idx_x+1:]:
                        z = target_sum - x - y
                        if z in others_set and z != x and z != y:
                            # Need to ensure z is actually in others and distinct
                            # More careful check:
                            # We need x, y, z all in others, all distinct positions
                            # Since others may have duplicates? No, selected has distinct elements
                            # So x, y, z must be distinct values
                            if z != x and z != y:
                                # Verify z is in others
                                if z in others_set:
                                    return False
            else:
                # For k >= 4, use combinations but this gets expensive
                # For now, use itertools.combinations
                found = False
                for comb in combinations(others, k):
                    if sum(comb) == target_sum:
                        found = True
                        break
                if found:
                    return False
    return True


def _is_valid_addition(selected, candidate):
    """
    Check if adding 'candidate' to 'selected' keeps the set valid.
    Only need to check:
    1. Is 'candidate' the mean of any subset of >=2 elements from 'selected'?
    2. Is any element in 'selected' the mean of a subset that includes 'candidate' and >=1 other from 'selected'?
       Actually, we need to check if any element a in selected∪{candidate} is the mean of a subset of >=2 OTHER elements.
       For a in selected: the other elements are (selected - {a}) ∪ {candidate}.
       We already know selected was valid, so a is not the mean of any subset of (selected - {a}).
       Now we need to check if a is the mean of a subset of (selected - {a}) ∪ {candidate} that includes candidate.
       That is, a is the mean of {candidate} ∪ S where S is a subset of (selected - {a}) with |S| >= 1.
       So sum(S) + candidate = (|S|+1)*a, i.e., sum(S) = (|S|+1)*a - candidate.
       
    3. Similarly, for candidate: is it the mean of any subset of >=2 from selected?
    """
    # Check 1: Is candidate the mean of any subset of >=2 from selected?
    n = len(selected)
    for k in range(2, n + 1):
        target_sum = k * candidate
        if k == 2:
            seen = set()
            for x in selected:
                y = target_sum - x
                if y in seen:
                    return False
                seen.add(x)
        elif k == 3:
            for i in range(n):
                for j in range(i+1, n):
                    z = target_sum - selected[i] - selected[j]
                    if z != selected[i] and z != selected[j]:
                        # Check if z is in selected
                        # For efficiency, we can use a set
                        pass
            # Simpler: use set
            sel_set = set(selected)
            for i in range(n):
                for j in range(i+1, n):
                    z = target_sum - selected[i] - selected[j]
                    if z in sel_set and z != selected[i] and z != selected[j]:
                        return False
        else:
            for comb in combinations(selected, k):
                if sum(comb) == target_sum:
                    return False
    
    # Check 2: For each a in selected, is a the mean of a subset including candidate?
    for a in selected:
        others_without_a = [x for x in selected if x != a]
        m = len(others_without_a)
        # We need subset S of others_without_a with |S| >= 1 such that sum(S) + candidate = (|S|+1)*a
        # i.e., sum(S) = (|S|+1)*a - candidate
        # Let k = |S|, k >= 1
        for k in range(1, m + 1):
            target_sum = (k + 1) * a - candidate
            if k == 1:
                # Just check if any x in others_without_a equals target_sum
                if target_sum in others_without_a:
                    return False
            elif k == 2:
                seen = set()
                for x in others_without_a:
                    y = target_sum - x
                    if y in seen:
                        return False
                    seen.add(x)
            else:
                for comb in combinations(others_without_a, k):
                    if sum(comb) == target_sum:
                        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s out of 90s budget
    
    # Strategy: Build a set greedily, trying to maximize cardinality.
    # Use multiplicative-structure candidates (2^i * 3^j) as they reduce additive-mean violations.
    
    # Generate candidate numbers with multiplicative structure
    candidates = set()
    for i in range(0, 10):  # 2^0 to 2^9
        for j in range(0, 10):  # 3^0 to 3^9
            val = (2 ** i) * (3 ** j)
            if 1 <= val <= 512:
                candidates.add(val)
    
    # Also include some other numbers to potentially get more
    # But start with the multiplicative ones
    candidate_list = sorted(candidates)
    
    # Greedy construction: try to add candidates in some order
    # To maximize cardinality, we should be strategic.
    # Try multiple random orders and keep the best.
    
    best_set = []
    best_size = 0
    
    # Try several random permutations of candidates
    random.seed(42)
    num_trials = 50
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        # Create a shuffled candidate list
        shuffled = candidate_list[:]
        random.shuffle(shuffled)
        
        current = []
        for cand in shuffled:
            if time.time() > deadline - 2:
                break
            if _is_valid_addition(current, cand):
                current.append(cand)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
            # If we've reached 48, we're done
            if best_size >= 48:
                break
    
    # If we didn't reach 48, try to add more numbers from the full range
    if best_size < 48:
        # Try adding numbers from 1 to 512 that aren't in best_set
        remaining = [x for x in range(1, 513) if x not in best_set]
        random.shuffle(remaining)
        
        for cand in remaining:
            if time.time() > deadline - 2:
                break
            if _is_valid_addition(best_set, cand):
                best_set.append(cand)
                if len(best_set) >= 48:
                    break
    
    return best_set[:48]
# EVOLVE-BLOCK-END
