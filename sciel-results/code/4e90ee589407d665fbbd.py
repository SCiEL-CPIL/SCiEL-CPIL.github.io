import time
import sys

def _validate_full(elements):
    """
    Full strict validation.
    1. All elements distinct, in 1..256, type int.
    2. All non-empty subset sums are unique.
    3. No two distinct non-empty subset sums divide one another.
    """
    if not elements:
        return True
    
    # Type and range check
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    if len(set(elements)) != len(elements):
        return False
        
    # Generate all subset sums
    # Use a set to detect duplicates early and store unique sums
    sums = set([0])
    for x in elements:
        new_sums = set(s + x for s in sums)
        # If any new sum already exists in sums (and is not 0+x where x=0, but x>=1), 
        # wait. If s+x is in sums, it means there are two subsets with same sum.
        # However, we must be careful: sums currently contains 0.
        # If x is in sums, then 0+x = x is a duplicate of subset {x} and subset... wait.
        # The set `sums` contains all sums of subsets of the *previous* elements.
        # We are adding x. The new sums are {s+x | s in sums}.
        # If any s+x is in `sums`, then we have a collision?
        # Not necessarily a collision in the *final* set of sums, but it implies 
        # that the subset generating s+x (which includes x) has the same sum as 
        # the subset generating s+x (which does NOT include x, since s+x is in old sums).
        # So yes, if (sums & new_sums) is non-empty, we have a duplicate sum.
        if sums & new_sums:
            return False
        sums |= new_sums
        
    # Now sums contains all unique subset sums (including 0).
    # We only care about non-empty sums, so remove 0.
    non_empty_sums = [s for s in sums if s > 0]
    
    # Check divisibility
    # Sort to make it easier to check "smaller divides larger"
    non_empty_sums.sort()
    n = len(non_empty_sums)
    
    # Optimization: if a > max_sum / 2, it cannot divide any other sum in the set (except itself, which is excluded).
    # Actually, if a > max_sum / 2, then 2*a > max_sum, so no multiple of a exists in the set.
    max_sum = non_empty_sums[-1]
    
    # We can iterate i from 0 to n-1
    # For each a = non_empty_sums[i], check if any b = non_empty_sums[j] (j>i) is divisible by a.
    # We can break early if a > max_sum // 2.
    
    for i in range(n):
        a = non_empty_sums[i]
        if a > max_sum // 2:
            break
        # Check all subsequent sums
        for j in range(i + 1, n):
            b = non_empty_sums[j]
            if b % a == 0:
                return False
    return True

def _try_add(elements, sums_set, candidate):
    """
    Attempts to add `candidate` to `elements`.
    `sums_set` is the set of all subset sums of `elements` (including 0).
    Returns (new_elements, new_sums_set) if valid, else (None, None).
    """
    # 1. Check for duplicate sums
    # New sums are {s + candidate for s in sums_set}
    # If any of these are already in sums_set, it's a collision.
    # Note: 0 + candidate = candidate. If candidate is in sums_set, collision.
    # Also, if s1 + candidate == s2 + candidate, then s1 == s2, so no internal collisions in the new set.
    # The only collisions are with the existing sums_set.
    
    # Efficient check:
    # We need to check if (sums_set + candidate) intersects with sums_set.
    # i.e., does there exist s1, s2 in sums_set such that s1 + candidate = s2?
    # This is equivalent to: does sums_set intersect with (sums_set - candidate)?
    # Or simply: for s in sums_set, is s + candidate in sums_set?
    
    # Since candidate > 0, s + candidate > s.
    # We can just iterate.
    # Optimization: 
    # If candidate is large, maybe fewer collisions?
    
    # Let's check collisions first.
    for s in sums_set:
        if s + candidate in sums_set:
            return None, None
            
    # 2. Generate new sums set
    new_sums_set = set(sums_set)
    new_sums_set.update(s + candidate for s in sums_set)
    
    # 3. Check divisibility
    # We need to check if any sum in new_sums_set (non-zero) divides any other.
    # This is expensive to do from scratch.
    # However, we only need to check:
    # a) Divisibility involving the NEW sums (s + candidate).
    # b) Divisibility between OLD sums is already guaranteed valid.
    
    # Let S_old be sums_set \ {0} (non-empty old sums)
    # Let S_new be {s + candidate for s in sums_set}
    # Note: S_new includes candidate (from s=0) and other sums.
    # We must check:
    # 1. Does any x in S_new divide any y in S_old?
    # 2. Does any x in S_old divide any y in S_new?
    # 3. Does any x in S_new divide any y in S_new?
    
    # To make this fast, we can maintain a sorted list of sums?
    # Or just use the sets.
    # Given the small size (max 30 elements -> 2^30 sums is too big, but we prune heavily).
    # Actually, the number of sums grows exponentially.
    # If we have 10 elements, we have ~1000 sums.
    # If we have 20 elements, ~1M sums.
    # Checking divisibility is O(N^2) on sums. 1M^2 is too slow.
    
    # We need a smarter divisibility check.
    # Key insight: If we maintain the property that no sum divides another, 
    # we can check new sums against the existing ones.
    
    # Let's extract the lists.
    # For performance, we should keep the sums in a sorted list or similar.
    # But for now, let's try to optimize the check.
    
    # S_new = {s + candidate for s in sums_set}
    # S_old = sums_set - {0}
    
    # Check 1: x in S_new, y in S_old, x | y or y | x
    # Check 2: x, y in S_new, x | y
    
    # This is still potentially expensive.
    # However, the number of elements is small.
    # Let's assume the number of sums stays manageable (e.g., < 50,000).
    
    new_sums_list = list(new_sums_set)
    # Remove 0 if present (it is, from 0+0? No, candidate>0, so min sum is candidate. 0 is in sums_set, but 0+candidate=candidate. So 0 is NOT in S_new. 
    # Wait, S_new is derived from s in sums_set. s=0 gives candidate. 
    # The old 0 remains in new_sums_set. 
    # So we need to check non-zero sums in new_sums_set.
    
    non_zero_new = [s for s in new_sums_list if s > 0]
    non_zero_new.sort()
    
    # To check divisibility efficiently:
    # We know S_old is valid.
    # We need to check interactions with S_new.
    
    # Let's just do a full check on the non-zero sums of the new set if the size is small enough.
    # If len(non_zero_new) > 10000, this might be slow.
    # But with greedy pruning, we hope it stays small.
    
    # Optimization: 
    # If a > max_sum/2, it divides nothing.
    max_sum = non_zero_new[-1]
    
    # We only need to check pairs (a, b) where a <= b and b % a == 0.
    # We can iterate a from smallest to largest.
    # For each a, we check if any multiple of a is in the set.
    
    # Build a set for fast lookup? We already have new_sums_set.
    
    for a in non_zero_new:
        if a > max_sum // 2:
            break
        # Check multiples of a
        # m = 2*a, 3*a, ... <= max_sum
        # If m is in new_sums_set, then a divides m.
        # Note: a is in the set. We don't check a dividing a.
        # We check if any OTHER sum is a multiple of a.
        # Since we break at a > max_sum/2, 2*a > max_sum, so we only need to check if 2*a, 3*a etc are in the set.
        # Actually, if a is small, 2*a might be in the set.
        
        # We can just check if any multiple exists.
        # But we must ensure the multiple is NOT a itself.
        # Since m >= 2*a, m != a.
        
        # So:
        for m in range(2 * a, max_sum + 1, a):
            if m in new_sums_set:
                return None, None
                
    # If we get here, it's valid.
    new_elements = elements + [candidate]
    return new_elements, new_sums_set

def construct():
    deadline = time.time() + 45.0
    
    # Start with an empty set
    best_elements = []
    best_sums = {0}
    
    # We will try to build the set greedily.
    # To allow for different structures, we can try different orderings of candidates.
    # However, the "validity" of adding an element depends on the current set of sums.
    # The order matters.
    
    # Strategy:
    # 1. Try adding candidates in descending order (large numbers first).
    # 2. If stuck, try random restarts with different orderings.
    
    # Helper to run a greedy pass
    def greedy_pass(order):
        current_elements = []
        current_sums = {0}
        
        for candidate in order:
            if time.time() > deadline:
                break
            new_elements, new_sums = _try_add(current_elements, current_sums, candidate)
            if new_elements is not None:
                current_elements = new_elements
                current_sums = new_sums
                
        return current_elements, current_sums

    # Candidate pool
    # We want distinct integers in 1..256.
    # Large numbers are generally better for avoiding divisibility and collisions?
    # Actually, large numbers make sums large, which reduces the chance of one sum dividing another 
    # (since the divisor must be small relative to the dividend).
    # Also, large numbers are less likely to create small subset sums that divide others.
    
    # Let's try descending order first.
    order_desc = list(range(256, 0, -1))
    elems, sums = greedy_pass(order_desc)
    
    if len(elems) > len(best_elements):
        best_elements = elems
        best_sums = sums
        
    # If the result is small, try randomized orders.
    # We want to explore different "shapes" of the sum set.
    
    import random
    random.seed(42)
    
    # Limit restarts to ensure we finish in time.
    # Each restart involves up to 256 attempts.
    # _try_add is the bottleneck.
    
    restarts = 0
    max_restarts = 200 # Adjust based on speed
    
    while time.time() < deadline - 5.0 and restarts < max_restarts:
        restarts += 1
        
        # Generate a random order, but maybe bias towards larger numbers?
        # Or just shuffle.
        # Shuffling 1..256
        order = list(range(1, 257))
        random.shuffle(order)
        
        # To improve success rate, maybe start with a few large numbers?
        # Let's just shuffle for now.
        
        elems, sums = greedy_pass(order)
        
        if len(elems) > len(best_elements):
            best_elements = elems
            best_sums = sums
            
            # If we found a significantly better solution, we can continue.
            # If we hit a local optimum, shuffling might escape it.
            
    # Final validation
    if best_elements:
        if not _validate_full(best_elements):
            # Fallback to a known valid set
            # [20, 21, 23] was found in the previous code comments to be valid.
            # Let's verify [20, 21, 23]
            # Sums: 20, 21, 23, 41, 43, 44, 64
            # 20, 21, 23, 41, 43, 44, 64
            # 20 divides 40(no), 60(no), 80(no). 44 no.
            # 21 divides 42(no), 63(no), 84(no).
            # 23 divides 46(no), 69(no).
            # 41 divides 82(no).
            # 43 divides 86(no).
            # 44 divides 88(no).
            # Seems valid.
            best_elements = [20, 21, 23]
            
    return list(best_elements)

if __name__ == "__main__":
    print(construct())
