"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def _is_safe(candidate, others):
    """Return True if 1/candidate is NOT equal to sum of reciprocals of any non-empty
    subset of `others`."""
    target = Fraction(1, candidate)
    n = len(others)
    if n == 0:
        return True
    
    # Meet-in-the-middle for subset sum
    mid = n // 2
    left = others[:mid]
    right = others[mid:]
    
    # Generate all subset sums for left
    left_sums = {Fraction(0, 1)}
    for x in left:
        new_sums = set()
        for s in left_sums:
            ns = s + Fraction(1, x)
            if ns < target:
                new_sums.add(ns)
            elif ns == target:
                # This sum equals target using only left elements
                # But we need a subset of OTHERS (which includes both left and right)
                # If we found target using only left, that's a valid subset of others.
                # Wait, the condition is: target must NOT be in subset sums of others.
                # If target is found, it's not safe.
                # However, we must ensure the subset is non-empty.
                # Fraction(0) corresponds to empty set. Target is 1/c > 0, so any match is non-empty.
                return False
        left_sums.update(new_sums)
    
    # Check if any left sum + right sum == target
    # For each sum in right, check if (target - right_sum) is in left_sums
    right_sums = {Fraction(0, 1)}
    for x in right:
        new_sums = set()
        for s in right_sums:
            ns = s + Fraction(1, x)
            if ns < target:
                new_sums.add(ns)
            elif ns == target:
                return False
        right_sums.update(new_sums)
    
    # Now check combinations
    for rs in right_sums:
        if rs > target:
            continue
        remainder = target - rs
        if remainder in left_sums:
            # We need to ensure the combined subset is non-empty.
            # Since target > 0, and all terms are positive, if remainder == 0 and rs == target, 
            # that's handled above (rs==target check in right loop).
            # If rs == 0 and remainder == target, then target is in left_sums, handled in left loop.
            # Actually, if remainder == 0, then rs == target. This would have been caught in the right loop generation?
            # No, in the right loop we check `elif ns == target: return False`.
            # So if rs == target, we would have already returned False.
            # Thus, any match here involves rs < target and remainder > 0, or rs > 0 and remainder < target.
            # The subset is non-empty because target > 0.
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Strategy: Greedy construction from largest to smallest
    # This tends to work well for reciprocal subsum problems because
    # larger numbers have smaller reciprocals, making it harder to sum to
    # the reciprocal of a smaller number? 
    # Actually, let's think: 
    # If we pick large numbers first, their reciprocals are small.
    # Then we try to add smaller numbers. The reciprocal of a smaller number is larger.
    # It is less likely that a sum of small reciprocals equals a large reciprocal?
    # No, sum of small positives can equal a larger positive.
    # Example: 1/2 + 1/3 + 1/6 = 1/1. 
    # So small reciprocals can sum to a large reciprocal.
    # Therefore, adding small numbers (large reciprocals) is risky because they are
    # targets that can be formed by sums of the already selected large numbers (small reciprocals).
    
    # Conversely, if we add small numbers first (large reciprocals),
    # then adding large numbers (small reciprocals) is safe because
    # the target (small reciprocal) is too small to be formed by the sum of
    # any subset of the already selected large reciprocals?
    # Yes! If all selected elements are < N, their reciprocals are > 1/N.
    # The sum of any non-empty subset is > 1/N.
    # If we then try to add N, the target is 1/N.
    # Since all existing reciprocals are > 1/N, their sum is > 1/N.
    # So 1/N cannot be formed.
    # This suggests adding in increasing order (1, 2, 3, ...) is very safe!
    
    # Let's try greedy increasing order.
    selected = []
    for i in range(1, 25):
        if time.time() > deadline:
            break
        if _is_safe(i, selected):
            selected.append(i)
            
    # Now, can we improve? 
    # Maybe we skipped some numbers. Can we swap or add?
    # The greedy increasing order will select 1, 2, 3, ...
    # Let's trace:
    # i=1: selected=[]. Safe. selected=[1].
    # i=2: others=[1]. Target=1/2. Subset sums of [1]: {1/1=1}. 1/2 != 1. Safe. selected=[1,2].
    # i=3: others=[1,2]. Target=1/3. Sums of {1,2}: {1, 1/2, 1.5}. 1/3 not in sums. Safe. selected=[1,2,3].
    # i=4: others=[1,2,3]. Target=1/4. Sums: 
    #   1, 1/2, 1/3, 1+1/2=1.5, 1+1/3=1.33, 1/2+1/3=0.833, 1+1/2+1/3=1.833.
    #   1/4=0.25. Not in sums. Safe. selected=[1,2,3,4].
    # i=5: Target=1/5=0.2. Sums of {1,2,3,4}. Min sum is 1/4=0.25. 
    #   Any subset sum is >= 1/4 = 0.25 > 0.2.
    #   So 1/5 is NOT in sums. Safe.
    # i=6: Target=1/6 ~ 0.166. Min sum of subset of {1,2,3,4,5} is 1/5=0.2.
    #   0.166 < 0.2. So safe.
    # ...
    # It seems we can select ALL numbers 1..24?
    # Let's check i=1 with others 2..24.
    # Wait, the greedy builds sequentially. 
    # When we add i, `selected` contains {1, ..., i-1}.
    # The min reciprocal in selected is 1/(i-1).
    # The target is 1/i.
    # Since i > i-1, 1/i < 1/(i-1).
    # The smallest subset sum of selected is 1/(i-1).
    # Since 1/i < 1/(i-1), it is impossible for any subset sum to equal 1/i.
    # Therefore, EVERY step is safe!
    # So {1, 2, ..., 24} is a valid set?
    
    # Let's double check the definition.
    # "no reciprocal of a selected element may equal the sum of reciprocals of a subset of the OTHER selected elements"
    # For i=1, others are {2..24}. Target 1/1=1.
    # Can 1 be formed by subset of {2..24}?
    # 1/2 + 1/3 + 1/6 = 1. But 6 is in {2..24}. 2 and 3 are in {2..24}.
    # So yes, 1/1 = 1/2 + 1/3 + 1/6.
    # So 1 is NOT safe if {2,3,6} are selected.
    # My greedy logic above:
    # i=1: selected=[]. Safe. Add 1.
    # i=2: selected=[1]. Target 1/2. Subset of [1] sums to 1. 1 != 1/2. Safe. Add 2.
    # i=3: selected=[1,2]. Target 1/3. Subsets of [1,2] sum to 1, 1/2, 1.5. None is 1/3. Safe. Add 3.
    # ...
    # i=6: selected=[1,2,3,4,5]. Target 1/6.
    # Min sum is 1/5=0.2. 1/6=0.166. 1/6 < 1/5. So no subset sum can be 1/6. Safe. Add 6.
    # 
    # The issue arises when we check a PREVIOUSLY selected element against the NEWLY added ones?
    # NO. The condition is symmetric?
    # "no reciprocal of a selected element a may equal the sum of reciprocals of a subset of the OTHER selected elements".
    # This must hold for ALL selected elements a.
    # When we add 6, we checked if 1/6 is a sum of subset of {1,2,3,4,5}. It wasn't.
    # But we did NOT check if 1/1 is a sum of subset of {2,3,4,5,6}.
    # And 1/1 = 1/2 + 1/3 + 1/6.
    # So the set {1,2,3,4,5,6} is INVALID because of element 1.
    
    # So simple increasing greedy fails because adding new elements can invalidate old ones.
    
    # We need a strategy where adding new elements doesn't create sums for existing targets.
    # If we add elements in DECREASING order (24, 23, ..., 1):
    # i=24: selected=[]. Safe. Add 24.
    # i=23: selected=[24]. Target 1/23. Subset of [24] sums to 1/24. 1/23 != 1/24. Safe. Add 23.
    # i=22: selected=[23,24]. Target 1/22. Min sum 1/24. Max sum 1/23+1/24.
    # 1/22 ~ 0.045. 1/24 ~ 0.041. 1/23 ~ 0.043.
    # 1/23+1/24 ~ 0.085.
    # 1/22 is between 1/23 and 1/23+1/24?
    # 1/22 = 0.0454. 1/23 = 0.0434.
    # So 1/22 > 1/23.
    # Can 1/22 be formed?
    # Subsets of {23,24}: {1/23, 1/24, 1/23+1/24}.
    # None is 1/22. Safe. Add 22.
    # 
    # In general, if we add in decreasing order, when we add k, selected are {k+1, ..., 24}.
    # Target is 1/k.
    # The largest reciprocal in selected is 1/(k+1).
    # The sum of ANY subset of selected is at most sum(1/j for j=k+1..24).
    # But more importantly, is it possible that 1/k is formed by sum of 1/j for j>k?
    # Since j>k, 1/j < 1/k.
    # Sum of multiple 1/j could be 1/k.
    # Example: k=1. selected={2..24}. 1/1 = 1. Sum of 1/2+1/3+1/6 = 1. So 1 fails.
    # k=2. selected={3..24}. Target 1/2. Can 1/2 be formed by 1/3+1/6? Yes. So 2 fails if 3,6 present.
    
    # So decreasing order also has issues.
    
    # The seed {13..24} works. Size 12.
    # Let's try to expand {13..24} by adding 12, 11, etc.
    # Check if 12 can be added to {13..24}.
    # Target 1/12. Others {13..24}.
    # Min sum 1/24. Max sum sum(1/13..1/24).
    # 1/12 ~ 0.0833.
    # 1/13 ~ 0.0769.
    # 1/14 ~ 0.0714.
    # 1/13+1/14 ~ 0.148.
    # 1/13 > 1/12? No, 1/13 < 1/12.
    # So 1/12 is larger than any single element's reciprocal in {13..24}.
    # Can it be a sum of two or more?
    # Smallest sum of two: 1/24 + 1/23 ~ 0.085.
    # 1/12 = 0.0833.
    # 0.0833 < 0.085.
    # So 1/12 is LESS than the smallest sum of two elements from {13..24}.
    # And 1/12 is GREATER than the largest single element 1/13.
    # Therefore, 1/12 CANNOT be formed by any subset of {13..24}.
    # So 12 is safe to add!
    
    # Now set is {12..24}. Size 13.
    # Check 11. Target 1/11 ~ 0.0909.
    # Others {12..24}.
    # Largest single: 1/12 ~ 0.0833.
    # Smallest sum of two: 1/24+1/23 ~ 0.085.
    # 1/11 = 0.0909.
    # 1/11 > 1/12.
    # 1/11 > 1/24+1/23.
    # Is 1/11 < 1/24+1/23+1/22?
    # 1/24+1/23+1/22 ~ 0.041+0.043+0.045 = 0.129.
    # So 1/11 is between sum of 2 smallest and sum of 3 smallest.
    # Could it be exactly a sum?
    # We need to check subset sums.
    # This is where the search comes in.
    
    # Let's use a backtracking search to find the maximum set.
    # Start with {13..24} and try to add smaller numbers, or explore other configurations.
    # Actually, a simple greedy from 24 down to 1, skipping unsafe ones, might work well.
    # Let's simulate decreasing greedy again with the "check all previous" logic?
    # No, the check must be: for the new element, check against existing.
    # AND, existing elements must remain valid against the new one?
    # If we add k, we check if 1/k is sum of subset of S.
    # But we also need to ensure that for all x in S, 1/x is not sum of subset of (S \ {x}) U {k}.
    # This is expensive.
    
    # However, note: if we add elements in decreasing order,
    # and we only check if 1/k is in subset sums of S,
    # does this guarantee the set is valid?
    # Suppose S is valid. We add k.
    # We check 1/k not in subset sums of S.
    # Now, is the new set S U {k} valid?
    # For any x in S, 1/x was not in subset sums of S\{x}.
    # Now the "others" for x are (S\{x}) U {k}.
    # Could 1/x be sum of subset of (S\{x}) U {k}?
    # It could be formed by including k.
    # i.e., 1/x = sum(subset of S\{x}) + 1/k.
    # This means sum(subset of S\{x}) = 1/x - 1/k.
    # Since we added k in decreasing order, k < x (if we go 24, 23...).
    # Wait, if we go 24, 23, ..., then k < x.
    # 1/k > 1/x.
    # So 1/x - 1/k < 0.
    # But subset sums are positive.
    # So it's impossible to form 1/x using k and any subset of S\{x} if k < x.
    # Therefore, adding in DECREASING order, we ONLY need to check the new element k against the existing set S!
    # We do NOT need to re-check existing elements.
    
    # This is a huge simplification.
    # Algorithm:
    # 1. Start with empty set.
    # 2. Iterate i from 24 down to 1.
    # 3. If 1/i is NOT in subset sums of current set, add i to set.
    # 4. Return set.
    
    # Let's test this logic on the 1,2,3,6 example.
    # Order: 6, 3, 2, 1 (decreasing).
    # i=6: S={}. 1/6 not in {}. Add 6. S={6}.
    # i=5: S={6}. 1/5 not in {1/6}. Add 5. S={5,6}.
    # i=4: S={5,6}. 1/4 not in {1/5,1/6,1/5+1/6}. Add 4. S={4,5,6}.
    # i=3: S={4,5,6}. 1/3 not in subset sums of {4,5,6}. 
    #    Sums: 1/6, 1/5, 1/4, 1/5+1/6, 1/4+1/6, 1/4+1/5, 1/4+1/5+1/6.
    #    1/3 ~ 0.33. 1/4=0.25, 1/5=0.2, 1/6=0.166.
    #    Max sum ~ 0.61. Min 0.166.
    #    Is 1/3 a sum? 1/4+1/12? No 12 not in set.
    #    1/3 is not formed by 4,5,6. Add 3. S={3,4,5,6}.
    # i=2: S={3,4,5,6}. Target 1/2.
    #    1/3+1/6 = 1/2. 
    #    So 1/2 IS in subset sums of S.
    #    Do NOT add 2.
    # i=1: S={3,4,5,6}. Target 1.
    #    1/3+1/6+1/2? 2 not in S.
    #    1/3+1/4+1/5+1/6 = 0.33+0.25+0.2+0.166 = 0.946.
    #    Max sum is < 1. So 1 is not formed.
    #    Add 1. S={1,3,4,5,6}.
    
    # Is {1,3,4,5,6} valid?
    # Check 1: Others {3,4,5,6}. 1 = 1/3+1/6+1/2? No 2 not there.
    # 1/3+1/4+1/5+1/6 < 1.
    # So 1 is safe.
    # Check 3: Others {1,4,5,6}. 1/3 = 1/1 + ...? No, 1/1=1 > 1/3.
    # Any sum involving 1 is > 1. So 1/3 cannot be formed if 1 is in others?
    # Wait, 1/3 < 1. So any subset containing 1 sums to > 1.
    # Subsets not containing 1 are from {4,5,6}. Max sum < 1.
    # So 1/3 is safe.
    
    # So decreasing greedy seems valid and efficient.
    
    selected = []
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        if _is_safe(i, selected):
            selected.append(i)
            
    return selected
# EVOLVE-BLOCK-END
