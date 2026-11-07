import random
import time
import math
from itertools import combinations

def _all_subset_sums(elements):
    """Compute all non-empty subset sums."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _check_no_divisibility(sums):
    """Check that no two distinct sums in the set divide each other."""
    sl = sorted(sums)
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            if sl[j] % sl[i] == 0:
                return False
    return True

def _is_valid(elements):
    """Full validity check."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not (1 <= x <= 256):
            return False
    sums = _all_subset_sums(elements)
    if len(sums) != (1 << len(elements)) - 1:
        return False  # duplicate sums exist
    return _check_no_divisibility(sums)

def _try_greeedy(elements, deadline):
    """
    Greedily try to add the smallest possible integer > max(elements) 
    that keeps the set valid. If none found in a reasonable range, 
    try random candidates.
    """
    current = list(elements)
    current_sums = _all_subset_sums(current) if current else set()
    max_elem = max(current) if current else 0
    
    # We want to maximize size, so we prefer smaller numbers if possible,
    # but smaller numbers create more divisibility conflicts.
    # Actually, for "no two sums divide", larger numbers are "safer" 
    # because they are less likely to be divisors of other sums.
    # However, we are constrained to 1..256.
    # Let's try adding candidates from 1 to 256, skipping those in current.
    # We will iterate through candidates in a specific order.
    
    # Strategy: Try candidates in increasing order? 
    # If we pick small numbers, we fill up the "small sum" space quickly, 
    # making it harder to add more. 
    # If we pick large numbers, we have more room.
    # But we want MAX SIZE. 
    # Usually, for these problems, a mix is needed.
    
    # Let's try a randomized greedy: 
    # 1. Generate all possible candidates 1..256 not in current.
    # 2. Shuffle them? No, let's try to add the one that is "least conflicting".
    # Simpler: Just try to add the next integer that works, scanning 1..256.
    # If we get stuck, we might have a local maximum.
    
    # Let's implement a simple loop that tries to add *any* valid candidate.
    # We'll prioritize candidates that are larger to avoid divisibility traps?
    # Actually, if $a | b$, $a$ must be smaller. So smaller numbers are the "divisors".
    # To avoid $s_1 | s_2$, we want to avoid small sums being divisors of large sums.
    # This suggests we should avoid small numbers if they create small sums.
    
    # Let's try to build the set by adding numbers in descending order?
    # No, let's stick to a robust randomized greedy with a tight deadline.
    
    # Re-evaluating the previous failure: It timed out.
    # The new approach: 
    # 1. Start with empty set.
    # 2. Repeatedly find the best candidate to add.
    #    "Best" = smallest index in a shuffled list of 1..256?
    #    Or just iterate 1..256 and pick the first valid one?
    
    # Let's try iterating 1..256 and picking the first valid one.
    # This is deterministic and fast.
    
    while time.time() < deadline:
        added = False
        for x in range(1, 257):
            if x in current:
                continue
            
            # Check if adding x is valid
            # New sums are {x} U {s+x for s in current_sums}
            # We need to check:
            # 1. No duplicate sums (x not in current_sums, and s1+x != s2+x which is true if s1!=s2, 
            #    and s+x != x which implies s!=0 which is true for non-empty subsets).
            #    Wait, duplicate sums check: 
            #    The new set of sums is S_new = S_old U {x} U {s+x | s in S_old}.
            #    Duplicates can occur if:
            #    a) x is in S_old
            #    b) x is in {s+x | s in S_old} -> 0 in S_old (False, non-empty sums > 0)
            #    c) s1 + x is in S_old
            #    d) s1 + x is in {s2 + x | s2 in S_old} -> s1 == s2 (False if distinct)
            
            if x in current_sums:
                continue
                
            new_sums_part = {s + x for s in current_sums}
            
            # Check if any element in new_sums_part is in current_sums
            if new_sums_part & current_sums:
                continue
                
            # Now we have candidate new_sums = current_sums U {x} U new_sums_part
            # We must check divisibility among ALL elements in this new set.
            # This is expensive. 
            # Optimization: We only need to check pairs where at least one is new.
            # Pairs to check:
            # 1. (x, s) for s in current_sums
            # 2. (x, s+x) for s in current_sums
            # 3. (s1+x, s2) for s1, s2 in current_sums
            # 4. (s1+x, s2+x) -> equivalent to (s1, s2) which is already valid.
            
            # So we check:
            # - Does x divide any s in current_sums?
            # - Does any s in current_sums divide x?
            # - Does any s in current_sums divide (s'+x)?
            # - Does (s'+x) divide any s in current_sums?
            # - Does (s1+x) divide (s2+x)? (This is same as s1|s2, already checked)
            
            # Actually, just checking the full set is safer and for n<=15, 
            # |S| <= 32767. O(|S|^2) is too slow if we do it for every candidate.
            # But |S| is small for small n.
            
            # Let's just do the full check for simplicity, but only for promising candidates.
            # We can pre-filter: if x divides any existing sum, skip.
            valid = True
            for s in current_sums:
                if s % x == 0:
                    valid = False
                    break
                if x % s == 0:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check new sums against old sums
            for s_new in new_sums_part:
                for s_old in current_sums:
                    if s_new % s_old == 0:
                        valid = False
                        break
                    if s_old % s_new == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # If we pass these, it's likely valid. 
            # We should still do a full check to be 100% sure, 
            # but for speed we might skip if we trust the partial checks.
            # The partial checks cover:
            # - x vs old
            # - new (s+x) vs old
            # - new vs new is covered by old vs old (since (s1+x)%(s2+x) != 0 is NOT equivalent to s1%s2 != 0).
            # WAIT. (s1+x) and (s2+x). 
            # Example: s1=1, s2=2, x=1. s1+x=2, s2+x=3. 2 does not divide 3.
            # Example: s1=1, s2=3, x=1. s1+x=2, s2+x=4. 2 divides 4. 
            # But s1=1, s2=3. 1 divides 3. So if (1,3) was valid in old set, it means 1 did NOT divide 3? 
            # No, the condition is "no two distinct values divide one another".
            # So if 1 and 3 are in the old set, 1|3 is a violation. So (1,3) cannot be in a valid set.
            # Therefore, if the old set is valid, s1 does not divide s2.
            # Does (s1+x) divide (s2+x) imply s1 divides s2? No.
            # Counter-example: s1=1, s2=2, x=1. 2, 3. No.
            # s1=2, s2=4, x=2. 4, 6. 4 does not divide 6.
            # s1=1, s2=4, x=1. 2, 5. No.
            # It is possible that s1 !| s2 but (s1+x) | (s2+x).
            # Example: s1=1, s2=2. 1|2 (Invalid old set).
            # We need a valid old set.
            # Let s1=3, s2=5. 3 !| 5.
            # x=1. s1+x=4, s2+x=6. 4 !| 6.
            # x=2. s1+x=5, s2+x=7.
            # x=3. s1+x=6, s2+x=8.
            # x=4. s1+x=7, s2+x=9.
            # x=5. s1+x=8, s2+x=10.
            # x=6. s1+x=9, s2+x=11.
            # x=7. s1+x=10, s2+x=12. 10 !| 12.
            # x=8. s1+x=11, s2+x=13.
            # x=9. s1+x=12, s2+x=14.
            # x=10. s1+x=13, s2+x=15.
            # x=11. s1+x=14, s2+x=16.
            # x=12. s1+x=15, s2+x=17.
            # x=13. s1+x=16, s2+x=18.
            # x=14. s1+x=17, s2+x=19.
            # x=15. s1+x=18, s2+x=20.
            # x=16. s1+x=19, s2+x=21.
            # x=17. s1+x=20, s2+x=22.
            # x=18. s1+x=21, s2+x=23.
            # x=19. s1+x=22, s2+x=24.
            # x=20. s1+x=23, s2+x=25.
            # x=21. s1+x=24, s2+x=26.
            # x=22. s1+x=25, s2+x=27.
            # x=23. s1+x=26, s2+x=28.
            # x=24. s1+x=27, s2+x=29.
            # x=25. s1+x=28, s2+x=30.
            # x=26. s1+x=29, s2+x=31.
            # x=27. s1+x=30, s2+x=32.
            # x=28. s1+x=31, s2+x=33.
            # x=29. s1+x=32, s2+x=34.
            # x=30. s1+x=33, s2+x=35.
            # x=31. s1+x=34, s2+x=36.
            # x=32. s1+x=35, s2+x=37.
            # x=33. s1+x=36, s2+x=38.
            # x=34. s1+x=37, s2+x=39.
            # x=35. s1+x=38, s2+x=40.
            # x=36. s1+x=39, s2+x=41.
            # x=37. s1+x=40, s2+x=42.
            # x=38. s1+x=41, s2+x=43.
            # x=39. s1+x=42, s2+x=44.
            # x=40. s1+x=43, s2+x=45.
            # x=41. s1+x=44, s2+x=46.
            # x=42. s1+x=45, s2+x=47.
            # x=43. s1+x=46, s2+x=48.
            # x=44. s1+x=47, s2+x=49.
            # x=45. s1+x=48, s2+x=50.
            # x=46. s1+x=49, s2+x=51.
            # x=47. s1+x=50, s2+x=52.
            # x=48. s1+x=51, s2+x=53.
            # x=49. s1+x=52, s2+x=54.
            # x=50. s1+x=53, s2+x=55.
            # x=51. s1+x=54, s2+x=56.
            # x=52. s1+x=55, s2+x=57.
            # x=53. s1+x=56, s2+x=58.
            # x=54. s1+x=57, s2+x=59.
            # x=55. s1+x=58, s2+x=60.
            # x=56. s1+x=59, s2+x=61.
            # x=57. s1+x=60, s2+x=62.
            # x=58. s1+x=61, s2+x=63.
            # x=59. s1+x=62, s2+x=64.
            # x=60. s1+x=63, s2+x=65.
            # x=61. s1+x=64, s2+x=66.
            # x=62. s1+x=65, s2+x=67.
            # x=63. s1+x=66, s2+x=68.
            # x=64. s1+x=67, s2+x=69.
            # x=65. s1+x=68, s2+x=70.
            # x=66. s1+x=69, s2+x=71.
            # x=67. s1+x=70, s2+x=72.
            # x=68. s1+x=71, s2+x=73.
            # x=69. s1+x=72, s2+x=74.
            # x=70. s1+x=73, s2+x=75.
            # x=71. s1+x=74, s2+x=76.
            # x=72. s1+x=75, s2+x=77.
            # x=73. s1+x=76, s2+x=78.
            # x=74. s1+x=77, s2+x=79.
            # x=75. s1+x=78, s2+x=80.
            # x=76. s1+x=79, s2+x=81.
            # x=77. s1+x=80, s2+x=82.
            # x=78. s1+x=81, s2+x=83.
            # x=79. s1+x=82, s2+x=84.
            # x=80. s1+x=83, s2+x=85.
            # x=81. s1+x=84, s2+x=86.
            # x=82. s1+x=85, s2+x=87.
            # x=83. s1+x=86, s2+x=88.
            # x=84. s1+x=87, s2+x=89.
            # x=85. s1+x=88, s2+x=90.
            # x=86. s1+x=89, s2+x=91.
            # x=87. s1+x=90, s2+x=92.
            # x=88. s1+x=91, s2+x=93.
            # x=89. s1+x=92, s2+x=94.
            # x=90. s1+x=93, s2+x=95.
            # x=91. s1+x=94, s2+x=96.
            # x=92. s1+x=95, s2+x=97.
            # x=93. s1+x=96, s2+x=98.
            # x=94. s1+x=97, s2+x=99.
            # x=95. s1+x=98, s2+x=100.
            # x=96. s1+x=99, s2+x=101.
            # x=97. s1+x=100, s2+x=102.
            # x=98. s1+x=101, s2+x=103.
            # x=99. s1+x=102, s2+x=104.
            # x=100. s1+x=103, s2+x=105.
            
            # It seems hard to find a counter example where s1 !| s2 but (s1+x) | (s2+x).
            # If (s1+x) | (s2+x), then s2+x = k(s1+x) = k*s1 + k*x.
            # s2 = k*s1 + (k-1)*x.
            # This doesn't imply s2 is a multiple of s1.
            # Example: s1=3, s2=5.
            # 5 = k*3 + (k-1)x.
            # If k=1, 5=3+x => x=2.
            # Check: s1=3, s2=5, x=2.
            # s1+x=5, s2+x=7. 5 !| 7.
            # Wait, if k=1, s2+x = 1*(s1+x) => s2=s1. But s1!=s2.
            # So k must be >= 2.
            # If k=2, s2+x = 2(s1+x) => s2 = 2*s1 + x.
            # Example: s1=3, x=2 => s2 = 6+2=8.
            # Check s1=3, s2=8. 3 !| 8. Valid old set.
            # x=2.
            # s1+x=5, s2+x=10.
            # 5 | 10. VIOLATION.
            # So we MUST check new vs new.
            
            # So the partial check is insufficient. We must check all pairs in the new set.
            # However, we can optimize by only checking pairs involving at least one new element.
            # Pairs:
            # 1. (x, s) - checked
            # 2. (x, s+x) - checked? 
            #    x | (s+x) => x | s. Checked in "x vs old".
            #    (s+x) | x => s+x <= x => s<=0. False.
            # 3. (s1+x, s2) - checked in "new vs old".
            # 4. (s1+x, s2+x) - NOT checked.
            
            # So we need to check (s1+x) vs (s2+x) for all s1, s2 in current_sums.
            # This is O(|S|^2).
            
            # Given the time limit, we will just do the full check on the candidate set.
            # To speed up, we only do this for candidates that pass the cheap filters.
            
            # Construct the full new set for the final check
            full_new_sums = set(current_sums)
            full_new_sums.add(x)
            full_new_sums.update(new_sums_part)
            
            if _check_no_divisibility(full_new_sums):
                current.append(x)
                current_sums = full_new_sums
                added = True
                break # Restart the scan for the next element
        
        if not added:
            break # No more elements can be added
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin
    
    best = []
    best_len = 0
    
    # Try a few different starting points or strategies
    # Strategy 1: Greedy from 1
    # Strategy 2: Greedy from random small set
    # Strategy 3: Random restarts with greedy completion
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if attempts % 10 == 1:
            # Start with empty set, let greedy pick
            current = []
        else:
            # Start with a small random seed of large numbers
            seed_size = random.randint(1, 3)
            seed = random.sample(range(128, 257), seed_size)
            current = seed
        
        grown = _try_greeedy(current, deadline)
        
        if len(grown) > best_len:
            best = grown
            best_len = len(grown)
            if best_len >= 12: # Good score
                break
        
        # If we found a decent set, try to improve it with local search?
        # For now, just keep trying.
        
    # Final verification
    if best and _is_valid(best):
        return [int(x) for x in best]
    else:
        # Fallback
        return [31, 32, 34]

construct()
