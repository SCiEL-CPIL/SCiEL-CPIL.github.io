"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest interval [start, start+length-1] within [1, 1008]
    such that for all x, y in the interval, (x * y) % 1009 != 1.
    
    Strategy:
    1. Precompute modular inverses for 1..1008 mod 1009.
    2. Identify "forbidden pairs" (i, inv[i]) where i != inv[i] (since if i == inv[i], i^2 % 1009 == 1, meaning i=1 or i=1008, which are self-inverse and thus forbidden individually).
       Actually, the condition is x*y % 1009 != 1.
       If x = y, we need x^2 % 1009 != 1.
       The solutions to x^2 = 1 mod 1009 are x = 1 and x = 1008 (-1 mod 1009).
       So 1 and 1008 cannot be in the interval.
       For x != y, we need y != inv[x].
       
    3. This is a constraint satisfaction problem on a 1D array.
       We want the longest contiguous subarray [s, e] such that:
       - 1 not in [s, e]
       - 1008 not in [s, e]
       - For all x in [s, e], inv[x] is not in [s, e] (if inv[x] != x).
         Note: if inv[x] == x, then x^2 % 1009 == 1, which implies x=1 or x=1008.
         Since 1 and 1008 are already excluded, we only need to worry about distinct pairs.
       
    4. Efficient approach:
       Let's define an array `right_limit` of size 1009 (indices 1..1008).
       For each index i in 1..1008:
       - If i == 1 or i == 1008, mark as invalid (cannot be in interval).
       - Otherwise, let j = inv[i].
         If j > i, then if i is in the interval, j must NOT be in the interval.
         This means if the interval contains i, it cannot extend to j.
         So, for a start s, if s <= i, the end e must be < j.
         
       Actually, a two-pointer (sliding window) approach is suitable here.
       Let's maintain a window [left, right].
       We want to maximize right - left + 1.
       
       Constraints:
       - The window cannot contain 1 or 1008.
       - For any x in the window, inv[x] must not be in the window.
       
       Let's iterate `left` from 1 to 1008.
       For a fixed `left`, we want to find the maximum `right` such that the window [left, right] is valid.
       
       How to check validity quickly?
       We can precompute for each `left`, the minimum `right` that would violate the condition.
       Or, we can use a two-pointer sweep.
       
       Let's try a two-pointer sweep:
       Initialize `right = 0`.
       For `left` from 1 to 1008:
         While `right` < 1008 and adding `right+1` to the window [left, right] keeps it valid:
           `right += 1`
         Update best length.
         Remove `left` from the window (implicitly by incrementing left in the next step).
         
       How to check if adding `next_right = right + 1` is valid?
       The new window is [left, next_right].
       Conditions for validity:
       1. next_right is not 1 or 1008. (Actually, if left > 1, 1 is already out. If left <= 1, 1 is in. So we must ensure the window doesn't contain 1 or 1008.)
          Since we are sweeping left from 1, if left == 1, the window contains 1, which is invalid. So left must be > 1.
          Similarly, if the window contains 1008, it's invalid.
          
       Let's refine:
       We can precompute `forbidden_in_window` or just check constraints.
       
       Alternative:
       For each start `s`, the end `e` is limited by:
       - e < 1009
       - s > 1 (since 1 is forbidden)
       - e < 1008 (since 1008 is forbidden) -- wait, if e=1008, 1008 is in the window. So e <= 1007.
       - For all x in [s, e], inv[x] is not in [s, e].
         This is equivalent to:
         For all x in [s, e], if inv[x] > x, then inv[x] > e.
         If inv[x] < x, then inv[x] < s.
         
       So, for a fixed `s`, `e` must be less than min( { inv[x] for x in [s, e] and inv[x] > x } union { 1009 } ).
       And `e` must be such that for all x in [s, e], if inv[x] < x, then inv[x] < s.
       The second condition is independent of `e` (for a fixed `s`). It just restricts which `s` are valid starts or which `x` can be in the window.
       Specifically, if there exists x in [s, e] such that inv[x] < s, then the window is invalid.
       This means for a fixed `s`, the window can only contain x such that inv[x] >= s (if inv[x] < x) or inv[x] > e (if inv[x] > x).
       
       Let's use the two-pointer approach with a count of "bad" pairs or just direct checking.
       Since N=1008 is small, O(N^2) is 10^6 operations, which is very fast.
       We can simply iterate all pairs (s, e) with s <= e and check validity?
       1008 * 1008 / 2 ~ 500,000 checks.
       Each check can be O(N) in the worst case, so 500,000 * 1000 = 5 * 10^8, which might be slow in Python (1-5 seconds).
       We can optimize the check.
       
       Better: Precompute `limit[s]` = the maximum `e` such that [s, e] is valid.
       How to compute `limit[s]`?
       Start with `limit[s] = 1008`.
       For each x in [s, 1008]:
         If x == 1 or x == 1008: invalid.
         Else:
           j = inv[x]
           If j == x: invalid (but this only happens for 1, 1008).
           If j < s: invalid for any e >= x. So limit[s] = min(limit[s], x-1).
           If j > x: then e must be < j. So limit[s] = min(limit[s], j-1).
           
       Wait, the condition "for all x in [s, e], inv[x] not in [s, e]" can be broken down:
       For a fixed s, as we increase e, we add elements x.
       When we add x, we must ensure inv[x] is not in [s, e].
       Since x is in [s, e], inv[x] is in [s, e] if s <= inv[x] <= e.
       So if we include x, we must have e < inv[x] (if inv[x] > x) or s > inv[x] (if inv[x] < x).
       The condition s > inv[x] is a constraint on s (or x). If it's violated, x cannot be in the window for this s.
       The condition e < inv[x] is a constraint on e.
       
       So for a fixed s:
       1. Check if s itself is valid: s != 1, s != 1008.
          If not, limit[s] = s-1 (empty).
       2. Let e_max = 1008.
       3. For x from s to 1008:
          If x == 1 or x == 1008:
             break (or set e_max = x-1 and break? No, if x is forbidden, e must be < x).
             e_max = min(e_max, x-1)
             break # since x is contiguous, if x is forbidden, any e >= x is invalid.
          j = inv[x]
          if j == x:
             e_max = min(e_max, x-1)
             break
          if j < s:
             # inv[x] is to the left of s, so it's not in [s, e] for any e.
             # This is fine.
             pass
          else:
             # j >= s.
             # If j <= e, then j is in [s, e], which is bad.
             # So we need e < j.
             e_max = min(e_max, j-1)
       
       Then the valid interval for start s is [s, e_max].
       Length = e_max - s + 1.
       
       We can compute this for all s in O(N^2) worst case, but with early breaks it's likely faster.
       Actually, we can do it in O(N) or O(N log N) with a smarter sweep, but O(N^2) with N=1008 is ~10^6 ops, which is fine.
       
       Let's implement the O(N^2) approach with early termination.
       
    5. Implementation details:
       - Compute inv[1..1008].
       - For s in 1..1008:
           e_max = 1008
           valid = True
           for x in s..1008:
               if x == 1 or x == 1008:
                   e_max = min(e_max, x-1)
                   break
               j = inv[x]
               if j == x:
                   e_max = min(e_max, x-1)
                   break
               if j >= s:
                   e_max = min(e_max, j-1)
               # if j < s, no constraint on e from this x.
           
           if e_max >= s:
               length = e_max - s + 1
               if length > best_length:
                   best_length = length
                   best_start = s
       
       Return [best_start, best_length].
       
    6. Optimization:
       The inner loop can be optimized.
       Notice that `e_max` is non-increasing as x increases? Not necessarily, but `j-1` can be smaller or larger.
       However, we can precompute `next_forbidden[s]` or similar.
       
       Actually, let's look at the constraints again.
       For a fixed s, the constraints on e are:
       e < 1009
       If s <= 1 <= e, invalid. (So if s=1, invalid. If s>1, 1 is out).
       If s <= 1008 <= e, invalid. (So if s<=1008, e must be < 1008).
       For each x in [s, e], if inv[x] >= s, then e < inv[x].
       
       So e_max(s) = min( 1007, min( { inv[x] - 1 for x in [s, e_max(s)] and inv[x] >= s and inv[x] > x } ) )
       This is recursive.
       
       Let's just use the two-pointer method directly, which is O(N) amortized if we can maintain the constraints.
       But the constraints are complex.
       
       Given N=1008, the O(N^2) approach is perfectly fine.
       1008 * 1008 = 1,016,064 iterations.
       Each iteration is a few integer ops.
       This will run in < 0.1 seconds in Python.
       
    7. Edge cases:
       - If no valid interval is found, return [1, 1]? But 1 is invalid.
       - The problem states "nonempty interval".
       - Are there any valid intervals?
         1 and 1008 are forbidden.
         2: inv[2] = 505. (2*505 = 1010 = 1 mod 1009).
         So if 2 is in the interval, 505 cannot be.
         If we pick [2, 3], is it valid?
         x=2, y=2: 4 != 1.
         x=2, y=3: 6 != 1.
         x=3, y=2: 6 != 1.
         x=3, y=3: 9 != 1.
         inv[2]=505, inv[3]=336.
         505 not in [2,3]. 336 not in [2,3].
         So [2,3] is valid.
         
       The seed program returns [2, 30], which means start=2, length=30, so interval [2, 31].
       Let's check if [2, 31] is valid.
       We need to ensure no pair (x,y) in [2,31] has x*y % 1009 == 1.
       This implies no x in [2,31] has its inverse in [2,31].
       inv[2]=505 (out).
       inv[3]=336 (out).
       ...
       inv[31] = ? 31 * x = 1 mod 1009.
       1009 / 31 = 32.54.
       31 * 32 = 992. 1009 - 992 = 17.
       Let's compute properly.
       
       The algorithm will find the true maximum.
       
    8. Code Structure:
       - Precompute inverses.
       - Loop s from 1 to 1008.
       - Loop x from s to 1008.
       - Track e_max.
       - Update best.
       - Return best.
       
    9. Time limit: 90s. This is way more than enough.
    
    10. Let's write the code.
    """
    N = 1008
    MOD = 1009
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
        
    best_start = 1
    best_length = 0
    
    for s in range(1, MOD):
        e_max = MOD - 1  # 1008
        # Check if s itself is forbidden
        if s == 1 or s == MOD - 1:
            continue
        
        # Determine the maximum valid end e for start s
        # We iterate x from s upwards
        # If we encounter a forbidden x (1 or 1008), e_max becomes x-1 and we stop.
        # If we encounter an x such that inv[x] is in [s, e_max], we restrict e_max.
        
        # Actually, we can just iterate x and update e_max.
        # But if e_max drops below s, we can break.
        
        valid = True
        current_e = MOD - 1
        
        for x in range(s, MOD):
            if x == 1 or x == MOD - 1:
                current_e = min(current_e, x - 1)
                break
            
            j = inv[x]
            
            # If j == x, then x*x % MOD == 1.
            # This happens for x=1 and x=1008.
            # Since we handled those above, j != x for x in [2, 1007].
            
            if j >= s:
                # If j is in the potential window [s, current_e], we must exclude it.
                # So the window must end before j.
                current_e = min(current_e, j - 1)
            
            # If j < s, it's outside the window to the left, so no restriction on e.
            
            # If current_e < s, no valid interval for this s.
            if current_e < s:
                break
                
        if current_e >= s:
            length = current_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
                
    return [best_start, best_length]

# EVOLVE-BLOCK-END
