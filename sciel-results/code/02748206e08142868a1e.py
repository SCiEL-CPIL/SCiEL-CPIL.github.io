"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008  # interval is in [1, 1008]
    
    # Precompute modular inverses for all x in [1, p-1]
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # For each position i in [1, n], we need to ensure that for all x, y in the interval,
    # x * y % p != 1. This means for each x in the interval, inv[x] must NOT be in the interval.
    
    # Key insight: if x is in the interval, then inv[x] must not be in the interval.
    # Since the interval is contiguous [start, start+length-1], we can use a two-pointer approach.
    
    # For a given start, what's the maximum end such that for all x in [start, end],
    # inv[x] is not in [start, end]?
    
    # Equivalently: for all x in [start, end], inv[x] < start or inv[x] > end.
    
    # Two-pointer: maintain window [left, right] and expand right as far as possible.
    # When adding new element x = right, we need inv[x] not in [left, right-1] and 
    # also x not equal to inv of any element in [left, right-1].
    
    # But since we're going left to right, when we add x, the only new constraint is:
    #   inv[x] must not be in [left, x-1] (since x is the new rightmost)
    # And also, for all y in [left, x-1], inv[y] must not be x. But inv[y] = x means y = inv[x].
    # So the condition is: inv[x] not in [left, x-1], which is equivalent to inv[x] < left or inv[x] > x-1.
    # Since inv[x] != x unless x is a self-inverse (x^2 ≡ 1 mod p), and x < p, 
    # self-inverses are x=1 and x=p-1=1008.
    
    # Actually, let me reconsider. The condition is:
    # For all x in [left, right], inv[x] not in [left, right].
    
    # When we extend right by 1 to new_right:
    # - We need inv[new_right] not in [left, new_right]
    # - We also need that for all x in [left, new_right-1], inv[x] != new_right
    #   But inv[x] = new_right iff x = inv[new_right]
    #   So we need inv[new_right] not in [left, new_right-1]
    # - Combined: inv[new_right] not in [left, new_right]
    #   Since inv[new_right] != new_right (unless self-inverse), this is:
    #   inv[new_right] < left or inv[new_right] > new_right
    #   But inv[new_right] > new_right is possible.
    
    # Wait, but we also need to check that the existing elements in [left, new_right-1] 
    # don't conflict with new_right. That is, for all x in [left, new_right-1], 
    # x * new_right % p != 1, which means new_right != inv[x] for all x in [left, new_right-1].
    # This is equivalent to: new_right not in {inv[x] : x in [left, new_right-1]}
    # Which is equivalent to: inv[new_right] not in [left, new_right-1]
    # (since inv is a bijection on [1, p-1])
    
    # So the constraint when adding new_right is:
    #   inv[new_right] < left OR inv[new_right] > new_right
    #   (inv[new_right] == new_right only for self-inverses, which would mean new_right is self-inverse,
    #    in which case inv[new_right] == new_right, so we need new_right < left or new_right > new_right,
    #    which is impossible, so self-inverses can never be in a valid interval)
    
    # Let me use a two-pointer sweep:
    left = 1
    best_start = 1
    best_length = 0
    
    for right in range(1, n + 1):
        # Check if right can be added to the current window [left, right]
        # We need to ensure the window [left, right] is valid.
        # But actually, we need to move left forward until the window [left, right] is valid.
        
        # The window [left, right] is valid iff for all x in [left, right], inv[x] not in [left, right].
        
        # When we increment right, we may need to increment left to resolve conflicts.
        # A conflict exists if there's some x in [left, right] such that inv[x] in [left, right].
        
        # Let's think differently. For the window [left, right] to be valid:
        # For all x in [left, right], inv[x] < left or inv[x] > right.
        
        # When we add right to a previously valid window [left, right-1]:
        # - The old elements x in [left, right-1] were fine: inv[x] < left or inv[x] > right-1
        #   Now we need inv[x] < left or inv[x] > right. If inv[x] was between left and right-1, it was already invalid.
        #   So the old elements are still fine as long as they were fine before.
        #   Wait, no. If inv[x] was > right-1 but <= right, that means inv[x] = right, which is a conflict.
        #   So we need: for all x in [left, right-1], inv[x] != right, i.e., x != inv[right].
        #   So if inv[right] in [left, right-1], we have a conflict.
        # - The new element right: we need inv[right] < left or inv[right] > right.
        
        # So the condition for [left, right] to be valid given [left, right-1] was valid:
        #   1. inv[right] < left or inv[right] > right
        #   2. inv[right] not in [left, right-1] (which is implied by 1 if inv[right] != right)
        
        # Actually condition 1 already implies condition 2 when inv[right] != right.
        # If inv[right] == right (self-inverse), then condition 1 requires right < left or right > right, impossible.
        
        # So: [left, right] is valid iff [left, right-1] was valid AND (inv[right] < left OR inv[right] > right).
        
        # But wait, this isn't quite right either. Let me re-examine.
        
        # If [left, right-1] is valid, then for all x in [left, right-1], inv[x] < left or inv[x] > right-1.
        # For [left, right] to be valid, we need for all x in [left, right], inv[x] < left or inv[x] > right.
        # For x in [left, right-1]: we had inv[x] < left or inv[x] > right-1.
        #   If inv[x] < left, still fine.
        #   If inv[x] > right-1, we need inv[x] > right. So if inv[x] == right, it's a problem.
        #   inv[x] == right means x == inv[right].
        # For x = right: we need inv[right] < left or inv[right] > right.
        
        # So [left, right] is valid iff:
        #   1. [left, right-1] is valid
        #   2. inv[right] < left OR inv[right] > right
        #   3. inv[right] != right (if inv[right] == right, condition 2 fails)
        #   4. No x in [left, right-1] has inv[x] == right, i.e., inv[right] not in [left, right-1]
        #      But this is implied by condition 2 when inv[right] != right.
        
        # So the check is: is inv[right] < left or inv[right] > right?
        
        # If not, we need to move left forward until the condition is satisfied.
        # But moving left forward might break validity for other reasons? No, because 
        # removing elements from the left only makes it easier to satisfy the condition.
        
        # Actually, if inv[right] is in [left, right], we need to move left past inv[right].
        # So left = max(left, inv[right] + 1).
        
        # But we also need to ensure that after moving left, the window is still valid for all pairs.
        # Since we're only adding right and the rest was valid, and we move left to exclude inv[right],
        # the window should be valid.
        
        # Let me verify: if inv[right] is in [left, right], we set left = inv[right] + 1.
        # Now for all x in [left, right-1], inv[x] was either < old_left <= new_left or > right-1.
        # If inv[x] > right-1 and inv[x] <= right, then inv[x] = right, so x = inv[right].
        # But inv[right] < new_left = inv[right] + 1, so x = inv[right] is not in [new_left, right-1]. Good.
        # For x = right: inv[right] < new_left = inv[right] + 1. Good.
        
        # So the algorithm is:
        left = 1
        best_start = 1
        best_length = 0
        
        for right in range(1, n + 1):
            # If right is a self-inverse, it can't be in any valid interval
            if inv[right] == right:
                # Skip this right, and move left past it
                left = right + 1
                continue
            
            # Check if inv[right] is in [left, right]
            if inv[right] >= left and inv[right] <= right:
                # Need to move left past inv[right]
                left = inv[right] + 1
            
            # Now [left, right] should be valid
            length = right - left + 1
            if length > best_length:
                best_length = length
                best_start = left
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
