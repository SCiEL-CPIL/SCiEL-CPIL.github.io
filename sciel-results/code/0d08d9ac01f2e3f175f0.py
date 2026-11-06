import time
import random
import sys

def _valid(S):
    """Check the property for sorted set S. Returns True if valid."""
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i + 1, n):
            b = lst[j]
            for k in range(j, n):
                c = lst[k]
                if (b + c) % a == 0:
                    return False
    return True

def _count_conflicts(a, S):
    """Count how many pairs (b,c) in S with b<=c, b>a satisfy a|(b+c)."""
    count = 0
    S_list = sorted(S)
    # Optimization: use two pointers or set lookup
    S_set = S
    # b+c = k*a
    max_sum = max(S_list) + max(S_list)
    for k in range(2, max_sum // a + 1):
        target = k * a
        # We need b, c in S such that b + c = target, b <= c, b > a
        # b ranges from max(a+1, target - max(S_list)) to target // 2
        b_min = max(a + 1, target - max(S_list))
        b_max = target // 2
        for b in range(b_min, b_max + 1):
            c = target - b
            if c in S_set and c >= b:
                count += 1
    return count

def _remove_min_conflict(S):
    """Remove the element from S that is part of the most conflicts."""
    if not S:
        return S
    
    S_list = sorted(S)
    max_sum = max(S_list) + max(S_list)
    
    # We need to find which element a is in the most conflicts.
    # A conflict is (a, b, c) with a < b <= c and a | (b+c).
    # We can iterate over all valid triples? No, too slow.
    # Instead, for each a in S, count conflicts where a is the divisor.
    # This is O(|S| * |S|^2) which is 256^3 ~ 16M, might be too slow in Python if done repeatedly.
    
    # Faster: Precompute for each a, the conflicts.
    # Let's just pick a random conflict and remove the element with highest degree in that conflict?
    # Or simpler: Just remove the smallest a that causes a conflict?
    # The smallest a is the most restrictive divisor. Removing it is safe.
    
    # Let's find one conflict and remove the divisor a.
    S_set = S
    lst = sorted(S)
    for i in range(len(lst)):
        a = lst[i]
        found = False
        for j in range(i + 1, len(lst)):
            b = lst[j]
            for k in range(j, len(lst)):
                c = lst[k]
                if (b + c) % a == 0:
                    # Conflict found
                    S.remove(a)
                    return S
    return S

def construct():
    start = time.time()
    deadline = start + 80.0  # Safety margin for 90s limit
    n = 256

    # Strategy: Start with upper half, fix conflicts, then add lower elements.
    
    # 1. Start with S = {129, ..., 256}
    S = set(range(129, n + 1))
    
    # 2. Make it valid by removing conflicting divisors
    # We repeatedly find a conflict and remove the divisor 'a' (the smallest element in the triple).
    # This ensures we keep the larger elements which are less likely to be divisors.
    # Is this optimal? Removing 'a' is good because 'a' divides many sums.
    # Removing 'b' or 'c' might be better if 'b' is involved in many conflicts as a summand.
    # But 'a' is the root cause of the divisibility.
    
    # To speed up conflict finding, we can cache or use a faster check.
    # For now, O(N^3) check in Python might be slow if many removals are needed.
    # 128 elements. 128^3 / 6 ~ 350k operations per check.
    # If we do 100 removals, 35M ops. Might be tight.
    
    # Let's use a slightly smarter removal:
    # Find ALL conflicts for the current smallest element. If it has any, remove it.
    # Else, check next smallest.
    
    while time.time() < deadline - 10.0:
        lst = sorted(S)
        removed_any = False
        for a in lst:
            if a not in S: continue
            has_conflict = False
            # Check if a divides any b+c with b,c in S, b>a
            S_set = S
            # b+c must be multiple of a.
            # Max sum is 256+256=512.
            # Multiples: 2a, 3a, ...
            for k in range(2, 512 // a + 1):
                target = k * a
                # Check if target can be formed by two elements in S greater than a
                # b <= c, b > a
                # b_min = max(a+1, target - 256)
                # b_max = target // 2
                b_min = max(a + 1, target - 256)
                b_max = target // 2
                if b_min > b_max:
                    continue
                for b in range(b_min, b_max + 1):
                    c = target - b
                    if c in S_set and c >= b:
                        has_conflict = True
                        break
                if has_conflict:
                    break
            if has_conflict:
                S.remove(a)
                removed_any = True
                break # Restart check from smallest again
        if not removed_any:
            break # S is valid
    
    if not _valid(S):
        # Fallback if loop exited due to time but not valid
        S = set(range(171, n + 1))
    
    # 3. Try to add elements from 1 to 128
    # Add in descending order (128 down to 1) because larger numbers have fewer multiples.
    candidates = list(range(128, 0, -1))
    
    for a in candidates:
        if time.time() > deadline - 5.0:
            break
        if a in S:
            continue
        # Check if a can be added
        can_add = True
        S_set = S
        for k in range(2, 512 // a + 1):
            target = k * a
            b_min = max(a + 1, target - 256)
            b_max = target // 2
            if b_min > b_max:
                continue
            for b in range(b_min, b_max + 1):
                c = target - b
                if c in S_set and c >= b:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            S.add(a)
            
    # 4. Local Search / Swap to improve
    # Try to swap an element in S with one not in S to increase size? 
    # Since we are maximizing size, we only add.
    # But maybe removing a small element allows adding two larger ones?
    # Given the constraints, simple addition is the main gain.
    
    # Final check
    if not _valid(S):
        S = set(range(171, n + 1))
        
    return list(S)

if __name__ == "__main__":
    print(construct())
