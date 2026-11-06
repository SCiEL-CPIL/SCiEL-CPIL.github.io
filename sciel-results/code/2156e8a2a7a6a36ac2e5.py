"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a set of distinct positive integers <= 512 such that no element
    is the arithmetic mean of any subset of at least two other elements.
    
    Strategy:
    1. Generate a large candidate set using a greedy algorithm that avoids
       3-term arithmetic progressions (3-APs). This ensures no element is
       the mean of any 2-element subset.
    2. Filter this candidate set by removing any element that is the mean
       of a larger subset (size >= 3) of the remaining elements.
    """
    max_val = 512
    
    # Step 1: Greedy 3-AP free set
    S = []
    for i in range(1, max_val + 1):
        valid = True
        # Check if adding i creates a 3-AP with any pair in S
        # A 3-AP (a, b, c) means b = (a+c)/2.
        # Cases:
        # 1. i is the middle term: 2*i = a + c for some a, c in S
        # 2. i is the last term: 2*b = a + i for some a, b in S
        # 3. i is the first term: 2*b = i + c for some b, c in S
        
        # Optimization: Store S in a set for O(1) lookups
        S_set = set(S)
        
        for a in S:
            # Case 1: i is middle. a + c = 2*i => c = 2*i - a
            c = 2 * i - a
            if c in S_set and c != i: # c must be distinct from a? Yes, distinct elements.
                # If c == a, then 2*i = 2*a => i=a, but i is new.
                # If c is in S, we have a 3-AP.
                valid = False
                break
            if not valid:
                break
            
            # Case 2: i is last. a + i = 2*b => b = (a + i) / 2
            # (a+i) must be even
            if (a + i) % 2 == 0:
                b = (a + i) // 2
                if b in S_set and b != i and b != a:
                    valid = False
                    break
            if not valid:
                break
            
            # Case 3: i is first. i + c = 2*b => c = 2*b - i
            # We need to check if there exists b in S such that c = 2*b - i is in S.
            # This is harder to check directly without iterating b.
            # But we can just iterate b in S for this case?
            # Or we can just check all pairs (a,c) in S to see if their mean is i?
            # That's Case 1.
            # Check all pairs (a,b) in S to see if their mean is i? No, that's Case 1.
            
            # Let's restart the check logic to be cleaner.
            # Check if i is mean of any pair (a,c) in S.
            # Check if any a in S is mean of (i, c) for c in S.
            # Check if any c in S is mean of (a, i) for a in S.
        
        # Re-implementing the check more simply:
        valid = True
        S_set = set(S)
        
        # 1. Is i the mean of any pair in S?
        for a in S:
            c = 2 * i - a
            if c in S_set and c != a:
                valid = False
                break
        
        if valid:
            # 2. Is any a in S the mean of (i, c) for some c in S?
            # 2*a = i + c => c = 2*a - i
            for a in S:
                c = 2 * a - i
                if c in S_set and c != a and c != i:
                    valid = False
                    break
        
        if valid:
            # 3. Is any c in S the mean of (a, i) for some a in S?
            # 2*c = a + i => a = 2*c - i
            # This is symmetric to case 2, just swapping roles of a and c.
            # Since we iterate all a in S in case 2, and check if c is in S,
            # this covers all pairs.
            # Wait.
            # Case 2: a is mean. Elements are i and c.
            # Case 3: c is mean. Elements are a and i.
            # In Case 2 loop: for each a in S, we check if c = 2a - i is in S.
            # This checks if there is a pair (i, c) with mean a.
            # In Case 3, we want to check if there is a pair (a, i) with mean c.
            # This is equivalent to: for each c in S, is there an a in S such that a = 2c - i?
            # Since we iterate over all elements in S as 'a' in Case 2, and 'c' is just another element in S,
            # the set of pairs checked is the same.
            # So Case 2 and 3 are redundant if we check all elements in S as the "mean".
            
            S.append(i)
            
    # Step 2: Filter for larger subset means
    # We have a 3-AP free set S.
    # Now we need to ensure no element is the mean of a subset of size >= 3.
    # We will iterate through elements and remove those that are means.
    # To maximize size, we might want to remove "critical" elements first?
    # Or just remove any that are invalid.
    # Since removing an element can only help validity of others, a single pass
    # in any order will result in a valid set.
    # However, the size might depend on the order.
    # Let's try removing elements in descending order of value?
    # Or just keep the ones that are not means.
    
    S_final = list(S)
    changed = True
    while changed:
        changed = False
        S_set = set(S_final)
        for i in range(len(S_final)):
            y = S_final[i]
            others = [x for x in S_final if x != y]
            # Check if y is mean of any subset of others of size >= 2
            # Use DP for subset sums
            # Max sum of others is roughly 50 * 512 = 25600
            # We need to check if sum = k * y for some k >= 2
            # and if that sum can be formed by a subset of size k.
            
            # Optimization: Only check if y * 2 <= sum(others) ?
            # Min sum of 2 others is roughly 1+2=3.
            # Max k is len(others).
            
            # DP table: dp[k][s]
            # To save memory/time, we can just check if y is a mean.
            # Given the small size of S (likely < 100), we can do a simple check.
            
            # Actually, let's just use a helper function.
            if is_mean(y, others):
                S_final.pop(i)
                changed = True
                break # Restart loop with new set
        
    return S_final

def is_mean(y, others):
    """Check if y is the mean of any subset of others of size >= 2."""
    if len(others) < 2:
        return False
    
    # DP: dp[k][s] = True if there is a subset of size k with sum s
    # Max sum
    max_sum = sum(others)
    # We only care about sums that are multiples of y
    # k * y <= max_sum => k <= max_sum / y
    max_k = min(len(others), max_sum // y)
    
    if max_k < 2:
        return False
    
    # Initialize DP
    # dp[k] is a set of sums? Or a bitset?
    # Using a list of sets might be faster for sparse sums.
    # Or a list of bitsets (integers).
    # Let's use a list of sets of sums for simplicity, given small N.
    # dp[k] = set of possible sums for subsets of size k
    
    dp = [set() for _ in range(max_k + 1)]
    dp[0].add(0)
    
    for x in others:
        # Update backwards
        for k in range(max_k, 0, -1):
            if not dp[k-1]:
                continue
            new_sums = set()
            for s in dp[k-1]:
                ns = s + x
                if ns <= max_sum:
                    new_sums.add(ns)
            dp[k] = dp[k] | new_sums
    
    # Check if any k >= 2 has sum k * y
    for k in range(2, max_k + 1):
        target = k * y
        if target in dp[k]:
            return True
            
    return False

# EVOLVE-BLOCK-END
