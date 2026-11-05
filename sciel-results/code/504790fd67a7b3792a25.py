import time
import random

def construct():
    n = 256
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Base set: [171, 256]
    # This set is valid because for any a < b <= c in S, 
    # b+c <= 512 and a >= 171. 
    # Multiples of a: 2a >= 342, 3a >= 513.
    # So b+c can only be 2a. 
    # We need to ensure no b+c = 2a for a < b <= c in S.
    # For S = [171, 256], min b=172, min c=172 -> sum=344.
    # Max a=171 -> 2a=342.
    # If a=171, b,c > 171. Min sum 172+172=344. 344 != 342.
    # Next multiple 3*171=513 > 512.
    # So [171, 256] is valid.
    
    base_S = list(range(171, n + 1))
    best_S = set(base_S)
    best_size = len(base_S)
    
    def can_add(x, S, sums):
        """
        Check if x can be added to S.
        S is a set of integers.
        sums is the set of all b+c for b,c in S with b <= c.
        """
        # Case 1: x is the new minimum (x < min(S))
        # Then x is 'a'. We need x not to divide any b+c in sums.
        # Note: sums contains sums of pairs from S.
        # If x is new min, all b,c in S are > x.
        # So we just check if any multiple of x is in sums.
        min_S = min(S)
        if x < min_S:
            # Check multiples of x in sums
            # Min sum in sums is 2*min_S. Max is 2*max(S).
            # We can iterate through sums and check mod x? 
            # Or iterate multiples of x?
            # sums size is ~ |S|^2 / 2. |S| ~ 100. sums ~ 5000.
            # Iterating sums is O(|sums|).
            for s in sums:
                if s % x == 0:
                    return False
            return True
        
        # Case 2: x is inserted into the middle or is the max
        # This is more complex. We need to check triples involving x.
        # a < b <= c
        # x can be a, b, or c.
        
        # If x is in S, it's not added. We assume x not in S.
        
        # Let's just do a full check for the new element x against existing S.
        # New triples:
        # 1. x is a: a=x. Need x < b <= c. b,c in S.
        #    Check if x divides b+c for any b,c in S with x < b <= c.
        # 2. x is b: b=x. Need a < x <= c. a,c in S.
        #    Check if a divides x+c for any a,c in S with a < x <= c.
        # 3. x is c: c=x. Need a < b <= x. a,b in S.
        #    Check if a divides b+x for any a,b in S with a < b <= x.
        # 4. x is b=c: b=c=x. Need a < x. a in S.
        #    Check if a divides 2x for any a in S with a < x.
        
        # Optimization: Pre-filter S into lists based on value relative to x.
        # S_less = [y in S if y < x]
        # S_greater = [y in S if y > x]
        # S_equal = [] (x not in S)
        
        S_list = list(S)
        
        # 1. x as a
        # b, c in S, x < b <= c
        # This implies b, c > x.
        # Check if x | (b+c)
        candidates_b = [y for y in S_list if y > x]
        for i in range(len(candidates_b)):
            b = candidates_b[i]
            for j in range(i, len(candidates_b)):
                c = candidates_b[j]
                if (b + c) % x == 0:
                    return False
        
        # 2. x as b
        # a, c in S. a < x <= c.
        # Check if a | (x+c)
        candidates_a = [y for y in S_list if y < x]
        candidates_c = [y for y in S_list if y >= x] # c >= x
        for a in candidates_a:
            for c in candidates_c:
                if (x + c) % a == 0:
                    return False
        
        # 3. x as c
        # a, b in S. a < b <= x.
        # Check if a | (b+x)
        # b <= x. Since x not in S, b < x.
        candidates_b2 = [y for y in S_list if y < x] # b < x
        # a < b
        for i in range(len(candidates_b2)):
            b = candidates_b2[i]
            # a must be < b
            # candidates_a is sorted? No.
            # Let's just iterate a in S_list where a < b
            for a in S_list:
                if a < b:
                    if (b + x) % a == 0:
                        return False
                    # Also check if a < x is required? 
                    # Condition is a < b <= c=x. So a < b.
                    # b < x. So a < x is implied if a < b.
        
        # 4. x as b=c
        # a in S, a < x.
        # Check if a | (2x)
        for a in candidates_a:
            if (2 * x) % a == 0:
                return False
        
        return True

    def update_sums(sums, S, x):
        """
        Update sums set after adding x to S.
        New sums are x+x and x+y for y in S.
        """
        sums.add(2 * x)
        for y in S:
            sums.add(x + y)
        return sums

    # Greedy Descending (Fast Path)
    # This is the most likely to work well and is very fast.
    def greedy_descending():
        S = set(base_S)
        # Compute initial sums
        sums = set()
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        
        # Try adding from 170 down to 1
        for x in range(170, 0, -1):
            if can_add(x, S, sums):
                S.add(x)
                sums = update_sums(sums, S, x)
        return S

    # Run greedy descending
    S_greedy = greedy_descending()
    if len(S_greedy) > best_size:
        best_S = S_greedy
        best_size = len(S_greedy)

    # Random Restarts with General Check
    # We can try to add elements in random order to see if we get a larger set.
    # This is slower, so we limit iterations.
    
    num_restarts = 0
    while time.time() < deadline - 5:
        candidates = list(range(1, 171))
        random.shuffle(candidates)
        
        S = set(base_S)
        # Compute initial sums
        sums = set()
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        
        for x in candidates:
            if can_add(x, S, sums):
                S.add(x)
                sums = update_sums(sums, S, x)
        
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            # If we find a significantly better set, we might want to break or restart
            # But let's keep searching.
        
        num_restarts += 1
        # Optional: If we've run many restarts without improvement, we could break
        # But the check is fast enough.

    return sorted(best_S)
