import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    def check_valid(S, a):
        """Check if a can be added to S (a must be less than all elements in S for the constraint to apply as 'a' role)"""
        if a in S:
            return False
        # a will be the smallest, so it's 'a' in the triple
        # Need: for all b, c in S with b <= c, a does NOT divide (b+c)
        m = len(S)
        for i in range(m):
            b = S[i]
            for j in range(i, m):
                c = S[j]
                if (b + c) % a == 0:
                    return False
        return True
    
    def check_valid_general(S, a):
        """Check if a can be in S, considering a could be in any role"""
        if a in S:
            return False
        S_new = sorted(S + [a])
        m = len(S_new)
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j, m):
                    a_val = S_new[i]
                    b_val = S_new[j]
                    c_val = S_new[k]
                    if a_val < b_val and b_val <= c_val:
                        if (b_val + c_val) % a_val == 0:
                            return False
        return True
    
    # Start with the safe range: [171, 256]
    # For any a, b, c in this range with a < b <= c:
    # b+c >= 171+171 = 342, and a <= 256
    # But we need a ∤ (b+c). This isn't automatically satisfied.
    # Actually, the condition 3*min > 2*max ensures sum-free, but that's different.
    
    # Let me reconsider. The current best is 86 from [171, 256].
    # Let's start there and try to add more elements.
    
    S = list(range(171, 257))
    
    # Try to add elements from 1 to 170
    # Sort by "conflict score" - try elements that might have fewer conflicts
    candidates = list(range(1, 171))
    
    # Sort candidates: try larger ones first (they have fewer divisors to worry about)
    # Actually, smaller a divides more numbers, so larger a are "safer"
    candidates.sort(reverse=True)
    
    for a in candidates:
        if time.time() > deadline - 5.0:
            break
        if check_valid(S, a):
            S.append(a)
    
    S.sort()
    
    # Try local search: try to swap elements to improve
    # If we remove an element and add two others, we gain 1
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        S_sorted = sorted(S)
        for i in range(len(S_sorted)):
            if time.time() > deadline - 10.0:
                break
            removed = S_sorted[i]
            S_temp = S_sorted[:i] + S_sorted[i+1:]
            # Try to add two elements that were previously excluded
            for a in range(1, 171):
                if time.time() > deadline - 10.0:
                    break
                if a in S_temp:
                    continue
                if check_valid(S_temp, a):
                    S_temp_new = S_temp + [a]
                    # Try to add one more
                    for b in range(a+1, 171):
                        if time.time() > deadline - 10.0:
                            break
                        if b in S_temp_new:
                            continue
                        if check_valid(S_temp_new, b):
                            if len(S_temp_new + [b]) > len(S):
                                S = sorted(S_temp_new + [b])
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return S
