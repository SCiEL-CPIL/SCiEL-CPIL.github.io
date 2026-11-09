import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # Base set: {65, ..., 128}
    # Size 64.
    # Any a, b in this set have a+b >= 130 > 128.
    # So no sum of two base elements is in the set.
    # Thus, no forbidden triple can consist of two base elements.
    base_set = set(range(65, n + 1))
    candidates = list(range(1, 65))
    
    best_size = len(base_set)
    best_set = base_set.copy()
    
    def can_add(S, x):
        """
        Check if adding x to S keeps S valid.
        S is assumed to be valid.
        x is not in S.
        Forbidden triple {a, b, x} requires:
        a+b in S, a+x in S, b+x in S.
        """
        s_list = list(S)
        m = len(s_list)
        for i in range(m):
            a = s_list[i]
            # If a + x > n, then a+x not in S (since max element is n)
            # We can skip checking sum_ax if it's out of bounds, but we still need sum_ab
            # Actually, if a+x > n, the condition "a+x in S" is false, so this pair (a,b) is safe regarding x.
            # Wait, the condition is (a+b in S) AND (a+x in S) AND (b+x in S).
            # If a+x > n, then a+x not in S. So the triple is NOT forbidden.
            # So we can skip this pair entirely if a+x > n OR b+x > n?
            # No, if a+x > n, then a+x not in S. So the conjunction is false.
            # So we can break early or continue.
            # However, we must check if a+b in S. If a+b not in S, also safe.
            
            # Optimization:
            # If a + x > n, then a+x is not in S. So this a cannot form a forbidden triple with x and any b.
            # So we can skip all b for this a?
            # Yes.
            if a + x > n:
                continue
                
            for j in range(i + 1, m):
                b = s_list[j]
                if b + x > n:
                    # If b+x > n, then b+x not in S. Safe.
                    # Since s_list is not sorted, we can't break.
                    continue
                
                sum_ab = a + b
                if sum_ab > n:
                    continue
                    
                if sum_ab in S:
                    sum_ax = a + x
                    # We know sum_ax <= n
                    if sum_ax in S:
                        sum_bx = b + x
                        # We know sum_bx <= n
                        if sum_bx in S:
                            return False
        return True

    def try_construction(order):
        current_set = base_set.copy()
        for x in order:
            if x in current_set:
                continue
            if can_add(current_set, x):
                current_set.add(x)
        return current_set

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Different strategies for ordering candidates
        if attempts % 3 == 1:
            c = candidates.copy()
            random.shuffle(c)
        elif attempts % 3 == 2:
            c = sorted(candidates)
        else:
            c = sorted(candidates, reverse=True)
            
        current_set = try_construction(c)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
        # If we hit a high score, we might want to try more random shuffles
        # to see if we can beat it.
        
    return list(best_set)
