import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    all_numbers = set(range(1, N + 1))
    
    def get_score(A):
        return sum(1.0 / x for x in A)

    def is_3ap_free_with(A_set, x, exclude=None):
        """
        Checks if adding x to A_set (with exclude removed) results in a 3-AP-free set.
        Assumes A_set \ {exclude} is already 3-AP-free.
        """
        # We only need to check 3-APs involving x.
        # A 3-AP is (a, b, c) with 2b = a + c.
        # If x is in the AP, it can be a, b, or c.
        
        # Case 1: x is the middle term (b=x).
        # Then a + c = 2x.
        # We need to check if there exist a, c in A_set \ {exclude} such that a + c = 2x and a != c.
        # Iterate a in A_set \ {exclude}:
        #   c = 2x - a
        #   if c in A_set and c != a and c != exclude: return False
        
        # Case 2: x is an end term (a=x or c=x).
        # If a=x, then 2b = x + c => c = 2b - x.
        # We need to check if there exist b, c in A_set \ {exclude} such that c = 2b - x and b != c.
        # Iterate b in A_set \ {exclude}:
        #   c = 2b - x
        #   if c > 0 and c in A_set and c != b and c != exclude: return False
        
        # Note: If we check all b in A_set \ {exclude}, we cover both "x is smallest" and "x is largest"
        # because if x is largest, b is middle, c is smallest. c = 2b - x.
        # If x is smallest, b is middle, c is largest. c = 2b - x.
        # The formula is the same.
        
        # We can combine the loops?
        # For efficiency, we just iterate once.
        
        for b in A_set:
            if b == exclude:
                continue
            
            # Check if b and some other element form a 3-AP with x
            # 1. x is middle: a + c = 2x. Let one element be b.
            #    If b is 'a', then c = 2x - b.
            #    If b is 'c', then a = 2x - b.
            #    So check if (2x - b) is in A_set.
            v_mid = 2 * x - b
            if v_mid > 0 and v_mid in A_set and v_mid != b and v_mid != exclude:
                return False
                
            # 2. x is end: 2b' = a + c. Let one element be b.
            #    If b is 'b' (middle), then a = 2b - x and c = 2b - x? No.
            #    If b is the middle term of the AP (x, b, c) or (c, b, x).
            #    Then c = 2b - x.
            #    Check if (2b - x) is in A_set.
            v_end = 2 * b - x
            if v_end > 0 and v_end in A_set and v_end != b and v_end != exclude:
                return False
                
            # 3. What if b is the other end?
            #    If AP is (x, b, c), b is middle. Covered above.
            #    If AP is (a, x, c), x is middle. Covered above.
            #    If AP is (a, b, x), x is end. b is middle. Covered above.
            #    If AP is (x, a, b), x is end. a is middle. 
            #       Here b is the other end.
            #       2a = x + b => a = (x+b)/2.
            #       This requires x+b to be even.
            #       Is this covered?
            #       In case 2, we iterated over `b` as the *middle* element.
            #       Here `b` is an *end* element.
            #       So we missed the case where the existing element `b` is an end, and `x` is the other end.
            #       Let's re-evaluate.
            
            # The 3-AP involves x and two elements u, v from A_set.
            # Possible roles for x:
            # 1. x is middle: u + v = 2x.
            #    For each u in A_set, v = 2x - u. Check if v in A_set.
            # 2. x is end: u + x = 2v (v is middle) => v = (u+x)/2.
            #    For each u in A_set, if (u+x) is even, v = (u+x)//2. Check if v in A_set.
            # 3. x is end: v + x = 2u (u is middle) => u = (v+x)/2.
            #    This is the same as 2, just swapping u and v.
            
            # So the complete check is:
            # Iterate u in A_set \ {exclude}:
            #   1. v1 = 2*x - u. If v1 in A_set and v1 != u and v1 != exclude: conflict.
            #   2. s = u + x. If s % 2 == 0:
            #        v2 = s // 2. If v2 in A_set and v2 != u and v2 != x and v2 != exclude: conflict.
            #      Note: v2 != x is always true if u != x (since x not in A_set).
            #      v2 != u implies u + x = 2u => x = u, which is false.
            #      So just check v2 in A_set and v2 != exclude.
            
            # Let's rewrite the function with this logic.
            pass # Placeholder for logic update in final code
            
        return True

    # Corrected Logic for is_3ap_free_with
    def check_valid_addition(A_set, x, exclude=None):
        for u in A_set:
            if u == exclude:
                continue
            # Case 1: x is middle. u is one end. Other end is 2x - u.
            v1 = 2 * x - u
            if v1 > 0 and v1 in A_set and v1 != u and v1 != exclude:
                return False
            # Case 2: x is end. u is one end. Middle is (u+x)/2.
            s = u + x
            if s % 2 == 0:
                v2 = s // 2
                if v2 in A_set and v2 != u and v2 != exclude:
                    return False
        return True

    def greedy_init():
        A = set()
        for x in range(1, N + 1):
            if check_valid_addition(A, x, exclude=None):
                A.add(x)
        return A

    def local_search(A):
        A = set(A)
        score = get_score(A)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swaps: remove x, add y. y < x to improve score.
            # Iterate x in A
            for x in list(A):
                if time.time() > deadline:
                    break
                # Find candidate y
                # To speed up, we can just check all y < x
                for y in range(1, x):
                    if y in A:
                        continue
                    # Check if swapping x for y is valid
                    # Remove x, add y
                    if check_valid_addition(A, y, exclude=x):
                        # Perform swap
                        A.remove(x)
                        A.add(y)
                        score = get_score(A)
                        improved = True
                        break # Restart scan
                if improved:
                    break
        return A, score

    # Main Logic
    best_A = greedy_init()
    best_score = get_score(best_A)
    
    # Run local search on initial
    best_A, best_score = local_search(best_A)
    
    # Random Restarts
    while time.time() < deadline - 5.0:
        # Perturb best_A
        current_A = set(best_A)
        # Remove 1-3 random elements
        for _ in range(random.randint(1, 3)):
            if current_A:
                to_remove = random.choice(list(current_A))
                current_A.remove(to_remove)
        
        # Run local search
        current_A, current_score = local_search(current_A)
        
        if current_score > best_score:
            best_score = current_score
            best_A = current_A

    return list(best_A)
