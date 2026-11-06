import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 256
    
    def is_valid_with(S, x):
        """
        Check if x can be added to set S.
        S is a set of ints.
        """
        # Case 1: x is a
        # x < b <= c in S
        # We need to check if x divides b+c for any b, c in S with x < b <= c
        # Since we don't know if S is sorted, we iterate all pairs.
        # Optimization: If S is small, this is fine.
        # To avoid O(N^2) overhead of list creation every time, we can just iterate the set.
        
        s_list = list(S)
        s_len = len(s_list)
        
        # Case 1: x is a
        # x < b <= c
        for i in range(s_len):
            b = s_list[i]
            if b <= x:
                continue
            # b > x
            for j in range(s_len):
                c = s_list[j]
                if c < b:
                    continue
                # b <= c
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is b
        # a < x <= c in S
        for i in range(s_len):
            a = s_list[i]
            if a >= x:
                continue
            # a < x
            for j in range(s_len):
                c = s_list[j]
                if c < x:
                    continue
                # c >= x
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is c
        # a < b <= x in S
        for i in range(s_len):
            a = s_list[i]
            if a >= x:
                continue
            # a < x
            for j in range(s_len):
                b = s_list[j]
                if b <= a or b > x:
                    continue
                # a < b <= x
                if (b + x) % a == 0:
                    return False
        
        return True

    def polish(S):
        """
        Try to add any element from [1, n] not in S to S if valid.
        Repeat until no more can be added.
        """
        changed = True
        while changed:
            changed = False
            # Iterate in a deterministic order for consistency, or random?
            # Deterministic is faster to check, but might miss opportunities if order matters?
            # No, if it's valid, it's valid. Order doesn't matter for final set if we add all possible.
            for x in range(1, n + 1):
                if x not in S:
                    if is_valid_with(S, x):
                        S.add(x)
                        changed = True
                        if time.time() > deadline:
                            return S
        return S

    def greedy(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_with(S, x):
                S.add(x)
        S = polish(S)
        return S

    best_S = set()
    
    # 1. Descending order (usually best because large numbers are less restrictive as 'a')
    order_desc = list(range(n, 0, -1))
    S_desc = greedy(order_desc)
    if len(S_desc) > len(best_S):
        best_S = set(S_desc)
        
    # 2. Ascending order
    order_asc = list(range(1, n + 1))
    S_asc = greedy(order_asc)
    if len(S_asc) > len(best_S):
        best_S = set(S_asc)
        
    # 3. Random orders
    # Use a few random restarts
    max_restarts = 20
    for i in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        random.seed(i)
        order_rand = list(range(1, n + 1))
        random.shuffle(order_rand)
        S_rand = greedy(order_rand)
        if len(S_rand) > len(best_S):
            best_S = set(S_rand)
            
    # 4. Local Search / Hill Climbing
    # Try to improve best_S by swapping elements
    
    # We will try to remove one element and add another (or more via polish)
    # To keep it efficient, we'll limit the number of swap attempts.
    
    S_current = set(best_S)
    current_size = len(S_current)
    
    # Pre-compute all candidates not in S
    # This might change as we swap, so we do it inside the loop or update it.
    
    swap_attempts = 0
    max_swap_attempts = 1000
    
    while time.time() < deadline - 5.0 and swap_attempts < max_swap_attempts:
        swap_attempts += 1
        improved = False
        
        # Try removing each element in S_current
        # We iterate over a copy because we might modify S_current
        elements_to_try = list(S_current)
        # Randomize the order of removal attempts to avoid bias
        random.shuffle(elements_to_try)
        
        for x_remove in elements_to_try:
            if time.time() > deadline - 5.0:
                break
                
            S_temp = set(S_current)
            S_temp.discard(x_remove)
            
            # Try adding each element not in S_temp
            # We can limit this to a random subset of candidates if it's too slow
            candidates = [y for y in range(1, n + 1) if y not in S_temp]
            random.shuffle(candidates)
            
            # Try a limited number of candidates to keep speed up
            for y_add in candidates[:20]: # Try up to 20 random candidates
                if is_valid_with(S_temp, y_add):
                    S_new = set(S_temp)
                    S_new.add(y_add)
                    
                    # Polish the new set
                    S_new = polish(S_new)
                    
                    if len(S_new) > current_size:
                        S_current = S_new
                        current_size = len(S_new)
                        best_S = set(S_new)
                        improved = True
                        break # Restart the outer loop with the new best set
                    
                    # If size is same, we could still accept if it leads to better future?
                    # For now, only accept if size increases.
            
            if improved:
                break # Break out of elements_to_try loop to restart with new S_current
                
    return sorted(best_S)

# Test run
if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
    print(res)
