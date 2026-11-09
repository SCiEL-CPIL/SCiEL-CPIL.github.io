import time
import random

def is_valid_set(S):
    """
    Check if the set S is valid.
    Condition: No element a in S is the mean of any subset T of S\{a} with |T| >= 2.
    """
    n = len(S)
    if n < 3:
        return True
    
    S_list = list(S)
    S_set = set(S_list)
    
    # 1. Check 3-term Arithmetic Progressions (k=2)
    # a = (x+y)/2 => 2a = x+y
    for a in S_list:
        target = 2 * a
        for x in S_list:
            if x == a:
                continue
            y = target - x
            if y in S_set and y != a:
                return False
                
    # 2. Check k >= 3 using Bitset DP
    for a in S_list:
        others = [x for x in S_list if x != a]
        n_others = len(others)
        
        if n_others < 3:
            continue
            
        dp = [0] * (n_others + 1)
        dp[0] = 1
        
        for x in others:
            for k in range(n_others - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << x)
        
        for k in range(3, n_others + 1):
            target_sum = k * a
            if (dp[k] >> target_sum) & 1:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def greedy_construct(order):
        selected = []
        for x in order:
            if len(selected) >= 48:
                break
            if is_valid_set(selected + [x]):
                selected.append(x)
        return selected

    best_set = greedy_construct(list(range(1, 513)))
    
    if len(best_set) == 48:
        return best_set
        
    S = set(best_set)
    candidates = [x for x in range(1, 513) if x not in S]
    
    random.seed(42)
    
    while time.time() < deadline:
        if len(S) == 48:
            break
            
        added = False
        # Try adding
        for _ in range(20):
            if not candidates:
                break
            x = random.choice(candidates)
            if is_valid_set(list(S) + [x]):
                S.add(x)
                candidates.remove(x)
                added = True
                break
                
        if added:
            continue
            
        # Try swapping
        swapped = False
        for _ in range(20):
            if not candidates:
                break
            a_to_remove = random.choice(list(S))
            x_to_add = random.choice(candidates)
            
            new_S = S - {a_to_remove} | {x_to_add}
            if is_valid_set(list(new_S)):
                S = new_S
                if a_to_remove not in candidates:
                    candidates.append(a_to_remove)
                if x_to_add in candidates:
                    candidates.remove(x_to_add)
                swapped = True
                break
                
        if not added and not swapped:
            # Optional: If stuck, try a random removal to break symmetry?
            # Or just continue.
            pass

    return list(S)

if __name__ == "__main__":
    s = construct()
    print(len(s))
