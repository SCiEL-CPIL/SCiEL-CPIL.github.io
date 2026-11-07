import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_set = list(range(65, 129)) # Baseline 64
    best_size = len(best_set)
    
    # Odds are 1, 3, ..., 127
    # Try removing k smallest odds
    for k in range(0, 30):
        if time.time() > deadline:
            break
            
        # O_k contains odds from index k to 63 (0-indexed)
        # o_i = 2*i + 1
        # Min sum is o_k + o_{k+1} = (2k+1) + (2k+3) = 4k+4
        min_sum = 4*k + 4
        
        # Candidate evens: 2, 4, ..., min_sum-2
        # If min_sum-2 < 2, E is empty
        max_even = min_sum - 2
        if max_even < 2:
            E_cand = []
        else:
            E_cand = list(range(2, max_even + 1, 2))
            
        # Find a large sum-free subset of E_cand
        # Greedy: sort descending, pick if sum with any picked is not in picked
        # Actually, simpler: just pick the largest half?
        # Let's try a few strategies or just the largest half for speed
        # Largest half of E_cand
        if E_cand:
            mid = len(E_cand) // 2
            E = E_cand[mid:] # Upper half
        else:
            E = []
            
        # Verify E is sum-free (distinct elements)
        # For small sizes, we can check
        valid_E = True
        E_set = set(E)
        for i in range(len(E)):
            for j in range(i+1, len(E)):
                if E[i] + E[j] in E_set:
                    valid_E = False
                    break
            if not valid_E:
                break
        
        if not valid_E:
            # Fallback to largest single element or empty
            # Or just take the last element
            E = E_cand[-1:] if E_cand else []
            
        # Construct S
        O_k = [2*i + 1 for i in range(k, 64)]
        S = O_k + E
        
        # Verify S is valid
        if is_valid(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
        
    return best_set

def is_valid(S):
    S_set = set(S)
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i+1, n):
            for l in range(j+1, n):
                a, b, c = S_list[i], S_list[j], S_list[l]
                if a+b in S_set and a+c in S_set and b+c in S_set:
                    return False
    return True
