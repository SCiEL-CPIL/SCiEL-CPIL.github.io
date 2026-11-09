import random
import time

def construct():
    N = 512
    deadline = time.time() + 85.0  # Safety margin
    
    def base4_seed():
        # Base-4 construction: numbers with only 0s and 1s in base 4
        # This is a known 4-AP-free set
        S = bytearray(N + 1)
        for i in range(1, N + 1):
            # Check if i in base 4 has only 0s and 1s
            x = i
            valid = True
            while x > 0:
                if x % 4 > 1:
                    valid = False
                    break
                x //= 4
            if valid:
                S[i] = 1
        return S

    def can_add(x, S):
        # S is a bytearray of size N+1, S[i] is 1 if i is in S
        # Check if adding x creates a 4-AP
        # Patterns:
        # 1. x, x+d, x+2d, x+3d
        # 2. x-d, x, x+d, x+2d
        # 3. x-2d, x-d, x, x+d
        # 4. x-3d, x-2d, x-d, x
        
        # Case 1: x is first term
        d = 1
        while x + 3*d <= N:
            if S[x+d] and S[x+2*d] and S[x+3*d]:
                return False
            d += 1
            
        # Case 2: x is second term
        d = 1
        while x - d >= 1 and x + 2*d <= N:
            if S[x-d] and S[x+d] and S[x+2*d]:
                return False
            d += 1
            
        # Case 3: x is third term
        d = 1
        while x - 2*d >= 1 and x + d <= N:
            if S[x-2*d] and S[x-d] and S[x+d]:
                return False
            d += 1
            
        # Case 4: x is fourth term
        d = 1
        while x - 3*d >= 1:
            if S[x-3*d] and S[x-2*d] and S[x-d]:
                return False
            d += 1
                
        return True

    def greedy_random():
        S_arr = bytearray(N + 1)
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S_arr):
                S_arr[x] = 1
        return S_arr

    # Start with base-4 seed as initial best
    best_arr = base4_seed()
    best_size = sum(best_arr)
    
    # Phase 1: Randomized Greedy
    num_greedy = 2000
    for i in range(num_greedy):
        if time.time() > deadline:
            break
        S = greedy_random()
        size = sum(S)
        if size > best_size:
            best_size = size
            best_arr = S.copy()
            
    # Phase 2: Local Search (Remove-Add)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S = best_arr.copy()
        elements = [i for i in range(1, N+1) if S[i]]
        random.shuffle(elements)
        
        max_removals_to_try = 100
        for u in elements[:max_removals_to_try]:
            if time.time() > deadline:
                break
            S_temp = S.copy()
            S_temp[u] = 0
            # Try to add any element not in S_temp
            for v in range(1, N+1):
                if S_temp[v]:
                    continue
                if can_add(v, S_temp):
                    S_temp[v] = 1
                    
            new_size = sum(S_temp)
            if new_size > best_size:
                S = S_temp
                best_arr = S_temp
                best_size = new_size
                improved = True
                break # Restart with new best
                
    # Phase 3: Final Polish
    S = best_arr
    for v in range(1, N+1):
        if not S[v] and can_add(v, S):
            S[v] = 1
            
    return [i for i in range(1, N+1) if S[i]]
