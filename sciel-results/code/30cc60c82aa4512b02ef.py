import random
import time

def construct():
    N = 512
    deadline = time.time() + 85.0  # Safety margin
    
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

    def base4_seed():
        """Generate a 4-AP-free set using base-4 digits 0 and 1 only."""
        S = bytearray(N + 1)
        for num in range(1, N + 1):
            # Check if all base-4 digits are 0 or 1
            n = num
            valid = True
            while n > 0:
                digit = n % 4
                if digit > 1:
                    valid = False
                    break
                n //= 4
            if valid:
                S[num] = 1
        return S

    def greedy_random(start_set=None):
        S_arr = bytearray(N + 1)
        if start_set is not None:
            for i in range(1, N + 1):
                if start_set[i]:
                    S_arr[i] = 1
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            if not S_arr[x] and can_add(x, S_arr):
                S_arr[x] = 1
        return S_arr

    def local_search(S, max_iterations=50):
        """Swap-based local search: remove one element, try to add multiple others."""
        S = S.copy()
        current_size = sum(S)
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            elements = [i for i in range(1, N+1) if S[i]]
            if not elements:
                break
            random.shuffle(elements)
            
            best_swap = None
            best_new_size = current_size
            
            # Try removing each element and adding as many as possible
            for u in elements[:30]:
                if time.time() > deadline:
                    break
                S_temp = S.copy()
                S_temp[u] = 0
                added_count = 0
                for v in range(1, N+1):
                    if S_temp[v]:
                        continue
                    if can_add(v, S_temp):
                        S_temp[v] = 1
                        added_count += 1
                
                new_size = sum(S_temp)
                if new_size > best_new_size:
                    best_new_size = new_size
                    best_swap = S_temp
                    
            if best_swap is not None and best_new_size > current_size:
                S = best_swap
                current_size = best_new_size
            else:
                break
                
        return S, current_size

    best_arr = bytearray(N + 1)
    best_size = 0
    
    # Phase 1a: Base-4 seed + greedy completion
    b4 = base4_seed()
    b4_size = sum(b4)
    if b4_size > best_size:
        best_size = b4_size
        best_arr = b4.copy()
    
    # Complete the base-4 seed with greedy
    S_completed = greedy_random(start_set=b4)
    S_completed_size = sum(S_completed)
    if S_completed_size > best_size:
        best_size = S_completed_size
        best_arr = S_completed.copy()
    
    # Phase 1b: Randomized Greedy from empty
    for i in range(3000):
        if time.time() > deadline:
            break
        S = greedy_random()
        size = sum(S)
        if size > best_size:
            best_size = size
            best_arr = S.copy()
    
    # Phase 1c: Randomized Greedy from base-4 seed
    for i in range(1000):
        if time.time() > deadline:
            break
        S = greedy_random(start_set=b4)
        size = sum(S)
        if size > best_size:
            best_size = size
            best_arr = S.copy()
    
    # Phase 2: Local Search (Remove-Add with swap)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_result, size_result = local_search(best_arr, max_iterations=100)
        if size_result > best_size:
            best_size = size_result
            best_arr = S_result
            improved = True
    
    # Phase 3: Final Polish
    # Try to add any remaining element
    S = best_arr
    for v in range(1, N+1):
        if not S[v] and can_add(v, S):
            S[v] = 1
    
    return [i for i in range(1, N+1) if S[i]]
