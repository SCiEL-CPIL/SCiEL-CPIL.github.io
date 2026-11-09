import random
import time
import numpy as np
from scipy.fft import fft, ifft

def construct():
    N = 512
    deadline = time.time() + 85.0  # Safety margin
    
    def base4_seed():
        # Base-4 construction: numbers with only 0s and 1s in base 4
        S = bytearray(N + 1)
        for i in range(1, N + 1):
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

    # Precompute all 4-APs using FFT-based convolution
    # A 4-AP is (a, a+d, a+2d, a+3d) for d >= 1
    # We'll build a conflict matrix: conflict[x][y] = True if x and y cannot both be in S
    # Actually, we need: can_add(x, S) = True iff no 4-AP exists containing x and 3 other elements of S
    
    # Better approach: precompute for each x, the list of "forbidden triples"
    # If x is added, we need to check if there exists d such that:
    # - x-d, x-2d, x-3d all in S (x is 4th term)
    # - x-d, x+d, x+2d all in S (x is 2nd term)  
    # - x+d, x+2d, x+3d all in S (x is 1st term)
    # - x-2d, x-d, x+d all in S (x is 3rd term)
    
    # Precompute all possible 4-APs
    all_4aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3*d + 1):
            all_4aps.append((a, a+d, a+2*d, a+3*d))
    
    # For each element x, precompute which other elements form a 4-AP with x
    # conflict_pairs[x] = set of y such that x and y are in some 4-AP together
    # But we need stronger: for can_add(x, S), we need to check if S contains 3 elements that with x form a 4-AP
    
    # Let's precompute for each x, the list of (y1, y2, y3) triples that with x form a 4-AP
    # This is too large. Instead, let's use a different approach.
    
    # Key insight: For can_add(x, S), we can check each of the 4 patterns separately
    # and use bitsets for fast checking
    
    # Precompute bitsets for each position offset
    # For pattern "x, x+d, x+2d, x+3d": if we fix x, we need to check if for any d, 
    # S[x+d] & S[x+2d] & S[x+3d] is true
    
    # Let's use numpy for faster operations
    def can_add_fast(x, S_np):
        """S_np is a numpy boolean array of size N+1"""
        # Case 1: x is first term - check if exists d: S[x+d] & S[x+2d] & S[x+3d]
        max_d = (N - x) // 3
        if max_d >= 1:
            # Vectorized check
            ds = np.arange(1, max_d + 1)
            if np.any(S_np[x+ds] & S_np[x+2*ds] & S_np[x+3*ds]):
                return False
        
        # Case 2: x is second term - check if exists d: S[x-d] & S[x+d] & S[x+2d]
        max_d = min(x - 1, (N - x) // 2)
        if max_d >= 1:
            ds = np.arange(1, max_d + 1)
            if np.any(S_np[x-ds] & S_np[x+ds] & S_np[x+2*ds]):
                return False
        
        # Case 3: x is third term - check if exists d: S[x-2d] & S[x-d] & S[x+d]
        max_d = min(x // 2, N - x)
        if max_d >= 1:
            ds = np.arange(1, max_d + 1)
            if np.any(S_np[x-2*ds] & S_np[x-ds] & S_np[x+ds]):
                return False
        
        # Case 4: x is fourth term - check if exists d: S[x-3d] & S[x-2d] & S[x-d]
        max_d = x // 3
        if max_d >= 1:
            ds = np.arange(1, max_d + 1)
            if np.any(S_np[x-3*ds] & S_np[x-2*ds] & S_np[x-ds]):
                return False
        
        return True

    def greedy_random():
        S_arr = bytearray(N + 1)
        S_np = np.zeros(N + 1, dtype=bool)
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add_fast(x, S_np):
                S_arr[x] = 1
                S_np[x] = True
        return S_arr

    # Start with base-4 seed as initial best
    best_arr = base4_seed()
    best_size = sum(best_arr)
    
    # Phase 1: Randomized Greedy - run more iterations due to faster can_add
    for i in range(5000):
        if time.time() > deadline - 30:  # Leave time for local search
            break
        S = greedy_random()
        size = sum(S)
        if size > best_size:
            best_size = size
            best_arr = S.copy()
            
    # Phase 2: Local Search (Remove-Add) with faster checks
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S = best_arr.copy()
        S_np = np.array(S, dtype=bool)
        elements = [i for i in range(1, N+1) if S[i]]
        random.shuffle(elements)
        
        max_removals_to_try = min(len(elements), 150)
        for u in elements[:max_removals_to_try]:
            if time.time() > deadline - 5:
                break
            S_temp = S.copy()
            S_np_temp = S_np.copy()
            S_temp[u] = 0
            S_np_temp[u] = False
            
            # Try to add any element not in S_temp
            added = False
            for v in range(1, N+1):
                if S_temp[v]:
                    continue
                if can_add_fast(v, S_np_temp):
                    S_temp[v] = 1
                    S_np_temp[v] = True
                    added = True
                    
            new_size = sum(S_temp)
            if new_size > best_size:
                S = S_temp
                S_np = S_np_temp
                best_arr = S_temp
                best_size = new_size
                improved = True
                break # Restart with new best
                
    # Phase 3: Final Polish
    S = best_arr
    S_np = np.array(S, dtype=bool)
    for v in range(1, N+1):
        if not S[v] and can_add_fast(v, S_np):
            S[v] = 1
            S_np[v] = True
            
    return [i for i in range(1, N+1) if S[i]]
