import time
import random
from math import gcd

def construct():
    # Precompute LCM and A_i
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
        
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i
        
    best_K = None
    best_c = None
    
    deadline = time.time() + 50 # Safety margin
    
    def try_local_search(init_c):
        nonlocal best_K, best_c
        c = init_c[:]
        K = sum(c[i] * A[i] for i in range(1, 31))
        
        # Ensure K is not 0 at start
        if K == 0:
             # Try to perturb
             for i in range(1, 31):
                 if c[i] != 1:
                     c[i] = 1
                     K += A[i]
                     break
                 elif c[i] != -1:
                     c[i] = -1
                     K -= A[i]
                     break
        
        if K == 0:
            return

        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(1, 31))
            random.shuffle(indices)
            for i in indices:
                old_c = c[i]
                # Try other values
                # Order: try the one that moves K towards 0 first?
                # If K > 0, we want to decrease K. So try -1 if possible, or 0.
                # If K < 0, we want to increase K. So try 1 if possible, or 0.
                candidates = []
                if K > 0:
                    if old_c != -1: candidates.append(-1)
                    if old_c != 0: candidates.append(0)
                    if old_c != 1: candidates.append(1)
                else:
                    if old_c != 1: candidates.append(1)
                    if old_c != 0: candidates.append(0)
                    if old_c != -1: candidates.append(-1)
                
                for new_c in candidates:
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * A[i]
                    new_K = K + delta
                    if new_K == 0:
                        continue
                    if best_K is None or abs(new_K) < abs(best_K):
                        best_K = new_K
                        best_c = c[:]
                        c[i] = new_c
                        K = new_K
                        improved = True
                        break # Restart loop with new K and shuffled indices?
                        # Breaking allows re-shuffling, which is good.
            
            if not improved:
                # Random kick
                # Change 1 random coefficient
                i = random.randint(1, 30)
                old_c = c[i]
                # Pick a different value
                new_c = random.randint(-1, 1)
                if new_c == old_c:
                    new_c = (new_c + 1) % 3 - 1
                    if new_c == old_c:
                        new_c = (new_c + 1) % 3 - 1
                delta = (new_c - old_c) * A[i]
                new_K = K + delta
                if new_K != 0:
                    c[i] = new_c
                    K = new_K
                else:
                    # If it becomes 0, try another kick or revert
                    # Let's just revert and try another index
                    c[i] = old_c
                    # We don't update K since we reverted
                    # But we should mark that we didn't improve?
                    # The kick is meant to escape, so even if it doesn't improve immediately,
                    # it changes the landscape.
                    # However, if it hits 0, it's invalid.
                    # Let's try a different kick.
                    i2 = random.randint(1, 30)
                    old_c2 = c[i2]
                    new_c2 = random.randint(-1, 1)
                    if new_c2 == old_c2:
                        new_c2 = (new_c2 + 1) % 3 - 1
                        if new_c2 == old_c2:
                            new_c2 = (new_c2 + 1) % 3 - 1
                    delta2 = (new_c2 - old_c2) * A[i2]
                    new_K2 = K + delta2
                    if new_K2 != 0:
                        c[i2] = new_c2
                        K = new_K2
                    # If still 0, we are stuck in a 0-neighborhood? Unlikely.
        
        # End local search

    # Run multiple restarts
    while time.time() < deadline:
        # Random initialization
        c_init = [0] * 31
        for i in range(1, 31):
            c_init[i] = random.randint(-1, 1)
        
        try_local_search(c_init)
        
        # Occasionally try a greedy initialization
        if random.random() < 0.1:
            c_greedy = [0] * 31
            K_g = 0
            # Sort by A[i] descending? A[1] is largest.
            # Just iterate 1 to 30
            for i in range(1, 31):
                # Try -1, 0, 1
                best_val = float('inf')
                best_c_val = 0
                for val in [-1, 0, 1]:
                    new_K = K_g + val * A[i]
                    if new_K == 0:
                        continue
                    if abs(new_K) < best_val:
                        best_val = abs(new_K)
                        best_c_val = val
                c_greedy[i] = best_c_val
                K_g += best_c_val * A[i]
            try_local_search(c_greedy)
        
    if best_c is None:
        return [0] * 28 + [1, -1]
        
    return [best_c[i] for i in range(1, 31)]
