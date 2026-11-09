import numpy as np
import time

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    def greedy_numpy(order):
        """
        Greedy construction using numpy for speed.
        order: array of candidate elements to consider
        Returns: set of elements that form a 4-AP-free set
        """
        S = np.zeros(N + 1, dtype=bool)
        result = []
        
        for x in order:
            valid = True
            # Check if adding x creates any 4-AP
            # We need to check all 4 positions x can be in a 4-AP
            
            # Precompute max d for each position
            # Position 1 (start): x, x+d, x+2d, x+3d -> d <= (N-x)//3
            # Position 2: x-d, x, x+d, x+2d -> d >= x-1, d <= (N-x)//2
            # Position 3: x-2d, x-d, x, x+d -> d <= (x-1)//2, d <= (N-x)//1
            # Position 4 (end): x-3d, x-2d, x-d, x -> d <= (x-1)//3
            
            max_d = N // 3
            
            # Position 1: x is first term
            d_vals = np.arange(1, min(max_d, (N - x) // 3) + 1)
            if len(d_vals) > 0:
                t1 = x + d_vals
                t2 = x + 2 * d_vals
                t3 = x + 3 * d_vals
                if np.any(S[t1] & S[t2] & S[t3]):
                    valid = False
            
            if not valid:
                continue
                
            # Position 2: x is second term
            d_vals = np.arange(1, min(max_d, (N - x) // 2) + 1)
            d_vals = d_vals[d_vals < x]  # x-d >= 1 means d < x
            if len(d_vals) > 0:
                t1 = x - d_vals
                t2 = x + d_vals
                t3 = x + 2 * d_vals
                if np.any(S[t1] & S[t2] & S[t3]):
                    valid = False
            
            if not valid:
                continue
                
            # Position 3: x is third term
            d_vals = np.arange(1, min(max_d, x // 2) + 1)
            d_vals = d_vals[d_vals <= (N - x)]  # x+d <= N
            if len(d_vals) > 0:
                t1 = x - 2 * d_vals
                t2 = x - d_vals
                t3 = x + d_vals
                if np.any(S[t1] & S[t2] & S[t3]):
                    valid = False
            
            if not valid:
                continue
                
            # Position 4: x is fourth term
            d_vals = np.arange(1, min(max_d, (x - 1) // 3) + 1)
            if len(d_vals) > 0:
                t1 = x - 3 * d_vals
                t2 = x - 2 * d_vals
                t3 = x - d_vals
                if np.any(S[t1] & S[t2] & S[t3]):
                    valid = False
            
            if valid:
                S[x] = True
                result.append(x)
        
        return result
    
    # Run multiple random greedy restarts
    elements = np.arange(1, N + 1)
    best_size = 0
    best_set = []
    
    rng = np.random.default_rng()
    
    for i in range(5000):
        if time.time() > deadline:
            break
        order = rng.permutation(elements)
        S = greedy_numpy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    # Local Search: Try to add any remaining valid elements
    if best_set:
        S = np.zeros(N + 1, dtype=bool)
        for x in best_set:
            S[x] = True
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if not S[x]:
                    valid = True
                    max_d = N // 3
                    
                    # Position 1
                    d_vals = np.arange(1, min(max_d, (N - x) // 3) + 1)
                    if len(d_vals) > 0:
                        t1 = x + d_vals
                        t2 = x + 2 * d_vals
                        t3 = x + 3 * d_vals
                        if np.any(S[t1] & S[t2] & S[t3]):
                            valid = False
                    
                    if not valid:
                        continue
                        
                    # Position 2
                    d_vals = np.arange(1, min(max_d, (N - x) // 2) + 1)
                    d_vals = d_vals[d_vals < x]
                    if len(d_vals) > 0:
                        t1 = x - d_vals
                        t2 = x + d_vals
                        t3 = x + 2 * d_vals
                        if np.any(S[t1] & S[t2] & S[t3]):
                            valid = False
                    
                    if not valid:
                        continue
                        
                    # Position 3
                    d_vals = np.arange(1, min(max_d, x // 2) + 1)
                    d_vals = d_vals[d_vals <= (N - x)]
                    if len(d_vals) > 0:
                        t1 = x - 2 * d_vals
                        t2 = x - d_vals
                        t3 = x + d_vals
                        if np.any(S[t1] & S[t2] & S[t3]):
                            valid = False
                    
                    if not valid:
                        continue
                        
                    # Position 4
                    d_vals = np.arange(1, min(max_d, (x - 1) // 3) + 1)
                    if len(d_vals) > 0:
                        t1 = x - 3 * d_vals
                        t2 = x - 2 * d_vals
                        t3 = x - d_vals
                        if np.any(S[t1] & S[t2] & S[t3]):
                            valid = False
                    
                    if valid:
                        S[x] = True
                        best_set.append(x)
                        changed = True
                        break
    
    return sorted(best_set)
