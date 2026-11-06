import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 128
    A = np.ones(n + 1, dtype=bool)
    counts = np.zeros(257, dtype=np.int32)
    
    # Initialize counts for A = all ones
    # For each x in 0..128, for each y in x..128, counts[x+y] += 1
    for x in range(n + 1):
        for y in range(x, n + 1):
            counts[x + y] += 1
            
    def get_cost():
        # Peak and uncovered in [128, 256]
        target = counts[128:257]
        peak = np.max(target)
        uncovered = np.sum(target == 0)
        return peak * 1000 + uncovered, int(peak), int(uncovered)
    
    current_cost, current_peak, current_uncovered = get_cost()
    best_cost = current_cost
    best_peak = current_peak
    best_uncovered = current_uncovered
    best_A = A.copy()
    
    random.seed(42)
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline - 5.0:
        idx = random.randint(0, n)
        
        if A[idx]:
            # Remove idx
            A[idx] = False
            counts[2 * idx] -= 1
            for y in range(n + 1):
                if A[y]:
                    counts[idx + y] -= 1
            
            new_cost, new_peak, new_uncovered = get_cost()
            delta = new_cost - current_cost
            
            if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
                current_cost = new_cost
                current_peak = new_peak
                current_uncovered = new_uncovered
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_peak = new_peak
                    best_uncovered = new_uncovered
                    best_A = A.copy()
                    if best_peak == 1 and best_uncovered == 0:
                        break
            else:
                # Revert
                counts[2 * idx] += 1
                for y in range(n + 1):
                    if A[y]:
                        counts[idx + y] += 1
                A[idx] = True
        else:
            # Add idx
            for y in range(n + 1):
                if A[y]:
                    counts[idx + y] += 1
            counts[2 * idx] += 1
            A[idx] = True
            
            new_cost, new_peak, new_uncovered = get_cost()
            delta = new_cost - current_cost
            
            if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
                current_cost = new_cost
                current_peak = new_peak
                current_uncovered = new_uncovered
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_peak = new_peak
                    best_uncovered = new_uncovered
                    best_A = A.copy()
                    if best_peak == 1 and best_uncovered == 0:
                        break
            else:
                # Revert
                A[idx] = False
                counts[2 * idx] -= 1
                for y in range(n + 1):
                    if A[y]:
                        counts[idx + y] -= 1
                        
        T *= cooling
        
    # Polish: Greedy removal
    A = best_A.copy()
    # Recalculate counts for best_A to be safe
    counts = np.zeros(257, dtype=np.int32)
    for x in range(n + 1):
        if A[x]:
            for y in range(x, n + 1):
                if A[y]:
                    counts[x + y] += 1
                    
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        for idx in range(n, -1, -1):
            if not A[idx]:
                continue
            # Try remove
            A[idx] = False
            counts[2 * idx] -= 1
            for y in range(n + 1):
                if A[y]:
                    counts[idx + y] -= 1
            
            target = counts[128:257]
            new_peak = np.max(target)
            new_uncovered = np.sum(target == 0)
            
            if new_uncovered == 0 and new_peak <= best_peak:
                best_peak = new_peak
                best_uncovered = new_uncovered
                best_A = A.copy()
                best_cost = new_peak * 1000 + new_uncovered
                changed = True
            else:
                # Revert
                counts[2 * idx] += 1
                for y in range(n + 1):
                    if A[y]:
                        counts[idx + y] += 1
                A[idx] = True
                
    result = [i for i in range(n + 1) if best_A[i]]
    return result
