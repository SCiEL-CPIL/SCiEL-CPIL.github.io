import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    target_low = 128
    target_high = 256
    
    # Initial A: [64, 128]
    A = [False] * (n + 1)
    for i in range(64, n + 1):
        A[i] = True
    
    # Compute multiplicities
    mult = [0] * (2 * n + 1)
    for i in range(n + 1):
        if A[i]:
            for j in range(i, n + 1):
                if A[j]:
                    mult[i + j] += 1
    
    def get_cost():
        max_m = 0
        uncovered = 0
        for s in range(target_low, target_high + 1):
            if mult[s] == 0:
                uncovered += 1
            else:
                if mult[s] > max_m:
                    max_m = mult[s]
        return max_m + 1000 * uncovered
    
    current_cost = get_cost()
    best_A = A.copy()
    best_cost = current_cost
    
    temp = 10.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # Random move
        x = random.randint(0, n)
        
        # Apply move
        if A[x]:
            # Remove x
            for y in range(n + 1):
                if A[y]:
                    mult[x + y] -= 1
            A[x] = False
        else:
            # Add x
            for y in range(n + 1):
                if A[y]:
                    mult[x + y] += 1
            A[x] = True
            
        new_cost = get_cost()
        
        # Accept/Reject
        if new_cost < current_cost:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_A = A.copy()
        else:
            # Delta = new_cost - current_cost (positive)
            delta = new_cost - current_cost
            if temp > 0:
                p = np.exp(-delta / temp)
                if random.random() < p:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = A.copy()
                else:
                    # Revert
                    if A[x]:
                        # It was added, so revert by removing
                        for y in range(n + 1):
                            if A[y]:
                                mult[x + y] -= 1
                        A[x] = False
                    else:
                        # It was removed, so revert by adding
                        for y in range(n + 1):
                            if A[y]:
                                mult[x + y] += 1
                        A[x] = True
            else:
                # Revert
                if A[x]:
                    for y in range(n + 1):
                        if A[y]:
                            mult[x + y] -= 1
                    A[x] = False
                else:
                    for y in range(n + 1):
                        if A[y]:
                            mult[x + y] += 1
                    A[x] = True
                    
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    return [i for i in range(n + 1) if best_A[i]]
