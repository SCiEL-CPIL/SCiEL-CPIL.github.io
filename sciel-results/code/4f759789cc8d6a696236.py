import random
import time
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    n = len(A)
    for k in range(n, 0, -1):
        for comb in combinations(A, k):
            if is_sidon(list(comb)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initialize with arithmetic progression
    A = list(range(12))
    current_L = max_sidon_size(A)
    best_L = current_L
    best_A = A[:]
    
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    
    while time.time() < deadline:
        i = random.randint(0, 11)
        old_val = A[i]
        new_val = random.randint(0, 128)
        
        if new_val == old_val:
            continue
        if new_val in A:
            continue
            
        A[i] = new_val
        new_L = max_sidon_size(A)
        
        accepted = False
        if new_L < current_L:
            accepted = True
        elif new_L == current_L:
            accepted = True
        else:
            delta = new_L - current_L
            p = pow(2.71828, -delta / T)
            if random.random() < p:
                accepted = True
        
        if accepted:
            current_L = new_L
            if current_L < best_L:
                best_L = current_L
                best_A = A[:]
        else:
            A[i] = old_val
            
        T *= alpha
        if T < T_min:
            T = T_min

    return best_A
