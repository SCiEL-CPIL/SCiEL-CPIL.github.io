import time
import random
import math
from itertools import combinations

def _max_sidon_size(A):
    """
    Exhaustively computes the largest subset of A with all unordered pair sums unique.
    A is a list of distinct integers.
    """
    n = len(A)
    if n == 0:
        return 0
    
    # Sort for deterministic behavior and potential pruning
    A = sorted(A)
    
    best_size = 0
    # Check subsets in decreasing size order to find max quickly?
    # Or just check all and track max.
    # For n=12, 2^12 = 4096 subsets. Very fast.
    
    for size in range(n, 0, -1):
        found = False
        for subset in combinations(A, size):
            # Check if subset is Sidon
            sums = set()
            is_sidon = True
            for i in range(len(subset)):
                for j in range(i, len(subset)):
                    s = subset[i] + subset[j]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                best_size = size
                found = True
                break
        if found:
            break
            
    return best_size

def _count_sidon_4_subsets(A):
    """
    Counts the number of 4-element subsets of A that are Sidon.
    """
    count = 0
    for subset in combinations(A, 4):
        sums = set()
        is_sidon = True
        for i in range(4):
            for j in range(i, 4):
                s = subset[i] + subset[j]
                if s in sums:
                    is_sidon = False
                    break
                sums.add(s)
            if not is_sidon:
                break
        if is_sidon:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Initial candidate: Arithmetic Progression
    current_best = list(range(12))
    current_best_cost = _count_sidon_4_subsets(current_best)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate neighbor
        A = list(current_best)
        idx = random.randint(0, len(A) - 1)
        old_val = A[idx]
        
        # Local move: perturb value
        while True:
            if random.random() < 0.8:
                new_val = random.randint(max(0, old_val - 10), min(128, old_val + 10))
            else:
                new_val = random.randint(0, 128)
            
            if new_val not in A or new_val == old_val:
                break
        
        A[idx] = new_val
        
        new_cost = _count_sidon_4_subsets(A)
        delta_cost = new_cost - current_best_cost
        
        if delta_cost < 0 or (T > T_min and random.random() < math.exp(-delta_cost / T)):
            current_best = A
            current_best_cost = new_cost
            
            if current_best_cost == 0:
                break
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return current_best
