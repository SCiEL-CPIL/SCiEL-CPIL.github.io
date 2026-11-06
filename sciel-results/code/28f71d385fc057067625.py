"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    target_min = 128
    target_max = 256
    num_targets = 129
    
    def compute_rep_counts(A_list):
        k = len(A_list)
        rep_counts = np.zeros(num_targets, dtype=np.int32)
        for i in range(k):
            a = A_list[i]
            for j in range(i, k):
                b = A_list[j]
                s = a + b
                if target_min <= s <= target_max:
                    rep_counts[s - target_min] += 1
        return rep_counts
    
    def is_valid(rep_counts):
        return np.all(rep_counts > 0)
    
    def peak_of(rep_counts):
        return int(np.max(rep_counts))
    
    # Initial solution: A = {64, 65, ..., 128}
    A = list(range(64, 129))
    A_set = set(A)
    rep_counts = compute_rep_counts(A)
    best_peak = peak_of(rep_counts)
    best_A = A[:]
    
    random.seed(42)
    
    # Greedy removal: try to remove elements that reduce peak while maintaining coverage
    for _ in range(500):  # Limit greedy phase
        if time.time() > deadline:
            break
        if len(A) <= 2:
            break
        
        current_peak = peak_of(rep_counts)
        improved = False
        
        # Try removing each element, prefer those in high-contribution sums
        for elem in sorted(A_set, key=lambda x: -x):  # Try larger elements first
            A_test = [x for x in A if x != elem]
            rep_test = compute_rep_counts(A_test)
            
            if is_valid(rep_test):
                pm_test = peak_of(rep_test)
                if pm_test < current_peak:
                    A = A_test
                    A_set.discard(elem)
                    rep_counts = rep_test
                    best_peak = pm_test
                    best_A = A[:]
                    improved = True
                    break
        
        if not improved:
            # Try adding an element to enable better removals
            candidates = [x for x in range(0, 129) if x not in A_set]
            if not candidates:
                break
            random.shuffle(candidates)
            for elem_add in candidates[:10]:
                A_test = A + [elem_add]
                A_test.sort()
                rep_test = compute_rep_counts(A_test)
                if is_valid(rep_test):
                    pm_test = peak_of(rep_test)
                    # Only add if it doesn't worsen peak significantly
                    if pm_test <= best_peak + 1:
                        # Check if we can then remove something to improve
                        for elem_rm in sorted(A_test):
                            A_test2 = [x for x in A_test if x != elem_rm]
                            rep_test2 = compute_rep_counts(A_test2)
                            if is_valid(rep_test2):
                                pm_test2 = peak_of(rep_test2)
                                if pm_test2 < best_peak:
                                    A = A_test2
                                    A_set = set(A)
                                    rep_counts = rep_test2
                                    best_peak = pm_test2
                                    best_A = A[:]
                                    break
                        break
    
    # Random local search
    while time.time() < deadline:
        if len(A) < 2:
            break
        
        idx_remove = random.randint(0, len(A) - 1)
        elem_remove = A[idx_remove]
        
        candidates = [x for x in range(0, 129) if x not in A_set]
        if not candidates:
            break
        elem_add = random.choice(candidates)
        
        A_new = A[:]
        A_new.pop(idx_remove)
        A_new.append(elem_add)
        A_new.sort()
        
        rep_new = compute_rep_counts(A_new)
        
        if is_valid(rep_new):
            pm = peak_of(rep_new)
            if pm < best_peak:
                best_peak = pm
                best_A = A_new
                A = A_new
                A_set = set(A)
                rep_counts = rep_new
    
    return sorted(best_A)

# EVOLVE-BLOCK-END
