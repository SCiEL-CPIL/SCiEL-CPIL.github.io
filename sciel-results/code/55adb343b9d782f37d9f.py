"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    
    # Helper to calculate cost: peak multiplicity
    def get_cost(A):
        A = sorted(A)
        counts = [0] * (max_sum + 1)
        nA = len(A)
        for i in range(nA):
            ai = A[i]
            s_ii = ai + ai
            if min_sum <= s_ii <= max_sum:
                counts[s_ii] += 1
            for j in range(i + 1, nA):
                s_ij = ai + A[j]
                if min_sum <= s_ij <= max_sum:
                    counts[s_ij] += 1
        # Check coverage
        for s in target_sums:
            if counts[s] == 0:
                return float('inf')
        return max(counts[min_sum:max_sum+1])
    
    def get_cost_with_counts(A):
        A = sorted(A)
        counts = [0] * (max_sum + 1)
        nA = len(A)
        for i in range(nA):
            ai = A[i]
            s_ii = ai + ai
            if min_sum <= s_ii <= max_sum:
                counts[s_ii] += 1
            for j in range(i + 1, nA):
                s_ij = ai + A[j]
                if min_sum <= s_ij <= max_sum:
                    counts[s_ij] += 1
        for s in target_sums:
            if counts[s] == 0:
                return float('inf'), counts
        return max(counts[min_sum:max_sum+1]), counts
    
    # Start with a reasonable initial solution
    # The seed returns range(64,129) which has 65 elements
    # Let's try to find something better
    
    best_score = float('inf')
    best_set = None
    
    # Initial solution: try the seed
    initial = list(range(64, 129))
    cost, _ = get_cost_with_counts(initial)
    if cost < best_score:
        best_score = cost
        best_set = initial.copy()
    
    # Try some structured initial solutions
    # Symmetric sets around 64
    for offset in range(0, 20):
        candidate = list(range(64 - offset, 65 + offset))
        candidate = [x for x in candidate if 0 <= x <= 128]
        if len(candidate) > 0:
            c, _ = get_cost_with_counts(candidate)
            if c < best_score:
                best_score = c
                best_set = candidate.copy()
    
    # Try random sets of various sizes
    random.seed(42)
    
    # Strategy: try different sizes of A
    for size in range(20, 70):
        if time.time() > deadline - 5:
            break
        
        attempts = 50
        for _ in range(attempts):
            if time.time() > deadline - 3:
                    break
            
            # Generate random set of given size
            A = random.sample(range(0, 129), size)
            A = sorted(A)
            
            c, _ = get_cost_with_counts(A)
            if c < best_score:
                best_score = c
                best_set = A.copy()
                # Try to improve with local search
                improved = True
                while improved and time.time() < deadline - 2:
                    improved = False
                    c, counts = get_cost_with_counts(best_set)
                    if c == float('inf'):
                        break
                    
                    # Try swapping elements
                    for i in range(len(best_set)):
                        if time.time() > deadline - 1:
                            break
                        for new_val in range(0, 129):
                            if new_val in best_set:
                                continue
                            trial = best_set[:i] + [new_val] + best_set[i+1:]
                            trial = sorted(trial)
                            tc, _ = get_cost_with_counts(trial)
                            if tc < c:
                                best_set = trial
                                c = tc
                                best_score = c
                                improved = True
                                break
    
    # Final local search on the best solution found
    if best_set is not None and time.time() < deadline - 1:
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            c, counts = get_cost_with_counts(best_set)
            if c == float('inf'):
                break
            
            # Try removing elements
            for i in range(len(best_set)):
                if time.time() > deadline - 0.5:
                    break
                trial = best_set[:i] + best_set[i+1:]
                if len(trial) > 0:
                    tc, _ = get_cost_with_counts(trial)
                    if tc <= c:
                        best_set = trial
                        c = tc
                        best_score = min(best_score, c)
                        improved = True
                        break
            
            # Try adding elements
            for new_val in range(0, 129):
                if time.time() > deadline - 0.5:
                    break
                if new_val in best_set:
                    continue
                trial = best_set + [new_val]
                trial = sorted(trial)
                tc, _ = get_cost_with_counts(trial)
                if tc < c:
                    best_set = trial
                    c = tc
                    best_score = min(best_score, c)
                    improved = True
                    break
    
    if best_set is None or len(best_set) == 0:
        return list(range(64, 129))
    
    return list(best_set)

# EVOLVE-BLOCK-END
