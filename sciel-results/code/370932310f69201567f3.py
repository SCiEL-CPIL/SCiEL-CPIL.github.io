"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    global_deadline = start_time + 145  # 5s safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if len(S) <= 1:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        if 2 * x in S_set:
            return False
        for a in S_set:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
        return True

    def greedy_construct(S, order):
        S_set = set()
        result = []
        for x in order:
            if can_add(x, S_set):
                result.append(x)
                S_set.add(x)
        return result

    def polish(S, current_set):
        """Try to add any element that can be added."""
        S_set = set(current_set)
        improved = True
        while improved:
            improved = False
            for x in S:
                if x not in S_set:
                    if can_add(x, S_set):
                        S_set.add(x)
                        improved = True
        return list(S_set)

    def solve_instance(S, seed_offset, deadline):
        best_set = []
        best_size = 0
        
        # Baseline: Odd numbers
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > best_size:
            best_set = odds
            best_size = len(odds)
            
        # Baseline: Large absolute values
        large_abs = sorted(S, key=lambda x: -abs(x))
        cand = greedy_construct(S, large_abs)
        cand = polish(S, cand)
        if len(cand) > best_size:
            best_set = cand
            best_size = len(cand)
            
        # Baseline: Small absolute values
        small_abs = sorted(S, key=lambda x: abs(x))
        cand = greedy_construct(S, small_abs)
        cand = polish(S, cand)
        if len(cand) > best_size:
            best_set = cand
            best_size = len(cand)

        # Randomized Greedy Restarts
        rng = random.Random(seed_offset)
        
        while time.time() < deadline - 2:
            # Generate random order
            order = S[:]
            rng.shuffle(order)
            
            cand = greedy_construct(S, order)
            cand = polish(S, cand)
            
            if len(cand) > best_size:
                best_set = cand
                best_size = len(cand)
                
        # Final polish on the best set
        best_set = polish(S, best_set)
        
        # Verify
        if not is_sum_free(best_set):
             # Fallback
             best_set = odds
             
        return best_set

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Allocate time
    # N=60: 5s
    # N=120: 10s
    # N=240: rest (~130s)
    
    time_allocs = [5, 10, None]
    
    for i, (size, seed) in enumerate(instances):
        S = instance(size, seed)
        
        if i < 2:
            inst_deadline = start_time + time_allocs[i]
        else:
            inst_deadline = global_deadline - 2
            
        best = solve_instance(S, seed, inst_deadline)
        results.append(best)
        
        # If we are running out of time, just finish quickly for remaining
        if time.time() > global_deadline - 5:
             for j in range(i+1, len(instances)):
                 sz, sd = instances[j]
                 S_j = instance(sz, sd)
                 results.append([x for x in S_j if x % 2 != 0])
             break

    return results
# EVOLVE-BLOCK-END
