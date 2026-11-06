"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    N = 1000
    max_sum = 2 * N
    sum_counts = [0] * (max_sum + 1)
    
    def can_add(x, chosen_list):
        if sum_counts[2 * x] >= 2:
            return False
        for y in chosen_list:
            if sum_counts[x + y] >= 2:
                return False
        return True

    def add(x, chosen_list, chosen_set):
        sum_counts[2 * x] += 1
        for y in chosen_list:
            sum_counts[x + y] += 1
        chosen_list.append(x)
        chosen_set.add(x)

    def remove(x, chosen_list, chosen_set):
        sum_counts[2 * x] -= 1
        for y in chosen_list:
            if y != x:
                sum_counts[x + y] -= 1
        for i in range(len(chosen_list)):
            if chosen_list[i] == x:
                chosen_list[i] = chosen_list[-1]
                chosen_list.pop()
                break
        chosen_set.discard(x)

    def run_greedy(seed, order=None):
        rng = random.Random(seed)
        for i in range(max_sum + 1):
            sum_counts[i] = 0
        chosen_list = []
        chosen_set = set()
        
        if order is None:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        else:
            candidates = order[:]
            rng.shuffle(candidates)
            
        for x in candidates:
            if can_add(x, chosen_list):
                add(x, chosen_list, chosen_set)
                
        return list(chosen_set)

    def rebuild_counts(chosen):
        for i in range(max_sum + 1):
            sum_counts[i] = 0
        cl = []
        cs = set()
        for x in chosen:
            add(x, cl, cs)
        return cl, cs

    # 1. Find a good starting set using multiple strategies
    best_start = []
    best_start_size = 0
    
    # Strategy A: Ascending greedy
    asc_set = run_greedy(0, list(range(1, N + 1)))
    if len(asc_set) > best_start_size:
        best_start_size = len(asc_set)
        best_start = asc_set
        
    # Strategy B: Descending greedy
    desc_set = run_greedy(1, list(range(N, 0, -1)))
    if len(desc_set) > best_start_size:
        best_start_size = len(desc_set)
        best_start = desc_set
        
    # Strategy C: Randomized greedy with different seeds
    for s in range(30):
        candidate_set = run_greedy(s)
        if len(candidate_set) > best_start_size:
            best_start_size = len(candidate_set)
            best_start = candidate_set

    # Initialize state with best_start
    current_list, current_set = rebuild_counts(best_start)
    best_size = len(current_list)
    best_result = list(current_set)
    
    rng = random.Random(42)
    
    # Local search phase: kick-and-refill
    while time.time() < deadline - 5:
        # Save state
        saved_counts = sum_counts[:]
        saved_list = current_list[:]
        saved_set = set(current_set)
        
        # Decide how many to kick out
        kick_size = rng.randint(1, 4)
        
        if len(current_list) > kick_size:
            # Remove random elements
            indices = rng.sample(range(len(current_list)), kick_size)
            indices.sort(reverse=True)
            for idx in indices:
                u = current_list[idx]
                remove(u, current_list, current_set)
                
            # Greedy fill with randomized order
            candidates = list(set(range(1, N + 1)) - current_set)
            rng.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add(x, current_list):
                    add(x, current_list, current_set)
                
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_result = list(current_set)
            else:
                # Restore state
                sum_counts[:] = saved_counts
                current_list = saved_list
                current_set = saved_set
        else:
            # Not enough elements to kick, just try adding
            candidates = list(set(range(1, N + 1)) - current_set)
            rng.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add(x, current_list):
                    add(x, current_list, current_set)
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_result = list(current_set)
            else:
                sum_counts[:] = saved_counts
                current_list = saved_list
                current_set = saved_set

    # Final polish: try single-element swaps
    if time.time() < deadline - 3:
        saved_counts = sum_counts[:]
        saved_list = current_list[:]
        saved_set = set(current_set)
        
        for i in range(len(current_list)):
            if time.time() > deadline - 3:
                break
            x_remove = current_list[i]
            remove(x_remove, current_list, current_set)
            
            candidates = list(set(range(1, N + 1)) - current_set)
            rng.shuffle(candidates)
            added = False
            for x in candidates:
                if can_add(x, current_list):
                    add(x, current_list, current_set)
                    added = True
                    break
            
            if not added:
                # Revert
                sum_counts[:] = saved_counts
                current_list = saved_list
                current_set = saved_set
            else:
                if len(current_list) > best_size:
                    best_size = len(current_list)
                    best_result = list(current_set)
                else:
                    sum_counts[:] = saved_counts
                    current_list = saved_list
                    current_set = saved_set

    # Final validation
    def _validate(chosen):
        if len(chosen) != len(set(chosen)):
            return False
        if any(x < 1 or x > 1000 for x in chosen):
            return False
        counts = {}
        n = len(chosen)
        for i in range(n):
            for j in range(i, n):
                s = chosen[i] + chosen[j]
                counts[s] = counts.get(s, 0) + 1
                if counts[s] > 2:
                    return False
        return True

    if not _validate(best_result):
        # Fallback to pure ascending greedy
        best_result = run_greedy(0, list(range(1, N + 1)))

    return best_result
# EVOLVE-BLOCK-END
