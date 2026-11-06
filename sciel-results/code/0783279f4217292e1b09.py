import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    N = 1000
    max_sum = 2 * N
    sum_counts = [0] * (max_sum + 1)
    
    def can_add(x, chosen_list):
        if sum_counts[2*x] >= 2:
            return False
        for y in chosen_list:
            if sum_counts[x + y] >= 2:
                return False
        return True

    def add(x, chosen_list, chosen_set):
        sum_counts[2*x] += 1
        for y in chosen_list:
            sum_counts[x + y] += 1
        chosen_list.append(x)
        chosen_set.add(x)

    def remove(x, chosen_list, chosen_set):
        sum_counts[2*x] -= 1
        for y in chosen_list:
            if y != x:
                sum_counts[x + y] -= 1
        for i in range(len(chosen_list)):
            if chosen_list[i] == x:
                chosen_list[i] = chosen_list[-1]
                chosen_list.pop()
                break
        chosen_set.discard(x)

    def run_greedy(seed):
        rng = random.Random(seed)
        for i in range(max_sum + 1):
            sum_counts[i] = 0
        chosen_list = []
        chosen_set = set()
        
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, chosen_list):
                add(x, chosen_list, chosen_set)
                
        return list(chosen_set)

    # 1. Find a good starting set
    best_start = []
    best_start_size = 0
    # Run a few greedy starts
    for s in range(10):
        candidate_set = run_greedy(s)
        if len(candidate_set) > best_start_size:
            best_start_size = len(candidate_set)
            best_start = candidate_set
            
    # Initialize global state with best_start
    for i in range(max_sum + 1):
        sum_counts[i] = 0
    current_list = best_start[:]
    current_set = set(best_start)
    for x in current_list:
        # Re-add to populate sum_counts
        # Note: add() assumes sum_counts is correct for existing elements.
        # We need to rebuild sum_counts from scratch.
        pass
        
    # Rebuild sum_counts properly
    sum_counts = [0] * (max_sum + 1)
    temp_list = []
    temp_set = set()
    for x in current_list:
        add(x, temp_list, temp_set)
    current_list = temp_list
    current_set = temp_set
    
    best_size = len(current_list)
    
    rng = random.Random(123)
    best_result = current_list[:]
    
    while time.time() < deadline:
        # Save state
        saved_counts = sum_counts[:]
        saved_list = current_list[:]
        saved_set = set(current_set)
        
        # Larger kicks for better escape from local optima
        kick_size = rng.randint(2, 4)
        
        # Perform kicks
        if len(current_list) > kick_size:
            indices = rng.sample(range(len(current_list)), kick_size)
            indices.sort(reverse=True)
            for idx in indices:
                u = current_list[idx]
                remove(u, current_list, current_set)
                
        # Greedy Fill with multiple random orderings, keep best
        best_fill_size = 0
        best_fill_list = None
        best_fill_set = None
        best_fill_counts = None
        
        for attempt in range(3):
            # Reset to saved state for each attempt
            sum_counts[:] = saved_counts
            cur_list = saved_list[:]
            cur_set = set(saved_set)
            
            candidates = list(set(range(1, N + 1)) - cur_set)
            rng.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline - 1:
                    break
                if can_add(x, cur_list):
                    add(x, cur_list, cur_set)
            
            if len(cur_list) > best_fill_size:
                best_fill_size = len(cur_list)
                best_fill_list = cur_list[:]
                best_fill_set = set(cur_set)
                best_fill_counts = sum_counts[:]
        
        if best_fill_list is not None:
            if best_fill_size > best_size:
                best_size = best_fill_size
                current_list = best_fill_list
                current_set = best_fill_set
                sum_counts = best_fill_counts
                best_result = current_list[:]
            else:
                # Restore state
                sum_counts[:] = saved_counts
                current_list = saved_list
                current_set = saved_set

    return best_result
# EVOLVE-BLOCK-END
