import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Mandatory elements
    base = [1, 2, 5, 11]
    
    # Helper to create sum array
    def make_sums_arr(s_list):
        arr = bytearray(569) # Max sum 284+284=568
        for i, a in enumerate(s_list):
            for b in s_list[i:]:
                arr[a+b] = 1
        return arr

    # Initial state
    initial_arr = make_sums_arr(base)
    initial_max = max(base)
    initial_sum = sum(base)
    
    # State: (s_list, sums_arr, max_elem, sum_elem)
    beam = [(base, initial_arr, initial_max, initial_sum)]
    beam_width = 1000
    
    best_set = set(base)
    best_size = len(base)
    
    while True:
        if time.time() > deadline:
            break
            
        next_beam = []
        
        for s_list, sums_arr, max_elem, sum_elem in beam:
            # Try adding candidates x > max_elem
            for x in range(max_elem + 1, 285):
                conflict = False
                for y in s_list:
                    if sums_arr[x + y]:
                        conflict = True
                        break
                
                if conflict or sums_arr[2 * x]:
                    continue
                
                # Create new state
                new_arr = sums_arr[:]
                for y in s_list:
                    new_arr[x + y] = 1
                new_arr[2 * x] = 1
                
                new_list = s_list + [x]
                new_max = x
                new_sum = sum_elem + x
                
                next_beam.append((new_list, new_arr, new_max, new_sum))
            
        if not next_beam:
            break
            
        # Deduplicate
        seen = {}
        for state in next_beam:
            # Use tuple of list as key? Or frozenset?
            # List is sorted, so tuple is unique for the set.
            key = tuple(state[0])
            if key not in seen:
                seen[key] = state
        
        candidates_to_sort = list(seen.values())
        
        # Sort by max_elem (asc), then sum_elem (asc)
        candidates_to_sort.sort(key=lambda x: (x[2], x[3]))
        
        beam = candidates_to_sort[:beam_width]
        
        # Track best
        for state in beam:
            if len(state[0]) > best_size:
                best_size = len(state[0])
                best_set = set(state[0])
                
        # Optional: Randomization to escape local optima
        # If we have time, we could mix in some random states from the next batch
        # For now, deterministic beam with larger width should be better.
        
    return list(best_set)
