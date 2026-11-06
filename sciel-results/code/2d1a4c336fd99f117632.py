import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 1000
    MAX_SUM = 2 * N

    def is_valid_add(S_arr, counts, x, buffer):
        if len(S_arr) == 0:
            return counts[2*x] < 2
        
        # Reuse buffer
        np.add(x, S_arr, out=buffer)
        sums = buffer[:len(S_arr)]
        
        if counts[2*x] >= 2:
            return False
        if np.any(counts[sums] >= 2):
            return False
        return True

    def add_element(S_set, S_arr, counts, x):
        S_set.add(x)
        if len(S_arr) > 0:
            # Need to update counts for x + S_arr
            # We can use the same buffer logic or just do it
            sums = x + S_arr
            counts[sums] += 1
        counts[2*x] += 1
        return S_set, np.append(S_arr, x), counts

    def remove_element(S_set, S_arr, counts, x):
        S_set.discard(x)
        mask = S_arr != x
        new_S_arr = S_arr[mask]
        
        if len(new_S_arr) > 0:
            sums = x + new_S_arr
            counts[sums] -= 1
        counts[2*x] -= 1
        return S_set, new_S_arr, counts

    def greedy_init(order, rng):
        S_set = set()
        S_arr = np.array([], dtype=np.int32)
        counts = np.zeros(MAX_SUM + 1, dtype=np.uint8)
        
        for x in order:
            if is_valid_add(S_arr, counts, x, buffer):
                S_set, S_arr, counts = add_element(S_set, S_arr, counts, x)
        return S_set, S_arr, counts

    # Allocate buffer for sums (max size N)
    buffer = np.zeros(N, dtype=np.int32)

    def local_search(S_set, S_arr, counts, max_iters, rng):
        best_S_set = S_set.copy()
        best_S_arr = S_arr.copy()
        best_counts = counts.copy()
        best_size = len(S_set)
        
        current_size = len(S_set)
        all_elements = np.arange(1, N + 1, dtype=np.int32)
        
        for i in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            if current_size < 2:
                break

            # 1. Try to add a random element
            if rng.random() < 0.2:
                candidates = all_elements.copy()
                rng.shuffle(candidates)
                added = False
                for x in candidates:
                    if x not in S_set:
                        if is_valid_add(S_arr, counts, x, buffer):
                            S_set, S_arr, counts = add_element(S_set, S_arr, counts, x)
                            current_size += 1
                            added = True
                            if current_size > best_size:
                                best_size = current_size
                                best_S_set = S_set.copy()
                                best_S_arr = S_arr.copy()
                                best_counts = counts.copy()
                            break
                if added:
                    continue

            # 2. Swap move
            x_idx = rng.randrange(current_size)
            x = S_arr[x_idx]
            
            S_set, S_arr, counts = remove_element(S_set, S_arr, counts, x)
            current_size -= 1
            
            candidates = all_elements.copy()
            rng.shuffle(candidates)
            
            found = False
            for y in candidates:
                if y not in S_set and y != x:
                    if is_valid_add(S_arr, counts, y, buffer):
                        S_set, S_arr, counts = add_element(S_set, S_arr, counts, y)
                        current_size += 1
                        found = True
                        if current_size > best_size:
                            best_size = current_size
                            best_S_set = S_set.copy()
                            best_S_arr = S_arr.copy()
                            best_counts = counts.copy()
                        break
                        
            if not found:
                S_set, S_arr, counts = add_element(S_set, S_arr, counts, x)
                current_size += 1
        
        return best_S_set, best_S_arr, best_counts

    def final_polish(S_set, S_arr, counts):
        added = True
        while added and time.time() < deadline - 2:
            added = False
            for x in range(1, N + 1):
                if x not in S_set:
                    if is_valid_add(S_arr, counts, x, buffer):
                        S_set, S_arr, counts = add_element(S_set, S_arr, counts, x)
                        added = True
                        break
        return S_set, S_arr, counts

    rng = random.Random(42)
    
    best_size = 0
    best_S_set = set()
    best_S_arr = np.array([], dtype=np.int32)
    best_counts = np.zeros(MAX_SUM + 1, dtype=np.uint8)
    
    # Phase 1: Multiple Greedy Initializations
    num_trials = 10
    for trial in range(num_trials):
        if time.time() > deadline - 40:
            break
        
        if trial == 0:
            order = list(range(1, N + 1))
        else:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        
        S_set, S_arr, counts = greedy_init(order, rng)
        size = len(S_set)
        
        if size > best_size:
            best_size = size
            best_S_set = S_set.copy()
            best_S_arr = S_arr.copy()
            best_counts = counts.copy()
            
    # Phase 2: Local Search
    S_set, S_arr, counts = best_S_set, best_S_arr, best_counts.copy()
    
    # Run local search
    # We can run multiple rounds if time permits
    # Let's run 10000 iterations
    S_set, S_arr, counts = local_search(S_set, S_arr, counts, 10000, rng)
    
    # Phase 3: Final Polish
    S_set, S_arr, counts = final_polish(S_set, S_arr, counts)
    
    return list(S_set)
