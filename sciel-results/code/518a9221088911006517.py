"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    best_set = None
    best_size = 0
    
    def count_sums(arr):
        """Return a numpy array of sum counts for the given sorted array."""
        n = len(arr)
        counts = np.zeros(2001, dtype=np.int32)
        for i in range(n):
            for j in range(i, n):
                counts[arr[i] + arr[j]] += 1
        return counts
    
    def can_add(x, arr, counts):
        """Check if x can be added to arr without violating B2[2]."""
        for y in arr:
            if counts[x + y] >= 2:
                return False
        if counts[2 * x] >= 2:
            return False
        return True
    
    def add_element(x, arr, counts):
        """Add x to arr and update counts. Returns new arr (sorted) and counts."""
        for y in arr:
            counts[x + y] += 1
        counts[2 * x] += 1
        arr.append(x)
        arr.sort()
        return arr, counts
    
    def validate(arr):
        """Validate B2[2] property. Returns size if valid, 0 otherwise."""
        n = len(arr)
        if n == 0:
            return 0
        counts = np.zeros(2001, dtype=np.int32)
        for i in range(n):
            for j in range(i, n):
                s = arr[i] + arr[j]
                counts[s] += 1
                if counts[s] > 2:
                    return 0
        return n
    
    def greedy_fill(arr, counts, order=None):
        """Greedily fill the set by trying numbers in the given order."""
        if order is None:
            order = list(range(1, 1001))
        arr = list(arr)
        arr.sort()
        if not arr:
            counts = np.zeros(2001, dtype=np.int32)
        in_set = set(arr)
        
        for x in order:
            if time.time() > deadline:
                break
            if x in in_set:
                continue
            if can_add(x, arr, counts):
                arr, counts = add_element(x, arr, counts)
                in_set.add(x)
        return arr, counts
    
    def local_search_add(arr, counts):
        """Try to add more elements that fit."""
        arr = list(arr)
        arr.sort()
        if not arr:
            counts = np.zeros(2001, dtype=np.int32)
        in_set = set(arr)
        
        for x in range(1, 1001):
            if time.time() > deadline:
                break
            if x in in_set:
                continue
            if can_add(x, arr, counts):
                arr, counts = add_element(x, arr, counts)
                in_set.add(x)
        return arr, counts
    
    def try_swap_improve(arr, counts, max_swaps=100):
        """Try to improve by swapping elements."""
        arr = list(arr)
        arr.sort()
        if not arr:
            counts = np.zeros(2001, dtype=np.int32)
        in_set = set(arr)
        current_size = len(arr)
        swaps_done = 0
        
        candidates = [x for x in range(1, 1001) if x not in in_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline or swaps_done >= max_swaps:
                break
            
            # Try to find an element y to remove so x can be added
            # For each y in in_set, check if arr without y + x is valid
            for y in list(in_set):
                # Temporarily check: can x be added to arr without y?
                # New sums from x: x+z for z in arr\{y}, and 2x
                # We need: for all z in arr\{y}, counts[x+z] - (1 if x+z was formed by y+z or y+x) + 1 <= 2
                # This is complex. Let's just try: remove y, check if x fits, add x
                # Remove y
                arr_temp = [z for z in arr if z != y]
                counts_temp = np.zeros(2001, dtype=np.int32)
                for i in range(len(arr_temp)):
                    for j in range(i, len(arr_temp)):
                        counts_temp[arr_temp[i] + arr_temp[j]] += 1
                
                # Check if x can be added
                valid = True
                for z in arr_temp:
                    if counts_temp[x + z] >= 2:
                        valid = False
                        break
                if valid and counts_temp[2 * x] >= 2:
                    valid = False
                if valid:
                    # Do the swap
                    arr = arr_temp + [x]
                    arr.sort()
                    in_set.remove(y)
                    in_set.add(x)
                    counts = count_sums(arr)
                    swaps_done += 1
                    break
        return arr, counts
    
    # Strategy 1: Standard greedy from 1
    arr1, counts1 = greedy_fill([], None)
    sz1 = validate(arr1)
    if sz1 > best_size:
        best_set = arr1
        best_size = sz1
    
    # Strategy 2: Local search add on greedy result
    if best_set and time.time() < deadline:
        arr2, counts2 = local_search_add(best_set, None)
        sz2 = validate(arr2)
        if sz2 > best_size:
            best_set = arr2
            best_size = sz2
    
    # Strategy 3: Multiple random restarts
    for restart in range(30):
        if time.time() > deadline:
            break
        
        strategy = restart % 3
        if strategy == 0:
            # Random order
            order = list(range(1, 1001))
            random.shuffle(order)
            arr3, counts3 = greedy_fill([], None, order)
        elif strategy == 1:
            # Start with random valid seed, then fill
            seed_size = random.randint(5, 20)
            seed = random.sample(range(1, 101), seed_size)
            seed.sort()
            seed_counts = np.zeros(2001, dtype=np.int32)
            valid_seed = True
            for i in range(len(seed)):
                for j in range(i, len(seed)):
                    s = seed[i] + seed[j]
                    seed_counts[s] += 1
                    if seed_counts[s] > 2:
                        valid_seed = False
                        break
                if not valid_seed:
                    break
            if not valid_seed:
                continue
            arr_seed = list(seed)
            counts_seed = seed_counts
            order = [x for x in range(1, 1001) if x not in seed]
            random.shuffle(order)
            arr3, counts3 = greedy_fill(arr_seed, counts_seed, order)
        else:
            # Greedy with random skips in early range
            order = list(range(1, 1001))
            for i in range(min(30, len(order))):
                if random.random() < 0.4:
                    order[i] = -1
            order = [x for x in order if x > 0]
            arr3, counts3 = greedy_fill([], None, order)
        
        sz3 = validate(arr3)
        if sz3 > best_size:
            best_set = arr3
            best_size = sz3
        
        # Local search add
        if sz3 > 0 and time.time() < deadline:
            arr4, counts4 = local_search_add(arr3, None)
            sz4 = validate(arr4)
            if sz4 > best_size:
                best_set = arr4
                best_size = sz4
        
        # Try swap improvement
        if sz4 > best_size and time.time() < deadline:
            arr5, counts5 = try_swap_improve(arr4, None, max_swaps=20)
            sz5 = validate(arr5)
            if sz5 > best_size:
                best_set = arr5
                best_size = sz5
    
    if best_set is None:
        return [1]
    return best_set

# EVOLVE-BLOCK-END
