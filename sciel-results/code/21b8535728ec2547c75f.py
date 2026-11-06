import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    max_n = 1000
    max_sum = 2 * max_n
    
    # counts[s] = number of pairs (a, b) in S with a <= b and a + b = s
    counts = [0] * (max_sum + 1)
    in_set = [False] * (max_n + 1)
    
    def add_element(x):
        # Check if x can be added
        if counts[2 * x] >= 2:
            return False
        
        for a in range(1, x):
            if in_set[a]:
                s = a + x
                if counts[s] >= 2:
                    return False
        
        # If valid, actually add
        in_set[x] = True
        counts[2 * x] += 1
        for a in range(1, x):
            if in_set[a]:
                counts[a + x] += 1
        return True

    def remove_element(x):
        in_set[x] = False
        counts[2 * x] -= 1
        for a in range(1, x):
            if in_set[a]:
                counts[a + x] -= 1

    def get_size():
        return sum(1 for i in range(1, max_n + 1) if in_set[i])

    def save_state():
        return (in_set[:], counts[:])

    def restore_state(state):
        in_set[:], counts[:] = state

    # 1. Greedy Initialization (Ascending)
    for i in range(1, max_n + 1):
        add_element(i)
    
    best_size = get_size()
    best_in_set = in_set[:]
    best_counts = counts[:]
    
    def greedy_fill(order):
        # Try to add elements in the given order if they fit
        for x in order:
            if not in_set[x]:
                valid = True
                if counts[2 * x] >= 2:
                    valid = False
                else:
                    for a in range(1, x):
                        if in_set[a] and counts[a + x] >= 2:
                            valid = False
                            break
                if valid:
                    add_element(x)

    # Main Loop with proper save/restore
    while time.time() - start_time < deadline:
        # Try to add any element that fits
        added_any = False
        for x in range(1, max_n + 1):
            if not in_set[x]:
                if add_element(x):
                    added_any = True
                    break
        
        if added_any:
            if get_size() > best_size:
                best_size = get_size()
                best_in_set = in_set[:]
                best_counts = counts[:]
            continue
        
        # No single element can be added. Try a removal + refill.
        current_size = get_size()
        s_elements = [i for i in range(1, max_n + 1) if in_set[i]]
        
        if not s_elements:
            break
        
        # Pick an element to remove: try a few candidates
        # Strategy: try removing elements that are "high" (block more sums)
        # or random for diversity
        candidates = []
        if random.random() < 0.3:
            candidates = [random.choice(s_elements)]
        else:
            # Try a few random candidates
            n_try = min(5, len(s_elements))
            candidates = random.sample(s_elements, n_try)
        
        improved_this_iter = False
        for x_to_remove in candidates:
            if time.time() - start_time > deadline:
                break
            
            saved = save_state()
            remove_element(x_to_remove)
            
            # Greedy fill with slightly randomized order
            order = list(range(1, max_n + 1))
            if random.random() < 0.5:
                random.shuffle(order)
            else:
                # Ascending with occasional swaps
                for _ in range(10):
                    i = random.randint(0, max_n - 2)
                    order[i], order[i+1] = order[i+1], order[i]
            
            greedy_fill(order)
            
            new_size = get_size()
            
            if new_size > best_size:
                best_size = new_size
                best_in_set = in_set[:]
                best_counts = counts[:]
                improved_this_iter = True
                break
            elif new_size > current_size:
                # Keep this state even if not new best, as it might lead to further improvements
                improved_this_iter = True
                break
            else:
                restore_state(saved)
        
        if not improved_this_iter:
            # If no improvement from any candidate, try a more aggressive move:
            # remove 2 elements and refill
            if len(s_elements) >= 2 and random.random() < 0.5:
                pair = random.sample(s_elements, 2)
                saved = save_state()
                remove_element(pair[0])
                remove_element(pair[1])
                
                order = list(range(1, max_n + 1))
                random.shuffle(order)
                greedy_fill(order)
                
                new_size = get_size()
                if new_size > best_size:
                    best_size = new_size
                    best_in_set = in_set[:]
                    best_counts = counts[:]
                else:
                    restore_state(saved)
    
    # Restore best
    in_set[:] = best_in_set
    counts[:] = best_counts
    
    result = [i for i in range(1, max_n + 1) if in_set[i]]
    return result
