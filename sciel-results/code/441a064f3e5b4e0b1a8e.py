import random
import time

def construct():
    start_time = time.time()
    timeout = 145.0
    
    prescribed = {1, 2, 5, 11}
    max_val = 284
    target_size = 20
    
    def is_valid_sidon(s):
        sums = set()
        # s is a set, iterate
        # To speed up, we can sort? Sorting is O(K log K).
        # Iterating pairs is O(K^2).
        # For K=20, 190 pairs.
        # Creating set of sums and checking is fast.
        elems = list(s)
        n = len(elems)
        for i in range(n):
            xi = elems[i]
            for j in range(i, n):
                val = xi + elems[j]
                if val in sums:
                    return False
                sums.add(val)
        return True

    def greedy_init():
        s = set(prescribed)
        for x in range(1, max_val + 1):
            if x in s:
                continue
            # Quick check: if 2*x is in sums of s? 
            # Just use full check for simplicity, it's fast.
            if is_valid_sidon(s | {x}):
                s.add(x)
        return s

    curr_set = greedy_init()
    best_set = set(curr_set)
    
    if len(curr_set) >= target_size:
        return sorted(list(curr_set))

    # Maintain a list of unused numbers for faster random selection
    unused = [x for x in range(1, max_val + 1) if x not in curr_set]
    
    while time.time() - start_time < timeout:
        move_made = False
        
        # Try to add
        if unused:
            # Try a few random additions
            for _ in range(5):
                idx = random.randint(0, len(unused) - 1)
                x = unused[idx]
                if is_valid_sidon(curr_set | {x}):
                    curr_set.add(x)
                    # Remove from unused list
                    unused[idx] = unused[-1]
                    unused.pop()
                    move_made = True
                    if len(curr_set) > len(best_set):
                        best_set = set(curr_set)
                        if len(curr_set) >= target_size:
                            return sorted(list(curr_set))
                    break
        
        if not move_made:
            # Try to swap
            # Get list of removable elements
            removable = [x for x in curr_set if x not in prescribed]
            if removable and unused:
                y = random.choice(removable)
                
                # Try a few random swaps
                for _ in range(10):
                    idx = random.randint(0, len(unused) - 1)
                    x = unused[idx]
                    if x == y:
                        continue
                    new_set = (curr_set - {y}) | {x}
                    if is_valid_sidon(new_set):
                        curr_set = new_set
                        # Update unused list: y is now unused, x is now used
                        # Remove x from unused, add y to unused
                        unused[idx] = y
                        move_made = True
                        if len(curr_set) > len(best_set):
                            best_set = set(curr_set)
                            if len(curr_set) >= target_size:
                                return sorted(list(curr_set))
                        break

    return sorted(list(best_set))
