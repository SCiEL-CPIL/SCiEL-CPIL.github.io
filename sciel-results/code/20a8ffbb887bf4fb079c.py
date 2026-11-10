import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin
    
    best_set = []
    best_size = 0
    
    # Helper to check if x can be added to current_set with given counts
    def can_add(x, current_set, counts):
        # Check x + y for all y in current_set
        for y in current_set:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        # Check 2*x
        if counts.get(2*x, 0) >= 2:
            return False
        return True

    # Helper to add x to current_set and update counts
    def add_element(x, current_set, in_set, counts):
        current_set.append(x)
        in_set[x] = True
        for y in current_set[:-1]:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        counts[2*x] = counts.get(2*x, 0) + 1

    # Helper to remove x from current_set and update counts
    # Assumes x is in current_set
    def remove_element(x, current_set, in_set, counts):
        # Find index
        idx = -1
        for i, val in enumerate(current_set):
            if val == x:
                idx = i
                break
        if idx == -1:
            return
        
        # Swap with last element to make removal O(1)
        current_set[idx] = current_set[-1]
        current_set.pop()
        in_set[x] = False
        
        # Update counts
        # Decrement for x + y for all y in current_set (which is now S - {x})
        for y in current_set:
            s = x + y
            counts[s] -= 1
        counts[2*x] -= 1

    # Initial greedy pass
    counts = {}
    current_set = []
    in_set = [False] * 1001
    for x in range(1, 1001):
        if can_add(x, current_set, counts):
            add_element(x, current_set, in_set, counts)
            
    best_set = current_set.copy()
    best_size = len(current_set)
    
    num_iterations = 0
    while time.time() < deadline:
        num_iterations += 1
        random.seed(num_iterations)
        
        # Randomized Greedy Construction
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        counts = {}
        current_set = []
        in_set = [False] * 1001
        
        for x in candidates:
            if can_add(x, current_set, counts):
                add_element(x, current_set, in_set, counts)
                
        current_size = len(current_set)
        
        # Local Search Phase
        # Try to improve by swapping and adding
        # We run a fixed number of steps or until time runs out
        for _ in range(3000):
            if time.time() > deadline - 5:
                break
            
            if not current_set:
                break
                
            # Pick random x in set
            x = random.choice(current_set)
            
            # Pick random y not in set
            y = random.randint(1, 1000)
            if in_set[y]:
                continue
            
            # Check if swapping x with y is valid
            # We need to check if y fits in S - {x}
            # S' = S - {x}
            # For all z in S', count_{S'}(y+z) < 2
            # count_{S'}(s) = count_S(s) - (1 if s = x+w for some w in S' else 0)
            
            valid_swap = True
            for z in current_set:
                if z == x:
                    continue
                s = y + z
                # Determine if s was formed by x + w in S
                # s = x + w => w = s - x = y + z - x
                w = y + z - x
                # If w is in S' (i.e., in_set[w] and w != x), then count was 1 higher
                if in_set[w] and w != x:
                    count_in_s_prime = counts.get(s, 0) - 1
                else:
                    count_in_s_prime = counts.get(s, 0)
                
                if count_in_s_prime + 1 > 2:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Check 2*y
                s2 = 2 * y
                w2 = 2 * y - x
                if in_set[w2] and w2 != x:
                    count_in_s_prime_2 = counts.get(s2, 0) - 1
                else:
                    count_in_s_prime_2 = counts.get(s2, 0)
                
                if count_in_s_prime_2 + 1 > 2:
                    valid_swap = False
            
            if valid_swap:
                # Perform Swap
                remove_element(x, current_set, in_set, counts)
                add_element(y, current_set, in_set, counts)
                
                # Try to add any new element
                added = False
                for _ in range(10):
                    z = random.randint(1, 1000)
                    if in_set[z]:
                        continue
                    if can_add(z, current_set, counts):
                        add_element(z, current_set, in_set, counts)
                        added = True
                        break
                
                if added:
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set.copy()
                    
    return best_set
