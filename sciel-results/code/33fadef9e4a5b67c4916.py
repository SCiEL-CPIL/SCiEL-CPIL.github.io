import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    mandatory = {1, 2, 5, 11}
    best_size = 0
    best_set = set(mandatory)
    
    def get_sums(s):
        sums = set()
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums.add(lst[i] + lst[j])
        return sums

    def is_valid_addition(s, sums, x):
        if x in s:
            return False
        if 2 * x in sums:
            return False
        for y in s:
            if x + y in sums:
                return False
        return True

    def update_sums_add(s, sums, x):
        for y in s:
            sums.add(x + y)
        sums.add(2 * x)
        s.add(x)

    def update_sums_remove(s, sums, x):
        s.remove(x)
        for y in s:
            sums.discard(x + y)
        sums.discard(2 * x)

    def greedy_construct():
        s = set(mandatory)
        sums = get_sums(s)
        candidates = list(range(1, 285))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in s:
                continue
            if is_valid_addition(s, sums, x):
                update_sums_add(s, sums, x)
        return s, sums

    def simulated_annealing(initial_s, initial_sums):
        s = set(initial_s)
        sums = set(initial_sums)
        current_size = len(s)
        best_local_size = current_size
        best_local_s = set(s)
        
        T = 5.0
        cooling = 0.99
        
        # We run SA for a fixed number of steps or time
        steps = 0
        max_steps = 5000 # Adjust based on speed
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add
                x = random.randint(1, 284)
                if x not in s and is_valid_addition(s, sums, x):
                    update_sums_add(s, sums, x)
                    current_size += 1
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local_s = set(s)
            elif move_type < 0.8:
                # Try to remove
                # Pick a random element from s
                # Prefer non-mandatory to avoid breaking constraints unnecessarily? 
                # Actually, removing mandatory is invalid? No, we must contain mandatory.
                # So we can only remove non-mandatory.
                non_mandatory = [y for y in s if y not in mandatory]
                if non_mandatory:
                    y = random.choice(non_mandatory)
                    # Accept removal with prob exp(1/T)
                    if random.random() < math.exp(1.0 / T):
                        update_sums_remove(s, sums, y)
                        current_size -= 1
            else:
                # Try to swap
                non_mandatory = [y for y in s if y not in mandatory]
                if non_mandatory:
                    y = random.choice(non_mandatory)
                    x = random.randint(1, 284)
                    if x not in s and x != y:
                        # Check validity of swap
                        # Remove y from sums temporarily
                        sums_temp = set(sums)
                        for z in s:
                            sums_temp.discard(y + z)
                        sums_temp.discard(2 * y)
                        
                        if is_valid_addition(s, sums_temp, x):
                            # Apply swap
                            for z in s:
                                sums.discard(y + z)
                            sums.discard(2 * y)
                            for z in s:
                                sums.add(x + z)
                            sums.add(2 * x)
                            s.add(x)
                            # Size unchanged, so no update to best_local_size unless we want to track structure? 
                            # No, best is defined by size.
            
            T *= cooling

        return best_local_s, get_sums(best_local_s)

    # Main Search Loop
    while time.time() < deadline:
        s, sums = greedy_construct()
        best_s, best_sums = simulated_annealing(s, sums)
        
        if len(best_s) > best_size:
            best_size = len(best_s)
            best_set = set(best_s)
            
            if best_size == 20:
                break

    return sorted(list(best_set))
