import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s margin
    
    max_val = 4096
    max_size = 40  # Theoretical max
    
    best_set = []
    best_size = 0
    
    def get_all_sums(S):
        sums = set()
        n = len(S)
        for i in range(n):
            x = S[i]
            sums.add(3 * x)
            for j in range(i + 1, n):
                y = S[j]
                sums.add(2 * x + y)
                sums.add(x + 2 * y)
                for k in range(j + 1, n):
                    z = S[k]
                    sums.add(x + y + z)
        return sums

    def check_add(x, S, sum_set):
        """Check if x can be added to S. Returns (valid, new_sums)."""
        new_sums = set()
        new_sums.add(3 * x)
        n = len(S)
        for i in range(n):
            y = S[i]
            new_sums.add(2 * x + y)
            new_sums.add(x + 2 * y)
        for i in range(n):
            y = S[i]
            for j in range(i + 1, n):
                z = S[j]
                new_sums.add(x + y + z)
        
        # Check collisions with existing sums
        for s in new_sums:
            if s in sum_set:
                return False, None
        
        # Check internal collisions
        expected_count = 1 + 2 * n + n * (n - 1) // 2
        if len(new_sums) != expected_count:
            return False, None
        
        return True, new_sums

    def remove_sums_involving(x, S, sum_set):
        """Remove sums involving x from sum_set. S contains x."""
        sums_to_remove = set()
        sums_to_remove.add(3 * x)
        others = [y for y in S if y != x]
        for y in others:
            sums_to_remove.add(2 * x + y)
            sums_to_remove.add(x + 2 * y)
        for i in range(len(others)):
            y = others[i]
            for j in range(i + 1, len(others)):
                z = others[j]
                sums_to_remove.add(x + y + z)
        
        for s in sums_to_remove:
            sum_set.discard(s)

    def greedy_build():
        S = []
        sum_set = set()
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= max_size:
                break
            if x in S:
                continue
            valid, new_sums = check_add(x, S, sum_set)
            if valid:
                S.append(x)
                S.sort()
                sum_set.update(new_sums)
        return S, sum_set

    # Initial greedy
    current_set, current_sum_set = greedy_build()
    current_size = len(current_set)
    
    if current_size > best_size:
        best_size = current_size
        best_set = current_set[:]

    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.6:
            # Try to add
            if current_size < max_size:
                x = random.randint(1, max_val)
                if x not in current_set:
                    valid, new_sums = check_add(x, current_set, current_sum_set)
                    if valid:
                        current_set.append(x)
                        current_set.sort()
                        current_sum_set.update(new_sums)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_set[:]
                            T = max(T, 1.0) # Reheat slightly or keep
                            
        elif move_type < 0.9:
            # Try swap
            if current_size > 0:
                y = random.choice(current_set)
                x = random.randint(1, max_val)
                if x == y or x in current_set:
                    continue
                
                # Temporarily remove y
                # We need to check if x is valid in S \ {y}
                # S \ {y} sums = current_sum_set - sums_involving(y)
                
                # Compute sums involving y
                sums_y = set()
                sums_y.add(3 * y)
                others = [z for z in current_set if z != y]
                for z in others:
                    sums_y.add(2 * y + z)
                    sums_y.add(y + 2 * z)
                for i in range(len(others)):
                    z1 = others[i]
                    for j in range(i + 1, len(others)):
                        z2 = others[j]
                        sums_y.add(y + z1 + z2)
                
                # Check x against current_sum_set \ sums_y
                # And check internal validity of x in S\{y}
                
                # To avoid full check_add which is O(N^2), we can just do it.
                # N is small.
                
                S_temp = [z for z in current_set if z != y]
                
                # Check collisions of x with S_temp
                new_sums_x = set()
                new_sums_x.add(3 * x)
                for z in S_temp:
                    new_sums_x.add(2 * x + z)
                    new_sums_x.add(x + 2 * z)
                for i in range(len(S_temp)):
                    z1 = S_temp[i]
                    for j in range(i + 1, len(S_temp)):
                        z2 = S_temp[j]
                        new_sums_x.add(x + z1 + z2)
                
                collides = False
                for s in new_sums_x:
                    if s in current_sum_set and s not in sums_y:
                        collides = True
                        break
                
                if not collides:
                    # Check internal consistency
                    expected_count = 1 + 2 * len(S_temp) + len(S_temp) * (len(S_temp) - 1) // 2
                    if len(new_sums_x) == expected_count:
                        # Accept swap
                        # Remove y sums
                        remove_sums_involving(y, current_set, current_sum_set)
                        current_set.remove(y)
                        
                        # Add x
                        current_set.append(x)
                        current_set.sort()
                        current_sum_set.update(new_sums_x)
                        
        else:
            # Try removal (to escape local optima)
            if current_size > 1:
                y = random.choice(current_set)
                # Accept removal with probability
                if random.random() < 0.1 * T:
                    remove_sums_involving(y, current_set, current_sum_set)
                    current_set.remove(y)
                    current_size -= 1
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final polish
    if time.time() < deadline - 1:
        current_set = best_set[:]
        current_sum_set = get_all_sums(current_set)
        for x in range(1, max_val + 1):
            if time.time() > deadline - 1:
                break
            if x in current_set:
                continue
            if len(current_set) >= max_size:
                break
            valid, new_sums = check_add(x, current_set, current_sum_set)
            if valid:
                current_set.append(x)
                current_set.sort()
                current_sum_set.update(new_sums)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]

    return best_set
