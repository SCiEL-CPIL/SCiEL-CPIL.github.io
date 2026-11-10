import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s buffer
    
    def is_sidon(S):
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(S):
        if not S:
            return False
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                sums.add(a + b)
        
        S_set = set(S)
        for y in range(1, N + 1):
            if y in S_set:
                continue
            can_add = True
            for a in S:
                if (y + a) in sums:
                    can_add = False
                    break
            if can_add and (2 * y) in sums:
                can_add = False
            if can_add:
                return False
        return True

    def greedy_random():
        selected = []
        sums = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            added_sums = [x + a for a in selected] + [2 * x]
            if len(set(added_sums)) == len(added_sums) and not sums.intersection(added_sums):
                selected.append(x)
                sums.update(added_sums)
        
        return selected

    best_set = greedy_random()
    best_size = len(best_set)
    
    # Try multiple restarts
    for restart in range(200):
        if time.time() > deadline:
            break
            
        current_set = greedy_random()
        
        local_improved = True
        while local_improved and time.time() < deadline:
            local_improved = False
            
            # Try removals
            indices = list(range(len(current_set)))
            random.shuffle(indices)
            
            for idx in indices:
                x = current_set[idx]
                S_prime = [v for i, v in enumerate(current_set) if i != idx]
                
                if is_maximal(S_prime):
                    current_set = S_prime
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    local_improved = True
                    break
                    
            if local_improved:
                continue
                
            # Try swaps
            swap_attempts = 0
            max_swaps = 100 # Increased slightly to allow more exploration
            
            while not local_improved and swap_attempts < max_swaps and time.time() < deadline:
                swap_attempts += 1
                if not current_set:
                    break
                idx_x = random.randint(0, len(current_set) - 1)
                x = current_set[idx_x]
                
                S_set = set(current_set)
                # Optimization: Precompute missing elements? 
                # No, just filter.
                candidates_y = [v for v in range(1, N + 1) if v not in S_set]
                if not candidates_y:
                    break
                y = random.choice(candidates_y)
                
                S_swap = [v for i, v in enumerate(current_set) if i != idx_x]
                S_swap.append(y)
                
                if is_sidon(S_swap):
                    if is_maximal(S_swap):
                        current_set = S_swap
                        local_improved = True
                        break

    return best_set
