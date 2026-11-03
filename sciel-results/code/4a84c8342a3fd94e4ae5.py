import time
import random
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    total = q ** n
    
    # 1. Precompute neighbors (Hamming distance <= radius)
    # For q=3, n=6, radius=1: neighbors are the word itself + 2*6 = 13 words
    neigh = [[] for _ in range(total)]
    neigh_set = [set() for _ in range(total)]
    
    for idx in range(total):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        # Iterate through positions
        temp = idx
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
                    neigh_set[idx].add(new_idx)
    
    rng = random.Random(42)
    
    # 2. Greedy Set Cover Initialization with multiple restarts
    best_code = None
    best_size = float('inf')
    
    for restart in range(10):
        if time.time() > deadline - 30.0:
            break
            
        code_set = set()
        counts = [0] * total
        uncovered = [i for i in range(total)]
        
        # Greedy: repeatedly pick codeword covering most uncovered
        while uncovered and time.time() < deadline - 30.0:
            best_c = -1
            best_gain = 0
            for c in range(total):
                if c in code_set:
                    continue
                gain = sum(1 for u in neigh[c] if counts[u] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain == 0:
                break
                
            code_set.add(best_c)
            for u in neigh[best_c]:
                counts[u] += 1
                if counts[u] == 1:
                    uncovered.remove(u)
        
        # If still uncovered, force add
        if uncovered:
            for u in uncovered:
                if u not in code_set:
                    code_set.add(u)
                    for v in neigh[u]:
                        counts[v] += 1
        
        code = list(code_set)
        
        # 3. Aggressive Pruning: repeatedly try to remove any redundant codeword
        removed_any = True
        while removed_any and time.time() < deadline - 20.0:
            removed_any = False
            code_list = list(code_set)
            for c in code_list:
                if c not in code_set:
                    continue
                # Check if c is redundant
                redundant = True
                for u in neigh[c]:
                    if counts[u] == 1:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    removed_any = True
        
        code = list(code_set)
        
        # 4. Simulated Annealing Polish
        temp = 1.0
        cool_rate = 0.9998
        min_temp = 0.005
        
        def is_removable(c):
            for u in neigh[c]:
                if counts[u] == 1:
                    return False
            return True

        def is_valid_swap(c, new_c):
            # Check if removing c leaves any neighbor uncovered by new_c
            for u in neigh[c]:
                if counts[u] == 1 and u not in neigh_set[new_c]:
                    return False
            return True

        while time.time() < deadline:
            if temp < min_temp:
                temp = min_temp
                
            move_type = rng.random()
            
            if move_type < 0.6:
                # Try Remove
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_removable(c):
                        # Perform removal
                        code.pop(idx)
                        code_set.discard(c)
                        for u in neigh[c]:
                            counts[u] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                            
            elif move_type < 0.95:
                # Try Swap
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                        
                    if is_valid_swap(old_c, new_c):
                        # Perform swap
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]

            else:
                # Try Add (Perturbation)
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    if rng.random() < math.exp(-1.0 / temp):
                        code.append(new_c)
                        code_set.add(new_c)
                        for u in neigh[new_c]:
                            counts[u] += 1
                        
            temp *= cool_rate
        
        # Update best
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
    
    # Use the best code found
    final_code = best_code if best_code is not None else list(code_set)
    
    # Convert back to list of lists
    result = []
    for idx in final_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
