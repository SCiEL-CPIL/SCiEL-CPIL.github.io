import time
import random
import math

def _precompute_neighbors(q, n, radius):
    total = q ** n
    # For radius 1, neighbors are the word itself and words differing in 1 position
    neigh = [[] for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
    # Precompute sets for faster intersection/subset checks
    neigh_sets = [set(lst) for lst in neigh]
    return neigh, neigh_sets, total

def construct(q, n, radius):
    deadline = time.monotonic() + 110.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, neigh_sets, _ = _precompute_neighbors(q, n, radius)
    
    # 2. Greedy Initialization
    counts = [0] * total
    code = []
    code_set = set(code)
    remaining = [True] * total
    rem_count = total
    
    def add_to_code(c):
        nonlocal rem_count
        code.append(c)
        code_set.add(c)
        for u in neigh[c]:
            counts[u] += 1
            if counts[u] == 1:
                remaining[u] = False
                rem_count -= 1

    def remove_from_code(c):
        nonlocal rem_count
        code.remove(c)
        code_set.remove(c)
        for u in neigh[c]:
            counts[u] -= 1
            if counts[u] == 0:
                remaining[u] = True
                rem_count += 1

    # Greedy selection
    while rem_count > 0:
        if time.monotonic() > deadline:
            break
        best_c = -1
        best_gain = -1
        for c in range(total):
            gain = 0
            for u in neigh[c]:
                if remaining[u]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1:
            break
        add_to_code(best_c)
        
    if rem_count > 0:
        # Fallback
        while rem_count > 0:
            c = rng.randint(0, total - 1)
            if c not in code:
                add_to_code(c)
            else:
                for u in range(total):
                    if remaining[u]:
                        nc = neigh[u][0]
                        if nc not in code:
                            add_to_code(nc)
                            break

    best_code = code[:]
    best_size = len(code)
    
    # 3. Local Search with Kick (Perturbation)
    temp = 5.0
    cool_rate = 0.9995
    min_temp = 0.01
    
    # Helper to check if removing c is valid
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    # Helper to check if swapping c with new_c is valid
    def is_valid_swap(c, new_c):
        neigh_new_c = neigh_sets[new_c]
        for u in neigh[c]:
            if counts[u] == 1:
                if u not in neigh_new_c:
                    return False
        return True

    while time.monotonic() < deadline:
        if temp < min_temp:
            temp = min_temp
            
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try Remove
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                c = code[idx]
                if is_removable(c):
                    # Accept removal
                    code.pop(idx)
                    for u in neigh[c]:
                        counts[u] -= 1
                        if counts[u] == 0:
                            remaining[u] = True
                            rem_count += 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                else:
                    pass
                    
        elif move_type < 0.8:
            # Try Swap
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                new_c = rng.randint(0, total - 1)
                
                if new_c == old_c or new_c in code_set:
                    continue
                    
                if is_valid_swap(old_c, new_c):
                    # Accept swap
                    for u in neigh[old_c]:
                        counts[u] -= 1
                        if counts[u] == 0:
                            remaining[u] = True
                            rem_count += 1
                    for u in neigh[new_c]:
                        counts[u] += 1
                        if counts[u] == 1:
                            remaining[u] = False
                            rem_count -= 1
                    code[idx] = new_c
                    code_set.remove(old_c)
                    code_set.add(new_c)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                else:
                    pass

        else:
            # Try Kick: Remove a random word, then Greedy Repair
            if len(code) > 0:
                # Save state
                saved_code = code[:]
                saved_counts = counts[:]
                saved_remaining = remaining[:]
                saved_rem_count = rem_count
                saved_size = len(code)
                
                # Remove a random word
                idx = rng.randint(0, len(code) - 1)
                c = code[idx]
                code.pop(idx)
                code_set.remove(c)
                for u in neigh[c]:
                    counts[u] -= 1
                    if counts[u] == 0:
                        remaining[u] = True
                        rem_count += 1
                
                # Greedy Repair: Add words to cover remaining
                repair_steps = 0
                max_repair_steps = 10 # Limit repair depth to keep it fast
                while rem_count > 0 and repair_steps < max_repair_steps:
                    best_c = -1
                    best_gain = -1
                    # Only consider words not in code
                    for cw in range(total):
                        if cw in code_set:
                            continue
                        gain = 0
                        for u in neigh[cw]:
                            if remaining[u]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_c = cw
                    if best_c == -1:
                        break
                    # Add best_c
                    code.append(best_c)
                    code_set.add(best_c)
                    for u in neigh[best_c]:
                        counts[u] += 1
                        if counts[u] == 1:
                            remaining[u] = False
                            rem_count -= 1
                    repair_steps += 1
                
                # Check if we improved or stayed same size (and valid)
                if rem_count == 0:
                    if len(code) < saved_size:
                        # Improvement
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                    elif len(code) == saved_size:
                        # Same size, accept to move (if random probability allows, or just always accept to explore)
                        # Always accept to maintain diversity
                        pass
                    else:
                        # Got worse, revert
                        code = saved_code
                        code_set = set(code)
                        counts = saved_counts
                        remaining = saved_remaining
                        rem_count = saved_rem_count
                else:
                    # Failed to repair, revert
                    code = saved_code
                    code_set = set(code)
                    counts = saved_counts
                    remaining = saved_remaining
                    rem_count = saved_rem_count

        temp *= cool_rate

    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
