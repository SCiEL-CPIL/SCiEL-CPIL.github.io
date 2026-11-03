import time
import random
import math

def _precompute_neighbors(q, n, radius):
    total = q ** n
    # For radius 1, neighbors are the word itself and words differing in 1 position
    neigh = [[] for _ in range(total)]
    neigh_set = [set() for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
                    neigh_set[idx].add(new_idx)
    return neigh, neigh_set, total

def _greedy_cover(neigh, total, deadline, rng):
    counts = [0] * total
    code = []
    remaining = [True] * total
    rem_count = total
    
    def add_to_code(c):
        nonlocal rem_count
        code.append(c)
        for u in neigh[c]:
            counts[u] += 1
            if counts[u] == 1:
                remaining[u] = False
                rem_count -= 1

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
    return code, counts

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, neigh_set, _ = _precompute_neighbors(q, n, radius)
    
    # 2. Greedy Initialization
    code, counts = _greedy_cover(neigh, total, deadline - 10.0, rng)
    
    best_code = code[:]
    best_size = len(code)
    
    # 3. Simulated Annealing / Local Search
    temp = 1.0
    cool_rate = 0.999
    min_temp = 0.05
    
    # Helper to check if removing c is valid
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    # Helper to check if swapping c with new_c is valid
    def is_valid_swap(c, new_c):
        # Words at risk are those covered by c but not new_c
        for u in neigh[c]:
            if counts[u] == 1 and u not in neigh_set[new_c]:
                return False
        return True

    # QUBO-inspired energy function for perturbation
    # We want to minimize size, but ensure coverage.
    # Since we maintain validity strictly, we use SA to explore the space of valid covers.
    
    while time.monotonic() < deadline:
        if temp < min_temp:
            temp = min_temp
            
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try Remove
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                c = code[idx]
                if is_removable(c):
                    # Accept removal
                    code.pop(idx)
                    for u in neigh[c]:
                        counts[u] -= 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                        
        elif move_type < 0.85:
            # Try Swap
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                new_c = rng.randint(0, total - 1)
                
                if new_c == old_c or new_c in code:
                    continue
                    
                if is_valid_swap(old_c, new_c):
                    # Accept swap
                    for u in neigh[old_c]:
                        counts[u] -= 1
                    for u in neigh[new_c]:
                        counts[u] += 1
                    code[idx] = new_c
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]

        else:
            # Try Add (Perturbation to escape local minima)
            new_c = rng.randint(0, total - 1)
            if new_c not in code:
                # Accept with prob exp(-1/temp)
                if rng.random() < math.exp(-1.0 / temp):
                    code.append(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1
                    
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
