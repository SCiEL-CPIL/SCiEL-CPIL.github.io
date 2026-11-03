import time
import random
import math

def _precompute_neighbors(q, n, radius):
    total = q ** n
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
    return neigh, total

def construct(q, n, radius):
    deadline = time.monotonic() + 110.0
    total = q ** n
    rng = random.Random(42)
    
    neigh, _ = _precompute_neighbors(q, n, radius)
    # Convert to sets for fast intersection
    neigh_sets = [set(l) for l in neigh]
    
    # Greedy Initialization
    counts = [0] * total
    code = []
    code_set = set()
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

    while rem_count > 0:
        if time.monotonic() > deadline:
            break
        best_c = -1
        best_gain = -1
        for c in range(total):
            if c in code_set:
                continue
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
            if c not in code_set:
                add_to_code(c)

    best_code = code[:]
    best_size = len(code)
    
    def is_redundant(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    def cleanup():
        changed = True
        while changed:
            changed = False
            to_remove = []
            for c in code:
                if is_redundant(c):
                    to_remove.append(c)
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                for u in neigh[c]:
                    counts[u] -= 1
                changed = True
            if changed:
                continue
            break

    def find_swap_target(c):
        # Find unique words covered by c
        U = set()
        for u in neigh[c]:
            if counts[u] == 1:
                U.add(u)
        
        if not U:
            return None # Redundant
            
        # Intersection of neighbors of all u in U
        first_u = next(iter(U))
        candidates = neigh_sets[first_u].copy()
        
        for u in U:
            if u == first_u:
                continue
            candidates.intersection_update(neigh_sets[u])
            
        # Remove current code
        candidates.difference_update(code_set)
        
        if not candidates:
            return None
            
        # Heuristic: Pick candidate that maximizes overlap with code?
        # Or just pick one.
        # Let's pick one that is "central" or just random from candidates
        # To be deterministic-ish for debugging, pick smallest index?
        # Or random to escape cycles.
        return rng.choice(list(candidates))

    def try_swaps():
        swapped = True
        while swapped:
            swapped = False
            for c in code[:]:
                if is_redundant(c):
                    code.remove(c)
                    code_set.remove(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    swapped = True
                    continue
                
                target = find_swap_target(c)
                if target is not None:
                    # Perform swap
                    code.remove(c)
                    code_set.remove(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    
                    code.append(target)
                    code_set.add(target)
                    for u in neigh[target]:
                        counts[u] += 1
                    
                    swapped = True
                    break # Restart loop to re-evaluate
                    
            if swapped:
                cleanup()

    while time.monotonic() < deadline:
        cleanup()
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        # Try to improve
        # If we don't make progress in try_swaps, we are stuck.
        # We can check size before and after.
        size_before = len(code)
        try_swaps()
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        # Perturbation
        # If we are stuck (size didn't decrease), add a random word
        # This increases size, but cleanup/swaps might find a better configuration
        if len(code) == size_before:
             # Stuck
             # Add random word
             new_c = rng.randint(0, total - 1)
             if new_c not in code_set:
                 code.append(new_c)
                 code_set.add(new_c)
                 for u in neigh[new_c]:
                     counts[u] += 1
                 
    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
