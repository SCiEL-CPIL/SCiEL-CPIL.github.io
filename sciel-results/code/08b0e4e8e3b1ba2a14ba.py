import time
import random
import math
from collections import deque

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
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, neigh_sets, _ = _precompute_neighbors(q, n, radius)
    
    # 2. Greedy Initialization
    # We want to start with a reasonably good cover.
    # Greedy: pick word covering most uncovered words.
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
    
    # 3. Local Search with 2-opt and restarts
    # Helper to check if removing c is valid
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    def try_remove(c):
        nonlocal best_size, best_code
        if is_removable(c):
            code.remove(c)
            code_set.remove(c)
            for u in neigh[c]:
                counts[u] -= 1
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
            return True
        return False

    def try_swap(old_c, new_c):
        # Check if swap is valid
        neigh_new_c = neigh_sets[new_c]
        for u in neigh[old_c]:
            if counts[u] == 1:
                if u not in neigh_new_c:
                    return False
        # Perform swap
        idx = code.index(old_c)
        code[idx] = new_c
        code_set.remove(old_c)
        code_set.add(new_c)
        for u in neigh[old_c]:
            counts[u] -= 1
        for u in neigh[new_c]:
            counts[u] += 1
        return True

    def local_search_pass():
        improved = True
        while improved:
            improved = False
            # Try removing each codeword
            for c in code[:]:
                if try_remove(c):
                    improved = True
                    break
            if improved:
                continue
            # Try swapping each codeword with a random new one
            for old_c in code[:]:
                for _ in range(50):  # Try 50 random swaps per codeword
                    new_c = rng.randint(0, total - 1)
                    if new_c == old_c or new_c in code_set:
                        continue
                    if try_swap(old_c, new_c):
                        improved = True
                        break
                if improved:
                    break
    
    # Run local search multiple times with perturbations
    for restart in range(20):
        if time.monotonic() > deadline - 5.0:
            break
        
        local_search_pass()
        
        # Perturb: add some random codewords to escape local minimum
        if restart > 0:
            for _ in range(3):
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    code.append(new_c)
                    code_set.add(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1
    
    # Final polish: try to remove as many as possible
    local_search_pass()
    
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
