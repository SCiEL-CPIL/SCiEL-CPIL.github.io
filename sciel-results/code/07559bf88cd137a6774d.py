import time
import random
import math
import numpy as np

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

def _greedy_init(q, n, radius, neigh, total, deadline):
    """Greedy set cover initialization with multiple random restarts."""
    best_code = None
    best_size = float('inf')
    
    for restart in range(5):
        rng = random.Random(42 + restart * 1000)
        uncovered = [True] * total
        rem_count = total
        code = []
        code_set = set()
        
        while rem_count > 0:
            if time.monotonic() > deadline - 90.0:
                break
            best_c = -1
            best_gain = 0
            for c in range(total):
                if c in code_set:
                    continue
                gain = 0
                for u in neigh[c]:
                    if uncovered[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1 or best_gain == 0:
                break
            code.append(best_c)
            code_set.add(best_c)
            for u in neigh[best_c]:
                if uncovered[u]:
                    uncovered[u] = False
                    rem_count -= 1
                    
        # Force cover if needed
        if rem_count > 0:
            for u in range(total):
                if uncovered[u]:
                    for c in neigh[u]:
                        if c not in code_set:
                            code.append(c)
                            code_set.add(c)
                            for u2 in neigh[c]:
                                if uncovered[u2]:
                                    uncovered[u2] = False
                                    rem_count -= 1
                            break
                    if rem_count == 0:
                        break
        
        if rem_count == 0 and len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
    
    return best_code if best_code is not None else []

def _local_search(code, neigh, neigh_set, total, deadline, rng):
    """Enhanced local search with targeted swaps and removals."""
    code_set = set(code)
    counts = [0] * total
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
    
    best_code = code[:]
    best_size = len(code)
    
    # Precompute for each codeword, which words it uniquely covers
    unique_cover = {}
    for c in code:
        unique_cover[c] = set()
        for u in neigh[c]:
            if counts[u] == 1:
                unique_cover[c].add(u)
    
    # Phase 1: Aggressive removal of redundant codewords
    changed = True
    while changed:
        changed = False
        for c in code[:]:
            if c in code_set:
                removable = True
                for u in neigh[c]:
                    if counts[u] == 1:
                        removable = False
                        break
                if removable:
                    code.remove(c)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
    
    # Phase 2: Targeted swaps - try to swap codewords that have unique coverage
    # with codewords that cover more of their unique words
    for iteration in range(50):
        if time.monotonic() > deadline - 30.0:
            break
        
        # Find codewords with most unique coverage (candidates for improvement)
        candidates = []
        for c in code:
            if c in code_set and len(unique_cover.get(c, set())) > 0:
                candidates.append((len(unique_cover[c]), c))
        
        if not candidates:
            break
        
        candidates.sort(reverse=True)
        _, old_c = candidates[0]
        unique_words = unique_cover.get(old_c, set())
        
        if not unique_words:
            continue
        
        # Find best swap candidate: maximizes coverage of unique_words
        best_new_c = -1
        best_overlap = 0
        for new_c in range(total):
            if new_c == old_c or new_c in code_set:
                continue
            overlap = len(unique_words & neigh_set[new_c])
            if overlap > best_overlap:
                best_overlap = overlap
                best_new_c = new_c
        
        if best_new_c != -1 and best_overlap > 0:
            # Check if swap is valid
            valid = True
            for u in neigh[old_c]:
                if counts[u] == 1 and u not in neigh_set[best_new_c]:
                    valid = False
                    break
            
            if valid:
                # Perform swap
                for u in neigh[old_c]:
                    counts[u] -= 1
                for u in neigh[best_new_c]:
                    counts[u] += 1
                code[code.index(old_c)] = best_new_c
                code_set.discard(old_c)
                code_set.add(best_new_c)
                
                # Update unique_cover
                unique_cover[best_new_c] = set()
                for u in neigh[best_new_c]:
                    if counts[u] == 1:
                        unique_cover[best_new_c].add(u)
                unique_cover.pop(old_c, None)
                
                # Try to remove more after swap
                for c in code[:]:
                    if c in code_set:
                        removable = True
                        for u in neigh[c]:
                            if counts[u] == 1:
                                removable = False
                                break
                        if removable:
                            code.remove(c)
                            code_set.discard(c)
                            for u in neigh[c]:
                                counts[u] -= 1
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
    
    return best_code

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, neigh_set, _ = _precompute_neighbors(q, n, radius)
    
    # 2. Multiple greedy initializations
    code = _greedy_init(q, n, radius, neigh, total, deadline)
    
    # 3. Local search with multiple passes
    for pass_num in range(3):
        if time.monotonic() > deadline - 40.0:
            break
        code = _local_search(code, neigh, neigh_set, total, deadline, rng)
    
    # 4. Final minimal reduction
    best_counts = [0] * total
    for c in code:
        for u in neigh[c]:
            best_counts[u] += 1
    
    final_code = []
    for c in code:
        removable = True
        for u in neigh[c]:
            if best_counts[u] == 1:
                removable = False
                break
        if not removable:
            final_code.append(c)
            for u in neigh[c]:
                best_counts[u] -= 1
    
    best_code = final_code if len(final_code) < len(code) else code

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
