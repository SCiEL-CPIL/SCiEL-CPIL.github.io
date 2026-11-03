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
    
    # 3. Simulated Annealing / Local Search
    temp = 8.0
    cool_rate = 0.9997
    min_temp = 0.05
    
    # Helper to check if removing c is valid
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    # Helper to check if swapping c with new_c is valid
    # Returns True if valid
    def is_valid_swap(c, new_c):
        # A word u is at risk if it is covered by c but not new_c, and counts[u] == 1.
        # We only need to check words that lose coverage.
        # These are u in neigh[c] such that u NOT in neigh[new_c].
        # Using sets for O(1) membership
        neigh_new_c = neigh_sets[new_c]
        for u in neigh[c]:
            if counts[u] == 1:
                if u not in neigh_new_c:
                    return False
        return True

    # Generate candidate swap positions close to old_c
    def get_close_candidates(old_c, num_candidates=5):
        """Get codeword indices that are Hamming distance 1 from old_c."""
        candidates = []
        for i in range(n):
            pow_q_i = q ** i
            digit = (old_c // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = old_c + (new_digit - digit) * pow_q_i
                    candidates.append(new_idx)
        return candidates

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
                else:
                    # Reject
                    pass
                    
        elif move_type < 0.85:
            # Try Swap - use close candidates first, then random
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                
                # Try close candidates first
                tried = set()
                close_cands = get_close_candidates(old_c, 5)
                rng.shuffle(close_cands)
                
                new_c = None
                for cand in close_cands:
                    if cand not in code_set and cand != old_c:
                        new_c = cand
                        break
                
                if new_c is None:
                    # Fall back to random
                    new_c = rng.randint(0, total - 1)
                
                if new_c == old_c or new_c in code_set:
                    continue
                    
                if is_valid_swap(old_c, new_c):
                    # Accept swap
                    # Remove old_c
                    for u in neigh[old_c]:
                        counts[u] -= 1
                    # Add new_c
                    for u in neigh[new_c]:
                        counts[u] += 1
                    code[idx] = new_c
                    code_set.remove(old_c)
                    code_set.add(new_c)
                else:
                    pass

        else:
            # Try Add (Perturbation to escape local minima)
            new_c = rng.randint(0, total - 1)
            if new_c not in code_set:
                # Accept with prob exp(-1/temp)
                if rng.random() < math.exp(-1.0 / temp):
                    code.append(new_c)
                    code_set.add(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1
                    # Note: we do not update best_code here as size increased
                    
        temp *= cool_rate

    # 4. Deterministic polish: repeatedly remove any removable codeword
    # This ensures we reach a minimal cover
    polish_deadline = time.monotonic() + 10.0
    changed = True
    while changed and time.monotonic() < polish_deadline:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            if is_removable(c):
                code.pop(i)
                for u in neigh[c]:
                    counts[u] -= 1
                changed = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]

    # 5. Second SA pass with lower temperature for fine-tuning
    if time.monotonic() < deadline - 5.0:
        temp = 2.0
        cool_rate = 0.9998
        min_temp = 0.02
        
        while time.monotonic() < deadline - 5.0:
            if temp < min_temp:
                temp = min_temp
                
            move_type = rng.random()
            
            if move_type < 0.55:
                # Try Remove
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_removable(c):
                        code.pop(idx)
                        for u in neigh[c]:
                            counts[u] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                            
            elif move_type < 0.9:
                # Try Swap
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    
                    close_cands = get_close_candidates(old_c, 5)
                    rng.shuffle(close_cands)
                    
                    new_c = None
                    for cand in close_cands:
                        if cand not in code_set and cand != old_c:
                            new_c = cand
                            break
                    
                    if new_c is None:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                        
                    if is_valid_swap(old_c, new_c):
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        code[idx] = new_c
                        code_set.remove(old_c)
                        code_set.add(new_c)

            else:
                # Try Add
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    if rng.random() < math.exp(-1.0 / temp):
                        code.append(new_c)
                        code_set.add(new_c)
                        for u in neigh[new_c]:
                            counts[u] += 1
                        
            temp *= cool_rate
        
        # Final polish
        changed = True
        while changed and time.monotonic() < deadline:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if is_removable(c):
                    code.pop(i)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]

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
