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
    return neigh, total

def construct(q, n, radius):
    deadline = time.monotonic() + 110.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, _ = _precompute_neighbors(q, n, radius)
    
    # 2. Greedy Initialization
    # We want to start with a reasonably good cover.
    # Greedy: pick word covering most uncovered words.
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

    def remove_from_code(c):
        nonlocal rem_count
        code.remove(c)
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
    # Returns True if valid, and updates counts if we want to accept?
    # Better: just check validity.
    def is_valid_swap(c, new_c):
        # Tentative change: remove c, add new_c
        # We need to ensure no word has count 0 after change.
        # Words affected: neighbors of c and neighbors of new_c.
        # Let's just simulate the counts change on a temporary array? 
        # Or just check the critical words.
        # A word u is at risk if it is covered by c but not new_c, and counts[u] == 1.
        # Or if it is covered by new_c but not c? No, adding new_c only increases counts.
        # So we only need to check words that lose coverage.
        # These are u in neigh[c] such that u NOT in neigh[new_c].
        
        # To check "u in neigh[new_c]" quickly, we can use a set?
        # Precomputing sets for all 729 words might be memory heavy but fast.
        # 729 * 13 ints is small.
        # Let's just iterate.
        
        for u in neigh[c]:
            if counts[u] == 1:
                # This word is covered ONLY by c (in the current state).
                # If we remove c, it must be covered by new_c.
                # Check if u is in neigh[new_c]
                found = False
                for v in neigh[new_c]:
                    if v == u:
                        found = True
                        break
                if not found:
                    return False
        return True

    while time.monotonic() < deadline:
        if temp < min_temp:
            temp = min_temp
            
        move_type = rng.random()
        
        if move_type < 0.6:
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
                    
        elif move_type < 0.9:
            # Try Swap
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                new_c = rng.randint(0, total - 1)
                
                if new_c == old_c or new_c in code:
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
                else:
                    pass

        else:
            # Try Add (Perturbation to escape local minima)
            new_c = rng.randint(0, total - 1)
            if new_c not in code:
                # Accept with prob exp(-1/temp)
                if rng.random() < math.exp(-1.0 / temp):
                    code.append(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1
                    # Note: we do not update best_code here as size increased
                    
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
