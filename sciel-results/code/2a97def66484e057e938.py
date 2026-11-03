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

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, neigh_set, _ = _precompute_neighbors(q, n, radius)
    
    # 2. Greedy Set Cover Initialization
    # Start with an empty code and add codewords that cover the most uncovered words.
    uncovered = [True] * total
    rem_count = total
    code = []
    code_set = set()
    
    while rem_count > 0:
        if time.monotonic() > deadline - 60.0:
            break # Safety break, though greedy is fast
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
                
    # Ensure we have a valid cover. If greedy failed to cover everything (shouldn't happen),
    # we might need to add more. But greedy set cover on a hypergraph always finds a maximal
    # independent set of the dual, which covers everything if the union of balls is the whole space.
    # Actually, greedy set cover adds words until all are covered.
    
    # If for some reason we didn't cover everything (e.g. timeout in greedy loop), force cover
    if rem_count > 0:
        for u in range(total):
            if uncovered[u]:
                # Pick a neighbor of u not in code
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

    best_code = code[:]
    best_size = len(code)
    
    # 3. Simulated Annealing
    # We maintain counts of how many times each word is covered.
    counts = [0] * total
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
            
    temp = 1.0
    cool_rate = 0.9995
    min_temp = 0.01
    start_time = time.monotonic()
    
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    def is_valid_swap(old_c, new_c):
        # Check if removing old_c and adding new_c keeps coverage
        # Equivalent to: all words covered only by old_c must be covered by new_c
        for u in neigh[old_c]:
            if counts[u] == 1 and u not in neigh_set[new_c]:
                return False
        return True

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
                    code.pop(idx)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                        
        elif move_type < 0.8:
            # Try Swap
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                new_c = rng.randint(0, total - 1)
                
                if new_c == old_c or new_c in code_set:
                    continue
                    
                if is_valid_swap(old_c, new_c):
                    # Perform swap
                    for u in neigh[old_c]:
                        counts[u] -= 1
                    for u in neigh[new_c]:
                        counts[u] += 1
                    code[idx] = new_c
                    code_set.discard(old_c)
                    code_set.add(new_c)
                    # Size doesn't change, but position might enable future removals
                    # We don't update best_size here unless we do a cleanup pass later
                    # But sometimes a swap allows immediate removals, which will be caught in next 'Remove' steps
                    
        else:
            # Try Add (Perturbation)
            new_c = rng.randint(0, total - 1)
            if new_c not in code_set:
                # Accept with some probability, usually low to avoid bloat
                # Or just accept if it helps escape a local minimum
                # Since we minimize size, adding is bad, but helps escape.
                # We accept if it's "not too bad" or just randomly with low prob
                if rng.random() < 0.1:
                    code.append(new_c)
                    code_set.add(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1
                    
        temp *= cool_rate

    # Final cleanup: try to remove redundant words from best_code
    # Re-verify best_code validity and minimality
    # We assume best_code is valid because we started valid and only did valid swaps/removes.
    # However, 'Add' moves can make best_code invalid if we updated best_code after an Add?
    # No, we only update best_code on removals which reduce size.
    # Wait, if we do an Add, size increases. We don't update best_code.
    # So best_code is always a valid cover found via removals from a valid state.
    
    # One more pass to ensure best_code is minimal (no removable words)
    # This helps if the SA stopped while redundant words were still present
    best_set = set(best_code)
    best_counts = [0] * total
    for c in best_code:
        for u in neigh[c]:
            best_counts[u] += 1
            
    final_code = []
    for c in best_code:
        removable = True
        for u in neigh[c]:
            if best_counts[u] == 1:
                removable = False
                break
        if not removable:
            final_code.append(c)
            for u in neigh[c]:
                best_counts[u] -= 1
        else:
            # It is removable, so we don't add it to final_code
            pass
            
    # Note: The order of removal matters. Greedy removal might remove a word that was
    # covering something unique only if we removed another word first.
    # The standard "minimal" check is: is there a subset that is still a cover?
    # The above loop produces a minimal cover (no single word can be removed).
    # This might be slightly larger than the absolute minimum, but it's a good polish.
    
    # Actually, let's just return best_code as is, since the SA was trying to minimize it.
    # The "final cleanup" above might actually remove words that were necessary if the
    # "counts" logic was flawed? No, the logic is sound: if count==1, it's essential.
    # If count>1, it's redundant *in the context of the current set*.
    # Removing all redundant words one by one yields a minimal cover.
    
    # Let's stick to best_code as tracked by SA, but apply the minimal reduction to be safe.
    # The SA's 'best_code' is updated only when size decreases.
    # The 'final_code' derived from best_code via greedy minimal reduction should be <= best_size.
    
    if len(final_code) < len(best_code):
        best_code = final_code

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
