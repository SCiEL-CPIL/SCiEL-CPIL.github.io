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
    start_time = time.monotonic()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    total = q ** n
    rng = random.Random(42)
    
    neigh, _ = _precompute_neighbors(q, n, radius)
    neigh_sets = [set(neigh[i]) for i in range(total)]
    
    # Precompute degree-2 adjacency for the "distance-2" graph
    # Two words are adjacent if Hamming distance is exactly 2
    # This helps identify which codewords can be "merged" or are close
    # For the independent set approach, we want to avoid distance-2 pairs
    dist2_adj = [set() for _ in range(total)]
    for idx in range(total):
        for i in range(n):
            pow_q_i = q ** i
            digit_i = (idx // pow_q_i) % q
            for j in range(i + 1, n):
                pow_q_j = q ** j
                digit_j = (idx // pow_q_j) % q
                for new_i in range(q):
                    if new_i == digit_i:
                        continue
                    for new_j in range(q):
                        if new_j == digit_j:
                            continue
                        new_idx = idx + (new_i - digit_i) * pow_q_i + (new_j - digit_j) * pow_q_j
                        if new_idx != idx:
                            dist2_adj[idx].add(new_idx)
    
    # Precompute: for each word, which codewords at distance 2 would conflict
    # This is used in the independent set heuristic
    
    def greedy_init(seed, use_independent_set=False):
        """Greedy set cover with optional independent set bias."""
        r = random.Random(seed)
        counts = [0] * total
        code = []
        code_set = set()
        remaining = [True] * total
        rem_count = total
        
        # Track which words are "available" (not yet excluded by independent set constraint)
        if use_independent_set:
            available = [True] * total
        
        while rem_count > 0:
            if time.monotonic() > deadline - 10.0:
                break
            
            best_c = -1
            best_gain = -1
            best_tiebreak = -1
            
            # For independent set approach, we prefer words with fewer distance-2 conflicts
            # among already chosen words
            conflict_count = [0] * total if use_independent_set else None
            
            if use_independent_set and code:
                # Compute conflict counts for each candidate
                for c in range(total):
                    if c in code_set:
                        continue
                    # Count how many existing codewords are at distance 2
                    for d2 in dist2_adj[c]:
                        if d2 in code_set:
                            conflict_count[c] += 1
            
            for c in range(total):
                if c in code_set:
                    continue
                gain = 0
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                
                if use_independent_set:
                    # Prefer words with fewer conflicts, then higher gain
                    # Use a combined score: gain - penalty * conflicts
                    penalty = 0.1 * conflict_count[c]
                    score = gain - penalty
                else:
                    score = gain
                
                if score > best_gain:
                    best_tiebreak = best_gain
                    best_gain = score
                    best_c = c
                elif score == best_gain and gain > 0:
                    if r.random() < 0.5:
                        best_c = c
            
            if best_c == -1:
                # Fallback: pick any word that covers something
                for c in range(total):
                    if c not in code_set:
                        for u in neigh[c]:
                            if remaining[u]:
                                best_c = c
                                break
                        if best_c != -1:
                            break
                if best_c == -1:
                    break
            
            code.append(best_c)
            code_set.add(best_c)
            
            # If using independent set, mark distance-2 neighbors as less preferred
            # (we don't remove them, just bias against them)
            
            for u in neigh[best_c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining[u] = False
                    rem_count -= 1
        
        # Fallback if not fully covered
        if rem_count > 0:
            while rem_count > 0:
                if time.monotonic() > deadline - 5.0:
                    break
                # Find an uncovered word and add a codeword that covers it
                found = False
                for u in range(total):
                    if remaining[u]:
                        # Pick a codeword that covers u
                        candidates = [nc for nc in neigh[u] if nc not in code_set]
                        if candidates:
                            # Pick the one with max gain
                            best_nc = max(candidates, key=lambda nc: sum(1 for v in neigh[nc] if remaining[v]))
                            code.append(best_nc)
                            code_set.add(best_nc)
                            for v in neigh[best_nc]:
                                counts[v] += 1
                                if counts[v] == 1:
                                    remaining[v] = False
                                    rem_count -= 1
                            found = True
                            break
                if not found:
                    break
        
        return code, counts

    def polish(code, counts, time_budget):
        """Local search to minimize code size."""
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        end_time = time.monotonic() + time_budget
        
        def is_removable(c):
            for u in neigh[c]:
                if counts[u] == 1:
                    return False
            return True
        
        # First pass: remove all removable codewords
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if is_removable(code[i]):
                    c = code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        # SA phase with targeted swaps
        temp = 3.0
        cool_rate = 0.999
        min_temp = 0.05
        
        while time.monotonic() < end_time:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.35:
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
                        
            elif move_type < 0.85:
                # Try Swap - use targeted candidate (neighbor of existing codeword)
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    # Choose new_c from neighbors of old_c or random
                    if rng.random() < 0.7:
                        # Targeted: pick a neighbor of old_c
                        new_c = neigh[old_c][rng.randint(1, len(neigh[old_c])-1)]
                    else:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    # Check validity
                    valid = True
                    new_c_set = neigh_sets[new_c]
                    for u in neigh[old_c]:
                        if counts[u] == 1 and u not in new_c_set:
                            valid = False
                            break
                    
                    if valid:
                        # Accept
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                    else:
                        # SA acceptance for invalid swap (perturbation)
                        uncovered = 0
                        new_c_set = neigh_sets[new_c]
                        for u in neigh[old_c]:
                            if counts[u] == 1 and u not in new_c_set:
                                uncovered += 1
                        if uncovered <= 2 and rng.random() < math.exp(-uncovered / temp):
                            # Accept perturbation
                            for u in neigh[old_c]:
                                counts[u] -= 1
                            for u in neigh[new_c]:
                                counts[u] += 1
                            code[idx] = new_c
                            code_set.discard(old_c)
                            code_set.add(new_c)
                            # Immediately try to fix by adding back uncovered words
                            for u in neigh[old_c]:
                                if counts[u] == 0:
                                    for nc in neigh[u]:
                                        if nc not in code_set:
                                            code.append(nc)
                                            code_set.add(nc)
                                            for v in neigh[nc]:
                                                counts[v] += 1
                                            break
                            
            else:
                # Try Add (perturbation)
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    if rng.random() < math.exp(-1.0 / temp):
                        code.append(new_c)
                        code_set.add(new_c)
                        for u in neigh[new_c]:
                            counts[u] += 1
            
            temp *= cool_rate
        
        # Final removal pass
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if is_removable(code[i]):
                    c = code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        return best_code, best_size

    # Multi-restart strategy with different initialization methods
    overall_best_code = None
    overall_best_size = total + 1
    
    # Determine number of restarts based on time
    num_restarts = 6
    time_per_restart = (deadline - time.monotonic()) / num_restarts
    
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 10.0:
            break
        
        # Alternate between standard greedy and independent-set-biased greedy
        use_is = (restart % 2 == 1)
        code, counts = greedy_init(42 + restart * 7, use_independent_set=use_is)
        
        if not code:
            continue
        
        # Polish
        polished_code, polished_size = polish(code, counts, time_per_restart - 1.0)
        
        if polished_size < overall_best_size:
            overall_best_size = polished_size
            overall_best_code = polished_code[:]
    
    # If we haven't found anything, do one more greedy
    if overall_best_code is None:
        code, counts = greedy_init(42)
        overall_best_code = code
        overall_best_size = len(code)
    
    # Convert to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result
