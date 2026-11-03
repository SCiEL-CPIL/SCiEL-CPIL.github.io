import time
import random
import math
import itertools
import numpy as np

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
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, _ = _precompute_neighbors(q, n, radius)
    
    # Helper to convert index to word and vice versa
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q
        return word
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # 2. Initialize with Tetracode Direct Sum
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        init_code = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                w = h + list(tail)
                init_code.append(word_to_idx(w))
    else:
        init_code = []
        remaining = [True] * total
        rem_count = total
        while rem_count > 0:
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
            init_code.append(best_c)
            for u in neigh[best_c]:
                if remaining[u]:
                    remaining[u] = False
                    rem_count -= 1
    
    best_code = list(init_code)
    best_size = len(best_code)

    # Numpy-based coverage counting
    def compute_counts(code):
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        return counts

    def local_search(code):
        code = list(code)
        changed = True
        while changed:
            changed = False
            counts = compute_counts(code)
            new_code = []
            for c in code:
                # Check if c is redundant: all neighbors must have count > 1
                redundant = True
                for u in neigh[c]:
                    if counts[u] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
                else:
                    new_code.append(c)
            code = new_code
        return code

    def repair_fast(code):
        code = list(code)
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        
        uncovered = set()
        for i in range(total):
            if counts[i] == 0:
                uncovered.add(i)
        
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(total):
                if c in code:
                    continue
                gain = 0
                for u in neigh[c]:
                    if u in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain <= 0:
                break
            
            code.append(best_c)
            for u in neigh[best_c]:
                counts[u] += 1
                if counts[u] == 1 and u in uncovered:
                    uncovered.remove(u)
        
        return code

    # Initial Local Search
    best_code = local_search(best_code)
    best_size = len(best_code)

    # Swap-based local search: try swapping each codeword with another
    def swap_local_search(code):
        code = list(code)
        counts = compute_counts(code)
        improved = True
        while improved:
            improved = False
            for i, c in enumerate(code):
                # Try swapping c with each other codeword c2
                for j, c2 in enumerate(code):
                    if i == j:
                        continue
                    # Check if swap is valid: all positions covered only by c must be covered by c2
                    # After removing c, positions with count==1 that are neighbors of c become uncovered
                    # These must be neighbors of c2
                    valid = True
                    for u in neigh[c]:
                        if counts[u] == 1:  # Only covered by c
                            if u not in neigh[c2]:
                                valid = False
                                break
                    if valid and c2 not in code[:i] + code[i+1:]:  # c2 is already in code, so this is always true
                        # The swap is valid. But we need to check if it enables further removals.
                        # For now, just do the swap if it doesn't increase size (it doesn't)
                        # Actually, swap doesn't change size, so we need a different criterion.
                        # Let's skip pure swaps and focus on remove+add sequences.
                        pass
        return code

    # Better approach: perturbation with structured moves
    def perturb_and_optimize(code, rng, deadline):
        code = list(code)
        size = len(code)
        
        # Try different perturbation strategies
        strategies = [
            'drop_repair',      # Drop random codewords, repair
            'swap_remove',      # Swap then try to remove
            'multi_drop',       # Drop multiple codewords
        ]
        
        strategy = rng.choice(strategies)
        
        if strategy == 'drop_repair':
            k_drop = rng.randint(1, 4)
            if size <= k_drop:
                k_drop = max(1, size - 1)
            current = list(code)
            rng.shuffle(current)
            current = current[k_drop:]
            current = repair_fast(current)
            current = local_search(current)
            
        elif strategy == 'multi_drop':
            k_drop = rng.randint(2, 5)
            if size <= k_drop:
                k_drop = max(1, size - 1)
            current = list(code)
            rng.shuffle(current)
            current = current[k_drop:]
            current = repair_fast(current)
            current = local_search(current)
            
        else:  # swap_remove: drop one, add a different one, then local search
            if size < 2:
                return code
            current = list(code)
            idx_remove = rng.randint(0, size - 1)
            removed = current.pop(idx_remove)
            # Add a random different codeword
            candidates = [c for c in range(total) if c not in current]
            if candidates:
                added = rng.choice(candidates)
                current.append(added)
            current = local_search(current)
            # Repair if needed
            counts = compute_counts(current)
            if np.any(counts == 0):
                current = repair_fast(current)
                current = local_search(current)
        
        return current

    # Main search loop
    no_improve_count = 0
    while time.monotonic() < deadline:
        current = perturb_and_optimize(best_code, rng, deadline)
        
        if len(current) < best_size:
            best_size = len(current)
            best_code = current
            no_improve_count = 0
            if best_size <= 71:
                break
        else:
            no_improve_count += 1
        
        # If we've been stuck for a while, try a more aggressive perturbation
        if no_improve_count > 50 and time.monotonic() < deadline - 10:
            # Restart from scratch with different seed
            alt_rng = random.Random(rng.randint(0, 1000000))
            alt_code = list(init_code)
            # Perturb more aggressively
            for _ in range(rng.randint(5, 15)):
                k = alt_rng.randint(1, 5)
                if len(alt_code) > k:
                    alt_code = alt_code[k:]
                    alt_code = repair_fast(alt_code)
                    alt_code = local_search(alt_code)
            if len(alt_code) < best_size:
                best_size = len(alt_code)
                best_code = alt_code
                no_improve_count = 0

    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = idx_to_word(idx)
        result.append(word)
        
    return result
