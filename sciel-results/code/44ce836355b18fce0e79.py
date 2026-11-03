import time
import random
import math
import itertools

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
    deadline = time.monotonic() + 140.0
    total = q ** n
    rng = random.Random(12345)
    
    # Precompute neighbors
    neigh, _ = _precompute_neighbors(q, n, radius)
    
    # Build inverse index: for each position, which codewords cover it
    # covered_by[pos] = list of codeword indices that cover pos
    # We'll maintain this dynamically
    
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

    # Greedy construction with multiple restarts
    def greedy_construct(seed_offset):
        rng_local = random.Random(42 + seed_offset)
        remaining = bytearray([1]) * total  # 1 means uncovered
        rem_count = total
        code = []
        
        while rem_count > 0:
            best_c = -1
            best_gain = -1
            # Sample candidates for speed
            candidates = list(range(total))
            rng_local.shuffle(candidates)
            # Check first 200 candidates, or all if fewer
            check_limit = min(200, total)
            for c in candidates[:check_limit]:
                gain = 0
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain <= 0:
                # Fallback: full scan
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
                
            code.append(best_c)
            for u in neigh[best_c]:
                if remaining[u]:
                    remaining[u] = 0
                    rem_count -= 1
        
        return code

    # Local search: remove redundant codewords
    def local_search(code, time_limit=None):
        code = list(code)
        # Build coverage counts
        counts = [0] * total
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        
        changed = True
        while changed:
            if time_limit and time.monotonic() > time_limit:
                break
            changed = False
            new_code = []
            for c in code:
                # Check if c is redundant
                redundant = True
                for u in neigh[c]:
                    if counts[u] <= 1:
                        redundant = False
                        break
                if redundant:
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
                else:
                    new_code.append(c)
            code = new_code
        return code

    # 2-for-1 swap: replace two codewords with one if possible
    def try_swap(code, time_limit=None):
        code = list(code)
        counts = [0] * total
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        
        # For each pair of codewords, check if their union of neighbors can be covered by a single codeword
        # To make this efficient, we sample pairs
        improved = False
        n_iter = 0
        while not improved and n_iter < 50:
            if time_limit and time.monotonic() > time_limit:
                break
            n_iter += 1
            if len(code) < 2:
                break
            # Sample a random pair
            i, j = rng.sample(range(len(code)), 2)
            c1, c2 = code[i], code[j]
            
            # Union of neighbors of c1 and c2
            union_set = set(neigh[c1]) | set(neigh[c2])
            
            # Check if any single codeword covers the entire union
            found = -1
            for c in range(total):
                if c == c1 or c == c2:
                    continue
                # Check if neigh[c] covers union_set
                covers = True
                for u in union_set:
                    if u not in neigh[c]:
                        covers = False
                        break
                if covers:
                    found = c
                    break
            
            if found != -1:
                # Remove c1 and c2, add found
                new_code = []
                for k, c in enumerate(code):
                    if k != i and k != j:
                        new_code.append(c)
                new_code.append(found)
                return new_code
        
        return code

    # Repair: add codewords to cover uncovered positions
    def repair(code, time_limit=None):
        code = list(code)
        counts = [0] * total
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        
        uncovered = set()
        for i in range(total):
            if counts[i] == 0:
                uncovered.add(i)
        
        while uncovered:
            if time_limit and time.monotonic() > time_limit:
                break
            best_c = -1
            best_gain = -1
            # Sample candidates
            candidates = list(range(total))
            rng.shuffle(candidates)
            check_limit = min(150, total)
            for c in candidates[:check_limit]:
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
                # Full scan fallback
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
            
            if best_c == -1:
                break
                
            code.append(best_c)
            for u in neigh[best_c]:
                counts[u] += 1
                if counts[u] == 1 and u in uncovered:
                    uncovered.remove(u)
        
        return code

    # Multiple restarts of greedy + local search
    best_code = None
    best_size = float('inf')
    
    # Try several different greedy seeds
    for restart in range(8):
        if time.monotonic() > deadline - 20:
            break
        code = greedy_construct(restart)
        code = local_search(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            if best_size <= 73:
                break
    
    # If we haven't found something good yet, use tetracode-based initialization
    if best_code is None or best_size > 80:
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            init_code = []
            for h in tetracode:
                for tail in itertools.product(range(q), repeat=n - 4):
                    w = h + list(tail)
                    init_code.append(word_to_idx(w))
            init_code = local_search(init_code)
            if len(init_code) < best_size:
                best_size = len(init_code)
                best_code = init_code
    
    if best_code is None:
        # Last resort: pure greedy
        best_code = greedy_construct(999)
        best_code = local_search(best_code)
        best_size = len(best_code)

    # Perturbation loop: drop some codewords, repair, local search, try swaps
    while time.monotonic() < deadline - 5:
        # Choose number of codewords to drop
        k_drop = rng.randint(1, 4)
        if len(best_code) <= k_drop:
            k_drop = max(1, len(best_code) - 1)
        
        current_code = list(best_code)
        rng.shuffle(current_code)
        current_code = current_code[k_drop:]
        
        # Repair
        tl_repair = min(time.monotonic() + 10.0, deadline - 5)
        repaired_code = repair(current_code, time_limit=tl_repair)
        
        # Local search
        tl_ls = min(time.monotonic() + 10.0, deadline - 5)
        improved_code = local_search(repaired_code, time_limit=tl_ls)
        
        # Try swaps
        tl_swap = min(time.monotonic() + 5.0, deadline - 5)
        improved_code = try_swap(improved_code, time_limit=tl_swap)
        
        if len(improved_code) < best_size:
            best_size = len(improved_code)
            best_code = improved_code
            if best_size <= 73:
                break

    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = idx_to_word(idx)
        result.append(word)
        
    return result
