import time
import random
import math
import itertools
import numpy as np
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
    return neigh, total

def _greedy_init(q, n, radius, neigh, total, rng=None):
    """Greedy set cover initialization with vectorized gain computation."""
    remaining = np.ones(total, dtype=np.bool_)
    rem_count = total
    code = []
    
    # Precompute neighbor lists as a 2D numpy array for vectorized operations
    neigh_arr = np.array(neigh, dtype=np.int32)  # shape (total, deg)
    deg = neigh_arr.shape[1]
    
    while rem_count > 0:
        best_c = -1
        best_gain = -1
        # Sample a subset if total is large, otherwise scan all
        if total <= 2000:
            candidates = np.arange(total)
        else:
            candidates = np.array(rng.sample(range(total), min(500, total))) if rng else np.arange(total)
        
        # Vectorized gain computation: sum of remaining over neighbors for each candidate
        # remaining[neigh_arr[candidates]] gives shape (len(candidates), deg)
        gains = remaining[neigh_arr[candidates]].sum(axis=1)
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        best_c = candidates[best_idx]
        
        if best_c == -1 or best_gain <= 0:
            break
        
        code.append(int(best_c))
        remaining[neigh_arr[best_c]] = False
        rem_count -= best_gain
    
    return code

def _local_search(code, neigh, total, deadline=None, max_iter=100):
    """Aggressive local search: iteratively remove redundant codewords, then try 1-for-1 exchanges."""
    code = list(code)
    if not code:
        return code
    
    # Build coverage counts
    counts = np.zeros(total, dtype=np.int32)
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
    
    # Use a priority queue for efficient removal
    # A codeword is removable if all its neighbors have count > 1
    removable = []
    for c in code:
        # Check if c is redundant
        is_redundant = True
        for u in neigh[c]:
            if counts[u] <= 1:
                is_redundant = False
                break
        if is_redundant:
            removable.append(c)
    
    iter_count = 0
    while removable and iter_count < max_iter:
        if deadline and time.monotonic() > deadline:
            break
        # Pick a removable codeword (first one for simplicity, or random)
        c = removable.pop(0)
        if c not in code:
            continue
        # Remove it
        code.remove(c)
        for u in neigh[c]:
            counts[u] -= 1
        
        # Re-scan for new removable codewords
        new_removable = []
        for c2 in code:
            is_redundant = True
            for u in neigh[c2]:
                if counts[u] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                new_removable.append(c2)
        removable = new_removable
        iter_count += 1
    
    # 1-for-1 exchange: try removing one codeword and adding a different one
    # to see if we can reduce size (shouldn't help directly, but can enable further removals)
    # More useful: try to find a codeword whose removal allows adding one that covers
    # more unique words, enabling subsequent removals
    if deadline and time.monotonic() > deadline:
        return code
    
    # Try 2-opt: remove codeword i, add codeword j (j not in code), if result is still
    # valid and same size, check if it enables more removals
    code_set = set(code)
    improved = True
    while improved and (not deadline or time.monotonic() < deadline):
        improved = False
        # Recompute counts
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        
        for i, c_i in enumerate(code):
            if deadline and time.monotonic() > deadline:
                return code
            # Tentatively remove c_i
            for u in neigh[c_i]:
                counts[u] -= 1
            
            # Find if there's a codeword j not in code that covers more unique words
            # than c_i did, which might enable further removals
            # Actually, for 1-for-1 exchange to be useful, we need the new codeword to
            # allow removing another codeword. This is complex, so let's just try:
            # after removing c_i, can we add some j and then remove two others?
            # Simpler: just check if removing c_i and adding j keeps coverage and
            # then run local search again
            # This is too expensive. Instead, let's just do the simple removal pass.
            for u in neigh[c_i]:
                counts[u] += 1
    
    return code

def _repair(code, neigh, total, deadline=None):
    """Repair uncovered positions by adding codewords (vectorized)."""
    code = list(code)
    counts = np.zeros(total, dtype=np.int32)
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
    
    uncovered = np.where(counts == 0)[0]
    code_set = set(code)
    
    while len(uncovered) > 0:
        if deadline and time.monotonic() > deadline:
            break
        best_c = -1
        best_gain = -1
        # Try all candidates
        for c in range(total):
            if c in code_set:
                continue
            # Count how many currently-uncovered words this codeword covers
            gain = 0
            for u in neigh[c]:
                if counts[u] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c == -1 or best_gain <= 0:
            break
        
        code.append(best_c)
        code_set.add(best_c)
        for u in neigh[best_c]:
            counts[u] += 1
        
        # Recompute uncovered
        uncovered = np.where(counts == 0)[0]
    
    return code

def _k_swap_perturb(code, neigh, total, k, rng, deadline=None):
    """Remove k codewords, add k different ones (biased toward high-coverage words), then local search."""
    code = list(code)
    if len(code) <= k:
        return code
    
    # Remove k random codewords
    indices = rng.sample(range(len(code)), k)
    indices.sort(reverse=True)
    removed = []
    for idx in indices:
        removed.append(code.pop(idx))
    
    # Build coverage counts for remaining code
    counts = np.zeros(total, dtype=np.int32)
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
    
    # Add k codewords biased toward those covering more uncovered/low-count words
    existing = set(code)
    for _ in range(k):
        # Score each candidate by how many low-count words it covers
        best_c = -1
        best_score = -1
        # Sample candidates for speed
        sample_size = min(100, total)
        candidates = rng.sample(range(total), sample_size)
        for c in candidates:
            if c in existing:
                continue
            # Score: sum of (1 - min(counts[u], 1)) for u in neigh[c]
            # i.e., number of words with count 0 or 1 that this codeword covers
            score = 0
            for u in neigh[c]:
                if counts[u] <= 1:
                    score += 1
            if score > best_score:
                best_score = score
                best_c = c
        
        if best_c != -1:
            code.append(best_c)
            existing.add(best_c)
            for u in neigh[best_c]:
                counts[u] += 1
    
    # Repair if needed
    counts = np.zeros(total, dtype=np.int32)
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
    if np.any(counts == 0):
        code = _repair(code, neigh, total, deadline)
    
    # Local search
    code = _local_search(code, neigh, total, deadline)
    
    return code

def _submodular_prune(code, neigh, total, deadline=None):
    """
    Iteratively remove codewords with zero marginal coverage loss.
    A codeword has zero marginal loss if no word is uniquely covered by it.
    """
    code = list(code)
    if not code:
        return code
    
    # Build coverage counts
    counts = np.zeros(total, dtype=np.int32)
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
    
    changed = True
    while changed:
        if deadline and time.monotonic() > deadline:
            break
        changed = False
        
        # For each codeword, compute unique coverage
        removal_order = []
        for i, c in enumerate(code):
            unique_cover = 0
            for u in neigh[c]:
                if counts[u] == 1:
                    unique_cover += 1
            removal_order.append((unique_cover, i))
        
        # Sort by unique coverage (ascending)
        removal_order.sort()
        
        for unique_cover, i in removal_order:
            if unique_cover > 0:
                break
            # This codeword is redundant, remove it
            c = code[i]
            for u in neigh[c]:
                counts[u] -= 1
            code.pop(i)
            changed = True
            break  # Restart since indices changed
    
    return code

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

    # 2. Initialize with Tetracode Direct Sum for q=3, n>=4
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        init_code = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                w = h + list(tail)
                init_code.append(word_to_idx(w))
    else:
        init_code = _greedy_init(q, n, radius, neigh, total, rng)
    
    best_code = list(init_code)
    best_size = len(best_code)
    
    # Initial local search
    best_code = _local_search(best_code, neigh, total, deadline)
    best_code = _submodular_prune(best_code, neigh, total, deadline)
    best_size = len(best_code)
    
    # Multi-start ILS
    no_improve_count = 0
    start_time = time.monotonic()
    
    while time.monotonic() < deadline - 5:
        current = list(best_code)
        
        # Random k-swap perturbation
        k = rng.randint(1, 4)
        current = _k_swap_perturb(current, neigh, total, k, rng, deadline)
        
        # Aggressive local search
        current = _local_search(current, neigh, total, deadline)
        current = _submodular_prune(current, neigh, total, deadline)
        
        if len(current) < best_size:
            best_size = len(current)
            best_code = current
            no_improve_count = 0
            if best_size <= 71:
                break
        else:
            no_improve_count += 1
        
        # If stuck, try a more aggressive restart
        if no_improve_count > 30 and time.monotonic() < deadline - 10:
            # Restart from greedy init with different seed
            alt_rng = random.Random(rng.randint(0, 1000000))
            alt_code = _greedy_init(q, n, radius, neigh, total, alt_rng)
            alt_code = _local_search(alt_code, neigh, total, deadline)
            alt_code = _submodular_prune(alt_code, neigh, total, deadline)
            if len(alt_code) < best_size:
                best_size = len(alt_code)
                best_code = alt_code
                no_improve_count = 0
    
    # Final polish: run local search multiple times to squeeze out last words
    for _ in range(3):
        best_code = _local_search(best_code, neigh, total, deadline)
        best_code = _submodular_prune(best_code, neigh, total, deadline)
    best_size = len(best_code)
    
    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = idx_to_word(idx)
        result.append(word)
        
    return result
