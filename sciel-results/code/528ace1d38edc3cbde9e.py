import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    space_size = q ** n

    # Precompute all words as an array of shape (space_size, n)
    all_words = np.zeros((space_size, n), dtype=np.int32)
    for idx in range(space_size):
        val = idx
        for pos in range(n - 1, -1, -1):
            all_words[idx, pos] = val % q
            val //= q

    # Precompute neighbor indices for each word (distance <= radius)
    # For radius=1: each word has 1 + n*(q-1) neighbors
    neighbor_lists = []
    for i in range(space_size):
        w = all_words[i]
        nbrs = [i]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for nv in range(q):
                    if nv != orig:
                        # compute index of neighbor
                        idx2 = 0
                        for p in range(n):
                            if p == pos:
                                idx2 = idx2 * q + nv
                            else:
                                idx2 = idx2 * q + w[p]
                        nbrs.append(idx2)
        neighbor_lists.append(nbrs)

    # Convert to numpy arrays for faster access
    neighbor_arrays = [np.array(nbrs, dtype=np.int32) for nbrs in neighbor_lists]

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Build initial code: tetracode direct sum with full tail
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))

    initial_codewords = []
    for h in tetracode:
        for tail in tails:
            initial_codewords.append(h + list(tail))

    def compute_counts(codeword_indices):
        """Compute coverage counts for a set of codeword indices."""
        counts = np.zeros(space_size, dtype=np.int32)
        for ci in codeword_indices:
            counts[neighbor_arrays[ci]] += 1
        return counts

    def is_valid(codeword_indices):
        """Check if all words are covered."""
        counts = compute_counts(codeword_indices)
        return np.all(counts >= 1)

    def greedy_delete(codeword_indices, max_passes=50, rng=None):
        """Greedy deletion: remove codewords that don't cause any count to drop to 0."""
        if rng is None:
            rng = random.Random(42)
        counts = compute_counts(codeword_indices)
        active = list(codeword_indices)
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            removed_any = False
            order = list(range(len(active)))
            rng.shuffle(order)
            for i in reversed(order):
                ci = active[i]
                nbrs = neighbor_arrays[ci]
                # Check if any neighbor has count == 1
                if np.any(counts[nbrs] == 1):
                    continue
                # Safe to remove
                counts[nbrs] -= 1
                active.pop(i)
                removed_any = True
            if not removed_any:
                break
        return active

    def local_search_swap(codeword_indices, max_iters=2000, rng=None):
        """Try to improve by swapping codewords."""
        if rng is None:
            rng = random.Random(123)
        current = list(codeword_indices)
        counts = compute_counts(current)
        current_size = len(current)
        best_size = current_size
        best_code = current[:]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if len(current) >= best_size:
                # Try to find a swap that allows deletion
                # Pick a random codeword to potentially remove
                if len(current) == 0:
                    break
                ci = rng.choice(current)
                # Try to find a replacement that, when added, allows removing ci
                # Or just try random swaps
                # Strategy: try replacing ci with a random word, then do greedy delete
                # Only accept if size doesn't increase
                # Alternative: try to find a word that covers the "exposed" neighbors of ci
                # Simpler: random swap
                new_idx = rng.randint(0, space_size - 1)
                if new_idx in current:
                    continue
                
                # Tentative: remove ci, add new_idx
                # Check if coverage is maintained
                # Quick check: does new_idx cover all neighbors of ci that had count==1?
                exposed = neighbor_arrays[ci][counts[neighbor_arrays[ci]] == 1]
                if len(exposed) > 0:
                    # new_idx must cover all exposed
                    new_nbrs = set(neighbor_arrays[new_idx].tolist())
                    if not all(e in new_nbrs for e in exposed):
                        continue
                
                # Perform swap
                counts[neighbor_arrays[ci]] -= 1
                counts[neighbor_arrays[new_idx]] += 1
                current[current.index(ci)] = new_idx
                
                # Try greedy delete after swap
                current = greedy_delete(current, max_passes=5, rng=rng)
                
                if len(current) < best_size:
                    best_size = len(current)
                    best_code = current[:]
                elif len(current) > current_size:
                    # Revert
                    # This is complex; let's just restart from best_code
                    current = best_code[:]
                    counts = compute_counts(current)
        
        return best_code

    # Main: try multiple strategies
    best_code = None
    best_size = float('inf')

    # Strategy 1: Greedy delete from initial
    code1 = greedy_delete([word_to_idx(c) for c in initial_codewords], max_passes=100, rng=random.Random(1))
    if len(code1) < best_size:
        best_size = len(code1)
        best_code = code1[:]

    # Strategy 2: Local search from code1
    code2 = local_search_swap(code1, max_iters=3000, rng=random.Random(2))
    if len(code2) < best_size:
        best_size = len(code2)
        best_code = code2[:]

    # Strategy 3: Multiple random restarts of greedy delete with different orders
    for seed in [3, 7, 11, 42, 99]:
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        code = greedy_delete([word_to_idx(c) for c in initial_codewords], max_passes=100, rng=rng)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]

    # Strategy 4: Local search from best
    if best_code is not None and time.time() < deadline - 15:
        code = local_search_swap(best_code, max_iters=5000, rng=random.Random(777))
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]

    # Convert back to list of lists
    result = [idx_to_word(ci) for ci in best_code]
    return result
