# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random


def _base_cover(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, perfect covering radius 1)
    # with every word on the remaining n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    if n <= 4:
        return [list(h[:n]) for h in tetracode]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _is_cover(codewords, q, n, radius):
    if not codewords:
        return n == 0
    C = np.array(codewords, dtype=np.int32).reshape(-1, n)
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    # word i is covered by codeword j iff sum_{k} (w_i,k != c_j,k) <= radius
    diff = (words[:, None, :] != C[None, :, :])
    dist = diff.sum(axis=2)
    return bool(np.all(dist.min(axis=1) <= radius))


def _words_from_indices(indices, q, n):
    """Convert integer indices back to word tuples."""
    result = []
    for idx in indices:
        word = []
        rem = int(idx)
        for _ in range(n):
            word.append(rem % q)
            rem //= q
        result.append(word)
    return result


def _simulated_annealing(q, n, radius, deadline):
    M = q ** n
    if M > 100000:
        return _base_cover(q, n, radius)

    # Precompute balls for each possible word index
    # ball[idx] is a list of indices of words within radius of word idx
    all_words_idx = list(range(M))
    all_words = np.array([_words_from_indices([i], q, n)[0] for i in range(M)], dtype=np.int32)
    
    # This might be slow for large M, but M=729 is small.
    # For general q,n, we should be careful.
    # dist matrix M x M
    diff = (all_words[:, None, :] != all_words[None, :, :])
    dist = diff.sum(axis=2)
    balls = [np.where(dist[i] <= radius)[0] for i in range(M)]
    del diff, dist, all_words

    # Initialize with base cover
    init_code = _base_cover(q, n, radius)
    if not init_code:
        return [[]]
    
    # Map words to indices
    def word_to_idx(word):
        idx = 0
        for d in range(n):
            idx = idx * q + word[d]
        return idx

    code_indices = [word_to_idx(w) for w in init_code]
    # Ensure uniqueness (base cover should be unique)
    code_indices = list(set(code_indices))
    
    # coverage_count[w] = number of codewords in C that cover w
    coverage_count = np.zeros(M, dtype=np.int32)
    for c in code_indices:
        for w in balls[c]:
            coverage_count[w] += 1
    
    # Current cost: |C| + P * Uncovered
    # We want to minimize this.
    # P should be large enough to prioritize coverage.
    P = M + 10
    
    current_uncovered = np.sum(coverage_count == 0)
    current_cost = len(code_indices) + P * current_uncovered
    
    best_code = list(code_indices)
    best_cost = current_cost
    
    T = 1.0
    cooling = 0.9995
    min_T = 1e-6
    
    rng = random.Random(42)
    
    start_time = time.time()
    
    while time.time() < deadline - 5:
        if T < min_T:
            break
            
        # Pick a move type
        # Mostly try to remove, sometimes swap
        r = rng.random()
        if r < 0.5:
            # Remove
            if not code_indices:
                continue
            c = code_indices[rng.randrange(len(code_indices))]
            # Calculate delta
            newly_uncovered = 0
            affected_words = balls[c]
            for w in affected_words:
                if coverage_count[w] == 1:
                    newly_uncovered += 1
            delta_uncovered = newly_uncovered
            delta_cost = 1 + P * delta_uncovered
            
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                # Accept
                code_indices.remove(c)
                for w in affected_words:
                    coverage_count[w] -= 1
                current_uncovered += newly_uncovered
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(code_indices)
        elif r < 0.9:
            # Swap
            if not code_indices:
                continue
            c = code_indices[rng.randrange(len(code_indices))]
            c_new = rng.randrange(M)
            if c_new == c:
                continue
            if c_new in code_indices:
                continue
                
            # Calculate delta for swap
            # Remove c, Add c_new
            # We need to check coverage changes.
            # Words covered by c but not c_new might become uncovered if count was 1.
            # Words covered by c_new but not c might become covered (good).
            
            # Optimization: 
            # 1. Temporarily remove c
            # 2. Check if any word becomes uncovered
            # 3. Temporarily add c_new
            # 4. Check if any previously uncovered word becomes covered
            
            # Actually, let's just compute the new coverage state for affected words.
            # Affected words are union(balls[c], balls[c_new])
            
            affected = set(balls[c]) | set(balls[c_new])
            affected_list = list(affected)
            
            # Save old counts
            old_counts = {w: coverage_count[w] for w in affected_list}
            
            # Apply removal
            for w in balls[c]:
                coverage_count[w] -= 1
            # Apply addition
            for w in balls[c_new]:
                coverage_count[w] += 1
            
            new_uncovered = 0
            for w in affected_list:
                if coverage_count[w] == 0:
                    new_uncovered += 1
            
            old_uncovered = 0
            for w in affected_list:
                if old_counts[w] == 0:
                    old_uncovered += 1
            
            delta_uncovered = new_uncovered - old_uncovered
            delta_cost = P * delta_uncovered # Size doesn't change
            
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                # Accept
                code_indices[code_indices.index(c)] = c_new
                current_uncovered += delta_uncovered
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(code_indices)
            else:
                # Reject, revert
                for w in affected_list:
                    coverage_count[w] = old_counts[w]
                    
        T *= cooling
        
    # Final polish: try to remove redundant codewords
    # Rebuild coverage count for best_code
    coverage_count = np.zeros(M, dtype=np.int32)
    for c in best_code:
        for w in balls[c]:
            coverage_count[w] += 1
            
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for i in range(len(best_code)):
            if time.time() > deadline - 1:
                break
            c = best_code[i]
            # Check if redundant
            is_redundant = True
            for w in balls[c]:
                if coverage_count[w] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove
                best_code.pop(i)
                for w in balls[c]:
                    coverage_count[w] -= 1
                changed = True
                break
                
    return _words_from_indices(best_code, q, n)


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # safety margin vs 120s wall-clock

    if n == 0:
        return [[]]

    # Baseline that is guaranteed valid and always available to return.
    base = _base_cover(q, n, radius)
    if not _is_cover(base, q, n, radius):
        base = [list(w) for w in itertools.product(range(q), repeat=n)]
    best = base

    # Try Simulated Annealing
    try:
        cand = _simulated_annealing(q, n, radius, deadline)
        if len(cand) < len(best) and _is_cover(cand, q, n, radius):
            best = cand
    except Exception:
        pass

    if time.time() > deadline:
        return best
    return best
# EVOLVE-BLOCK-END
