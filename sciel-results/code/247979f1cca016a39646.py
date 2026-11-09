# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random
import sys

def _base_cover(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, perfect covering radius 1)
    # with every word on the remaining n - 4 coordinates: 9 * q**(n - 4) words.
    # For q=3, n=6, radius=1: 9 * 3^2 = 81 words.
    if q != 3 or n < 4:
        # Fallback for other parameters
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Ternary Tetracode: { (a, b, a+b, a+2b) | a,b in {0,1,2} }
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    if n <= 4:
        return [list(h[:n]) for h in tetracode]
    
    # Extend to length n by appending all possible tails
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]

def _is_cover(codewords, q, n, radius):
    if not codewords:
        return n == 0
    if len(codewords) == 0:
        return False
    C = np.array(codewords, dtype=np.int32).reshape(-1, n)
    # For n=6, q=3, total words is 729. This is small enough for brute force check.
    # For larger sizes, this might be slow, but the task is specific to 6 matches.
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
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
    all_words = np.array([_words_from_indices([i], q, n)[0] for i in range(M)], dtype=np.int32)
    
    # Compute distance matrix
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

    code_indices = set(word_to_idx(w) for w in init_code)
    code_list = list(code_indices)
    
    # coverage_count[w] = number of codewords in C that cover w
    coverage_count = np.zeros(M, dtype=np.int32)
    for c in code_list:
        for w in balls[c]:
            coverage_count[w] += 1
    
    current_uncovered = np.sum(coverage_count == 0)
    
    # We want to minimize |C| subject to uncovered == 0.
    # Cost = |C| + P * uncovered
    P = M + 100
    current_cost = len(code_list) + P * current_uncovered
    
    best_code = list(code_list)
    best_cost = current_cost
    best_uncovered = current_uncovered
    
    rng = random.Random(42)
    start_time = time.time()
    
    T = 1.0
    cooling = 0.999
    min_T = 1e-5
    
    # We will run until deadline
    while time.time() < deadline - 2:
        if T < min_T:
            # Restart with higher T if we've cooled down
            T = 0.5
            # Optional: restart from best if stuck? 
            # For now, just continue.
        
        # Pick a move
        r = rng.random()
        if r < 0.4:
            # Remove
            if not code_list:
                continue
            c = code_list[rng.randrange(len(code_list))]
            
            # Calculate delta
            # Words that will lose coverage
            newly_uncovered = 0
            for w in balls[c]:
                if coverage_count[w] == 1:
                    newly_uncovered += 1
            
            delta_uncovered = newly_uncovered
            delta_cost = 1 + P * delta_uncovered
            
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                # Accept
                code_list.remove(c)
                for w in balls[c]:
                    coverage_count[w] -= 1
                current_uncovered += newly_uncovered
                current_cost += delta_cost
                
                if current_uncovered == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(code_list)
                    best_uncovered = 0
                    
        elif r < 0.9:
            # Swap
            if not code_list:
                continue
            c = code_list[rng.randrange(len(code_list))]
            c_new = rng.randrange(M)
            
            if c_new == c or c_new in code_list:
                continue
                
            # Calculate delta for swap (remove c, add c_new)
            # Affected words: union(balls[c], balls[c_new])
            # We need to be careful with words in both balls
            
            # Save old counts for affected words
            affected = set(balls[c]) | set(balls[c_new])
            old_counts = {w: coverage_count[w] for w in affected}
            
            # Apply changes
            for w in balls[c]:
                coverage_count[w] -= 1
            for w in balls[c_new]:
                coverage_count[w] += 1
            
            new_uncovered = 0
            for w in affected:
                if coverage_count[w] == 0:
                    new_uncovered += 1
            
            old_uncovered = 0
            for w in affected:
                if old_counts[w] == 0:
                    old_uncovered += 1
            
            delta_uncovered = new_uncovered - old_uncovered
            delta_cost = P * delta_uncovered # Size doesn't change
            
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                # Accept
                code_list[code_list.index(c)] = c_new
                current_uncovered += delta_uncovered
                current_cost += delta_cost
                
                if current_uncovered == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(code_list)
                    best_uncovered = 0
            else:
                # Reject, revert
                for w in affected:
                    coverage_count[w] = old_counts[w]
                    
        else:
            # Add (rarely used, but helps if we get stuck in a bad state)
            if len(code_list) >= M:
                continue
            c_new = rng.randrange(M)
            if c_new in code_list:
                continue
            
            # Calculate delta
            newly_covered = 0
            for w in balls[c_new]:
                if coverage_count[w] == 0:
                    newly_covered += 1
            
            delta_uncovered = -newly_covered
            delta_cost = 1 + P * delta_uncovered
            
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                # Accept
                code_list.append(c_new)
                for w in balls[c_new]:
                    coverage_count[w] += 1
                current_uncovered -= newly_covered
                current_cost += delta_cost
                
                if current_uncovered == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(code_list)
                    best_uncovered = 0

        T *= cooling
        
    # Final polish: try to remove redundant codewords from best_code
    # Rebuild coverage count for best_code
    coverage_count = np.zeros(M, dtype=np.int32)
    for c in best_code:
        for w in balls[c]:
            coverage_count[w] += 1
            
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for i in range(len(best_code)):
            if time.time() > deadline - 0.5:
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
