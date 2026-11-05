"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _ball_indices(word_idx, q, n, radius):
    """
    Return a set of indices (into the full word list) within Hamming distance
    `radius` from the word with index `word_idx`.
    Words are indexed by base-q number: idx = sum(w[i] * q**(n-1-i)).
    """
    ball = {word_idx}
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = word_idx % q
        word_idx //= q
    # Precompute powers of q for fast index conversion
    powers = [1] * n
    for i in range(n - 2, -1, -1):
        powers[i] = powers[i + 1] * q

    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(1, q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    new_word[pos] = (word[pos] + val) % q
                # Convert back to index
                idx = 0
                for j in range(n):
                    idx = idx * q + new_word[j]
                ball.add(idx)
    return ball


def _tetracode(n, q):
    """
    Generate the tetracode-based construction for q=3, n=6.
    This gives a code of size 9 * 3^(n-4) = 81 for n=6.
    Actually for n=6: 9 * 3^2 = 81.
    """
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(h + list(tail))
    return code


def _index_word(word, q, n):
    """Convert a word (list of ints) to its base-q index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _word_from_index(idx, q, n):
    """Convert a base-q index to a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _greedy_cover_fast(q, n, radius, deadline, balls):
    """
    Greedy set cover using precomputed balls.
    Returns a list of word indices.
    """
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    solution = []
    
    # Precompute for each candidate, the list of indices it covers
    # To speed up, we track uncovered count per candidate
    
    # Initial state: nothing covered
    uncovered_count = np.full(total, total, dtype=np.int32)
    
    while not covered.all():
        if time.time() > deadline:
            break
        
        # Find candidate with maximum new coverage
        # For each candidate, new_coverage = number of uncovered words it covers
        best_idx = -1
        best_gain = -1
        
        # We can't iterate over all candidates each time efficiently
        # Instead, use a heuristic: iterate through candidates and compute gain
        # But this is O(total) per step, and we may need ~73 steps
        # So total is O(73 * 729) which is fine
        
        for cand in range(total):
            if covered[cand]:
                # This candidate is already covered, but it might still cover new words
                pass
            ball = balls[cand]
            gain = 0
            for w in ball:
                if not covered[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = cand
        
        if best_idx == -1 or best_gain == 0:
            break
        
        solution.append(best_idx)
        for w in balls[best_idx]:
            covered[w] = True
    
    return solution


def _local_search_remove(q, n, radius, solution, balls, deadline):
    """
    Try to remove redundant codewords.
    Returns a possibly smaller solution.
    """
    total = q ** n
    current = list(solution)
    
    # Precompute coverage for each codeword
    # For each position in current, compute what it uniquely covers
    # and what it covers in general
    
    while len(current) > 1:
        if time.time() > deadline:
            break
        
        # Try removing each codeword and check if coverage is maintained
        removed_any = False
        for i in range(len(current)):
            # Check if removing current[i] still covers everything
            # Build coverage without this codeword
            covered_without = np.zeros(total, dtype=bool)
            for j, cand in enumerate(current):
                if j == i:
                    continue
                for w in balls[cand]:
                    covered_without[w] = True
            
            if covered_without.all():
                current.pop(i)
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return current


def _swap_local_search(q, n, radius, solution, balls, deadline):
    """
    Swap-based local search: try replacing one codeword with another
    to maintain coverage while potentially enabling further removals.
    """
    total = q ** n
    current = list(solution)
    
    # Precompute coverage
    def get_coverage(codes):
        cov = np.zeros(total, dtype=bool)
        for cand in codes:
            for w in balls[cand]:
                cov[w] = True
        return cov
    
    best_solution = list(current)
    best_len = len(current)
    
    # Simulated annealing for swaps
    temp = 5.0
    cooling = 0.97
    
    for sa_iter in range(200):
        if time.time() > deadline:
            break
        if len(current) <= 1:
            break
        
        # Try a swap: remove one, add another
        idx_remove = random.randint(0, len(current) - 1)
        
        # Find a candidate to add that covers the most words uncovered by removing idx_remove
        # First, compute what's covered without current[idx_remove]
        covered_without = np.zeros(total, dtype=bool)
        for j, cand in enumerate(current):
            if j == idx_remove:
                continue
            for w in balls[cand]:
                covered_without[w] = True
        
        # Words that become uncovered
        uncovered = np.where(~covered_without)[0]
        
        if len(uncovered) == 0:
            # Safe to remove
            current.pop(idx_remove)
            temp *= cooling
            continue
        
        # Find best candidate to add
        best_add = -1
        best_gain = -1
        for cand in range(total):
            if cand in current:
                continue
            gain = 0
            for w in balls[cand]:
                if not covered_without[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_add = cand
        
        if best_add == -1:
            temp *= cooling
            continue
        
        # Check if adding best_add covers all uncovered
        new_covered = covered_without.copy()
        for w in balls[best_add]:
            new_covered[w] = True
        
        if new_covered.all():
            # Valid swap
            current[idx_remove] = best_add
            temp *= cooling
        else:
            # Invalid, accept with probability
            penalty = total - new_covered.sum()
            if random.random() < np.exp(-penalty / max(temp, 0.01)):
                current[idx_remove] = best_add
            temp *= cooling
        
        # Try to remove redundant codewords after swap
        for _ in range(3):
            if time.time() > deadline:
                break
            removed = False
            for i in range(len(current)):
                covered_test = np.zeros(total, dtype=bool)
                for j, cand in enumerate(current):
                    if j == i:
                        continue
                    for w in balls[cand]:
                        covered_test[w] = True
                if covered_test.all():
                    current.pop(i)
                    removed = True
                    break
            if not removed:
                break
        
        if len(current) < best_len:
            best_len = len(current)
            best_solution = list(current)
    
    return best_solution


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s safety margin before 150s kill
    
    total_words = q ** n
    
    # Precompute balls for all words
    balls = {}
    for idx in range(total_words):
        word = _word_from_index(idx, q, n)
        balls[idx] = _ball_indices(idx, q, n, radius)
    
    # Try multiple random restarts with greedy + local search
    best_solution = None
    best_size = float('inf')
    
    # Start with tetracode construction as initial guess
    tetracode_solution = _tetracode(n, q)
    if tetracode_solution is not None:
        # Convert to indices
        tetracode_indices = [_index_word(w, q, n) for w in tetracode_solution]
        # Verify coverage
        covered = np.zeros(total_words, dtype=bool)
        for idx in tetracode_indices:
            for w in balls[idx]:
                covered[w] = True
        if covered.all():
            best_solution = tetracode_indices
            best_size = len(tetracode_indices)
    
    # Greedy set cover with multiple restarts
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        
        random.seed(42 + attempt)
        
        # Greedy cover
        solution = _greedy_cover_fast(q, n, radius, deadline - 20, balls)
        
        if not solution:
            continue
        
        # Verify coverage
        covered = np.zeros(total_words, dtype=bool)
        for idx in solution:
            for w in balls[idx]:
                covered[w] = True
        
        if not covered.all():
            continue
        
        # Local search: remove redundant
        solution = _local_search_remove(q, n, radius, solution, balls, deadline - 10)
        
        # Swap-based local search
        solution = _swap_local_search(q, n, radius, solution, balls, deadline - 5)
        
        # Final verification
        covered = np.zeros(total_words, dtype=bool)
        for idx in solution:
            for w in balls[idx]:
                covered[w] = True
        
        if covered.all() and len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution
    
    if best_solution is None:
        # Fallback: use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert indices back to words
    return [_word_from_index(idx, q, n) for idx in best_solution]
# EVOLVE-BLOCK-END
