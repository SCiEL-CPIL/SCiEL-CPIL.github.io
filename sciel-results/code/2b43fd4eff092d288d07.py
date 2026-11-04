"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _covered_by(word, codewords, radius):
    for cw in codewords:
        if _hamming_dist(word, cw) <= radius:
            return True
    return False


def _build_coverage(codewords, q, n, radius):
    """Return a set of all words covered by the given codewords."""
    covered = set()
    for cw in codewords:
        covered.add(tuple(cw))
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    covered.add(tuple(lst))
    return covered


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    total_words = q ** n

    # Precompute all word indices for fast lookup
    # For q=3, n=6: 729 words
    # Precompute the covered indices for each possible codeword
    # A codeword covers itself + all words at distance 1
    
    # Fast index computation using numpy
    def word_to_idx_fast(word):
        # word is a list/tuple of n ints in 0..q-1
        return int(np.dot(np.array(word, dtype=np.int64), 
                          (q ** np.arange(n - 1, -1, -1, dtype=np.int64))))

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute powers of q
    q_powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    
    def word_to_idx(word):
        return int(np.dot(np.array(word, dtype=np.int64), q_powers[::-1]))

    # Precompute covered indices for each possible codeword (all q^n possible)
    # This is expensive but done once
    all_covered = {}
    for idx in range(total_words):
        cw = idx_to_word(idx)
        covered = [idx]
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    covered.append(word_to_idx(lst))
        all_covered[idx] = tuple(covered)
    
    # Convert to numpy array for fast access
    max_cover_per_word = 1 + n * (q - 1) if radius >= 1 else 1
    covered_arr = np.zeros((total_words, max_cover_per_word), dtype=np.int32)
    covered_lens = np.zeros(total_words, dtype=np.int32)
    for idx in range(total_words):
        cov = all_covered[idx]
        covered_arr[idx, :len(cov)] = cov
        covered_lens[idx] = len(cov)
    
    def get_covered_indices(cw):
        idx = word_to_idx(cw)
        return covered_arr[idx, :covered_lens[idx]]

    def build_coverage_count(cw_list):
        counts = np.zeros(total_words, dtype=np.int32)
        for cw in cw_list:
            idx = word_to_idx(cw)
            cov = covered_arr[idx, :covered_lens[idx]]
            counts[cov] += 1
        return counts

    def is_redundant(cw, counts):
        idx = word_to_idx(cw)
        cov = covered_arr[idx, :covered_lens[idx]]
        return not np.any(counts[cov] <= 1)

    # Build initial code using direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    if n >= 4:
        codewords = [h + list(tail) for h in tetracode 
                    for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        # Fallback: greedy covering
        codewords = []
        remaining = set(range(total_words))
        while remaining and time.time() < deadline:
            # Pick codeword that covers most remaining
            best_cw = None
            best_count = -1
            for idx in remaining:
                cov = set(covered_arr[idx, :covered_lens[idx]])
                count = len(cov & remaining)
                if count > best_count:
                    best_count = count
                    best_cw = idx
                if time.time() > deadline - 5:
                    break
            if best_cw is None:
                break
            codewords.append(idx_to_word(best_cw))
            cov = set(covered_arr[best_cw, :covered_lens[best_cw]])
            remaining -= cov
    
    # Greedy removal with vectorized operations
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        counts = build_coverage_count(codewords)
        
        # Try removing in multiple orders
        for order_type in ['reverse', 'forward', 'random']:
            if time.time() > deadline - 5:
                break
            counts = build_coverage_count(codewords)
            
            if order_type == 'reverse':
                indices = list(range(len(codewords) - 1, -1, -1))
            elif order_type == 'forward':
                indices = list(range(len(codewords)))
            else:
                indices = list(range(len(codewords)))
                random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 5:
                    break
                if is_redundant(codewords[i], counts):
                    codewords.pop(i)
                    improved = True
                    break
            
            if improved:
                break
    
    # Swap-based local search: try replacing codewords to reduce count
    # or find a configuration with fewer codewords
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        counts = build_coverage_count(codewords)
        
        # Try to remove more codewords
        removed = False
        for order_type in ['random', 'reverse', 'forward']:
            if time.time() > deadline - 10:
                break
            counts = build_coverage_count(codewords)
            
            if order_type == 'random':
                indices = list(range(len(codewords)))
                random.shuffle(indices)
            elif order_type == 'reverse':
                indices = list(range(len(codewords) - 1, -1, -1))
            else:
                indices = list(range(len(codewords)))
            
            for i in indices:
                if time.time() > deadline - 10:
                    break
                if is_redundant(codewords[i], counts):
                    codewords.pop(i)
                    removed = True
                    break
            
            if removed:
                break
        
        if not removed:
            # Try swap: replace one codeword with another that allows removal of another
            # This is more complex; for now, try random restarts of removal
            pass
    
    # Final pass: try all removals one more time
    counts = build_coverage_count(codewords)
    for i in range(len(codewords) - 1, -1, -1):
        if time.time() > deadline - 2:
            break
        if is_redundant(codewords[i], counts):
            codewords.pop(i)
            counts = build_coverage_count(codewords)

    return codewords
# EVOLVE-BLOCK-END
