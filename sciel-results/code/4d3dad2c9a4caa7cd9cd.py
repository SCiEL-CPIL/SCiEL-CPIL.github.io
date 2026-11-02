"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _ball_indices(word, q, n, radius):
    """Return list of integer indices covered by word within Hamming distance radius."""
    if radius == 0:
        idx = 0
        for d in range(n - 1, -1, -1):
            idx = idx * q + word[d]
        return [idx]
    if radius == 1:
        # For radius 1: the word itself + all single-position changes
        indices = []
        # Original word
        idx = 0
        for d in range(n - 1, -1, -1):
            idx = idx * q + word[d]
        indices.append(idx)
        # Single position changes
        for pos in range(n):
            original_val = word[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_word = word[:]
                new_word[pos] = new_val
                idx = 0
                for d in range(n - 1, -1, -1):
                    idx = idx * q + new_word[d]
                indices.append(idx)
        return indices
    # General radius (not expected for this task)
    from itertools import product
    indices = []
    for delta in product(range(-radius, radius + 1), repeat=n):
        if sum(abs(d) for d in delta) <= radius:
            word_new = []
            valid = True
            for i in range(n):
                v = word[i] + delta[i]
                if v < 0 or v >= q:
                    valid = False
                    break
                word_new.append(v)
            if valid:
                idx = 0
                for d in range(n - 1, -1, -1):
                    idx = idx * q + word_new[d]
                indices.append(idx)
    return indices


def _word_to_idx(word, q):
    idx = 0
    for d in range(len(word) - 1, -1, -1):
        idx = idx * q + word[d]
    return idx


def _idx_to_word(idx, q, n):
    word = [0] * n
    for d in range(n - 1, -1, -1):
        word[d] = idx % q
        idx //= q
    return word


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Direct sum of the ternary tetracode as initial seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    total_words = q ** n
    
    # Precompute ball indices for all possible words
    # This is expensive but we do it once
    all_balls = {}
    for idx in range(total_words):
        word = _idx_to_word(idx, q, n)
        all_balls[idx] = set(_ball_indices(word, q, n, radius))
    
    # Build coverage: which codewords cover which words
    # covered[word_idx] = set of codeword indices that cover it
    covered = [set() for _ in range(total_words)]
    for ci, cw in enumerate(code):
        cw_idx = _word_to_idx(cw, q)
        for w_idx in all_balls[cw_idx]:
            covered[w_idx].add(ci)
    
    # Greedy improvement: try to remove redundant codewords
    # A codeword is redundant if removing it doesn't uncover any word
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to remove each codeword in random order
        indices = list(range(len(code)))
        random.shuffle(indices)
        for ci in indices:
            if time.time() >= deadline:
                break
            # Check if this codeword is redundant
            # Find words covered ONLY by this codeword
            redundant = True
            cw_idx = _word_to_idx(code[ci], q)
            for w_idx in all_balls[cw_idx]:
                if len(covered[w_idx]) == 1 and ci in covered[w_idx]:
                    redundant = False
                    break
            if redundant:
                # Remove this codeword
                code.pop(ci)
                # Update covered sets
                for w_idx in all_balls[cw_idx]:
                    covered[w_idx].discard(ci)
                # Shift indices for all subsequent codewords
                for w_idx in range(total_words):
                    new_set = set()
                    for old_ci in covered[w_idx]:
                        if old_ci < ci:
                            new_set.add(old_ci)
                        elif old_ci > ci:
                            new_set.add(old_ci - 1)
                    covered[w_idx] = new_set
                improved = True
    
    # Now try greedy addition: find uncovered words and add best codeword
    # First check if we're fully covered
    while time.time() < deadline:
        # Find uncovered words
        uncovered = [w_idx for w_idx in range(total_words) if len(covered[w_idx]) == 0]
        if not uncovered:
            break
        
        # Find the codeword that covers the most uncovered words
        best_cw = None
        best_count = 0
        for idx in range(total_words):
            if time.time() >= deadline:
                break
            count = 0
            for w_idx in all_balls[idx]:
                if len(covered[w_idx]) == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_cw = idx
        
        if best_cw is None or best_count == 0:
            break
        
        # Add this codeword
        new_word = _idx_to_word(best_cw, q, n)
        code.append(new_word)
        new_ci = len(code) - 1
        for w_idx in all_balls[best_cw]:
            covered[w_idx].add(new_ci)
    
    # Final cleanup: try to remove redundant codewords again
    while time.time() < deadline:
        improved = False
        indices = list(range(len(code)))
        random.shuffle(indices)
        for ci in indices:
            if time.time() >= deadline:
                break
            cw_idx = _word_to_idx(code[ci], q)
            redundant = True
            for w_idx in all_balls[cw_idx]:
                if len(covered[w_idx]) == 1 and ci in covered[w_idx]:
                    redundant = False
                    break
            if redundant:
                code.pop(ci)
                for w_idx in all_balls[cw_idx]:
                    covered[w_idx].discard(ci)
                for w_idx in range(total_words):
                    new_set = set()
                    for old_ci in covered[w_idx]:
                        if old_ci < ci:
                            new_set.add(old_ci)
                        elif old_ci > ci:
                            new_set.add(old_ci - 1)
                    covered[w_idx] = new_set
                improved = True
        if not improved:
            break
    
    return code
# EVOLVE-BLOCK-END
