"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct(q, n, radius):
    """
    Greedy set-cover algorithm to find a small covering code.
    For each candidate codeword, compute the set of words within Hamming distance <= radius.
    Iteratively pick the codeword that covers the most currently uncovered words.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words <= 2**20:
        return _construct_greedy_bitmask(q, n, radius)
    else:
        return _construct_greedy_general(q, n, radius)


def _construct_greedy_bitmask(q, n, radius):
    """Greedy set-cover using bitmasks for small spaces."""
    total_words = q ** n
    
    # Precompute index mappings using numpy for speed
    # idx = sum(word[i] * q^(n-1-i)) for i in range(n)
    
    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
    
    def word_to_idx(word):
        return sum(word[i] * q_powers[n-1-i] for i in range(n))
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute ball masks using efficient neighbor generation
    ball_masks = np.zeros(total_words, dtype=object)
    
    # For radius=1, each ball has 1 + n*(q-1) words
    # For general radius, we need to be more careful
    
    # Precompute all neighbors for each index
    # For efficiency, precompute the list of neighbors for each position and value
    
    # neighbor_lists[pos][val] = list of indices that differ only at position pos from a word with val at pos
    # But this depends on the rest of the word, so we need a different approach
    
    # Better: for each index, generate its neighbors directly
    # For radius=1: neighbors are the word itself plus all words differing in exactly one position
    
    # Precompute: for each position pos and each value val, and for each "prefix-suffix" pattern
    # This is complex, so let's just generate neighbors on the fly but cache the results
    
    # Actually, let's precompute ball_masks more efficiently
    # For radius=1, we can use the following approach:
    # For each index idx, convert to word, then for each position, change to each other value
    
    # Let's precompute a mapping from (position, old_val, new_val) to the offset in index space
    # offset = (new_val - old_val) * q_powers[n-1-pos]
    
    # Precompute offsets for each position and value change
    offsets = {}
    for pos in range(n):
        for old_val in range(q):
            for new_val in range(q):
                if old_val != new_val:
                    offset = (new_val - old_val) * q_powers[n-1-pos]
                    offsets[(pos, old_val, new_val)] = offset
    
    # Now precompute ball masks
    for idx in range(total_words):
        mask = 1 << idx
        if radius >= 1:
            # Get the word to know the current values at each position
            # But we can also compute neighbors using modular arithmetic
            # For each position, try all other values
            # We need to know the current value at each position
            word = idx_to_word(idx)
            for pos in range(n):
                cur_val = word[pos]
                for new_val in range(q):
                    if new_val != cur_val:
                        nidx = idx + (new_val - cur_val) * q_powers[n-1-pos]
                        mask |= 1 << nidx
        
        if radius >= 2:
            # For radius >= 2, we need to include words at distance 2
            # This is more complex. For now, handle radius=1 optimally
            # and use a more general approach for larger radii
            word = idx_to_word(idx)
            # Generate all words at distance exactly 2
            for pos1 in range(n):
                for pos2 in range(pos1+1, n):
                    for val1 in range(q):
                        if val1 == word[pos1]:
                            continue
                        for val2 in range(q):
                            if val2 == word[pos2]:
                                continue
                            # Compute the index
                            nidx = idx + (val1 - word[pos1]) * q_powers[n-1-pos1] + (val2 - word[pos2]) * q_powers[n-1-pos2]
                            mask |= 1 << nidx
        
        ball_masks[idx] = mask
    
    # Convert to a list of Python ints for easier manipulation
    # Actually, let's use Python ints directly since we need bitwise operations
    
    # Re-do with Python ints
    ball_masks_list = [0] * total_words
    for idx in range(total_words):
        mask = 1 << idx
        if radius >= 1:
            word = idx_to_word(idx)
            for pos in range(n):
                cur_val = word[pos]
                for new_val in range(q):
                    if new_val != cur_val:
                        nidx = idx + (new_val - cur_val) * q_powers[n-1-pos]
                        mask |= 1 << nidx
        
        if radius >= 2:
            word = idx_to_word(idx)
            for pos1 in range(n):
                for pos2 in range(pos1+1, n):
                    for val1 in range(q):
                        if val1 == word[pos1]:
                            continue
                        for val2 in range(q):
                            if val2 == word[pos2]:
                                continue
                            nidx = idx + (val1 - word[pos1]) * q_powers[n-1-pos1] + (val2 - word[pos2]) * q_powers[n-1-pos2]
                            mask |= 1 << nidx
        
        ball_masks_list[idx] = mask
    
    # Greedy set cover
    uncovered = (1 << total_words) - 1
    codewords = []
    
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            covered = ball_masks_list[idx] & uncovered
            count = bin(covered).count('1')
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1:
            break
        
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks_list[best_idx]
    
    return codewords


def _construct_greedy_general(q, n, radius):
    """Greedy set-cover for larger spaces using a different representation."""
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute ball masks
    ball_masks = [0] * total_words
    
    for idx in range(total_words):
        center = idx_to_word(idx)
        mask = 0
        if radius == 0:
            mask = 1 << idx
        elif radius == 1:
            mask = 1 << idx
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        mask |= 1 << nidx
        else:
            def generate_neighbors(center, remaining_radius, pos=0):
                if pos == n:
                    yield list(center)
                    return
                if remaining_radius >= 0:
                    yield from generate_neighbors(center, remaining_radius, pos + 1)
                if remaining_radius > 0:
                    for val in range(q):
                        if val != center[pos]:
                            old_val = center[pos]
                            center[pos] = val
                            yield from generate_neighbors(center, remaining_radius - 1, pos + 1)
                            center[pos] = old_val
            
            for neighbor in generate_neighbors(center, radius):
                nidx = word_to_idx(neighbor)
                mask |= 1 << nidx
        
        ball_masks[idx] = mask
    
    uncovered = (1 << total_words) - 1
    codewords = []
    
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            covered = ball_masks[idx] & uncovered
            count = bin(covered).count('1')
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1:
            break
        
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]
    
    return codewords
# EVOLVE-BLOCK-END
