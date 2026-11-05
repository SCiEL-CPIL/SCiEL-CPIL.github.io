"""Solver for ternary covering code K_3(6,1) using bitset-accelerated greedy MIS."""
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Solve covering code K_q(n, radius) using a bitset-accelerated greedy
    maximum independent set heuristic on the square of the Hamming graph.
    
    Each codeword is represented as a Python int (bitset of 729 bits for q=3,n=6).
    Coverage check: (union_of_bitsets).bit_count() == q**n
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    total_words = q ** n
    
    # Precompute bitsets for each word: word i covers itself and all words at distance <= radius
    # For radius=1, q=3, n=6: each word covers 1 + n*(q-1) = 1 + 6*2 = 13 words
    # We use Python's arbitrary-precision integers as bitsets
    word_to_idx = {}
    idx_to_word = []
    all_indices = list(range(total_words))
    
    # Build index mapping
    for idx in all_indices:
        # Convert idx to base-q representation of length n
        digits = []
        tmp = idx
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        word_to_idx[tuple(digits)] = idx
        idx_to_word.append(digits)
    
    # Precompute coverage bitsets
    # coverage_bits[i] = bitmask of all words within distance <= radius from word i
    coverage_bits = [0] * total_words
    for i in range(total_words):
        bits = 1 << i  # self
        for pos in range(n):
            for val in range(q):
                if val == idx_to_word[i][pos]:
                    continue
                # Create new word with position pos set to val
                new_digits = list(idx_to_word[i])
                new_digits[pos] = val
                new_idx = 0
                for d in new_digits:
                    new_idx = new_idx * q + d
                bits |= (1 << new_idx)
        coverage_bits[i] = bits
    
    # Full mask: all total_words bits set
    full_mask = (1 << total_words) - 1
    
    def is_covered(selected_bits):
        """Check if selected codewords cover all words."""
        union = 0
        for b in selected_bits:
            union |= coverage_bits[b]
        return union == full_mask
    
    def coverage_union(selected_bits):
        """Compute the union of coverage bitsets."""
        union = 0
        for b in selected_bits:
            union |= coverage_bits[b]
        return union
    
    # Strategy: Greedy set cover with bitset acceleration
    # Start with a valid code, then try to remove codewords
    
    # Initial code: direct sum of tetracode
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    initial_code = []
    for h in tetracode:
        for tail in range(q ** (n - 4)):
            # Convert tail to base-q digits
            tail_digits = []
            tmp = tail
            for _ in range(n - 4):
                tail_digits.append(tmp % q)
                tmp //= q
            tail_digits.reverse()
            initial_code.append(h + tail_digits)
    
    # Convert to indices
    initial_indices = [word_to_idx[tuple(c)] for c in initial_code]
    
    # Verify initial code is valid
    initial_union = coverage_union(initial_indices)
    assert initial_union == full_mask, "Initial code is not valid"
    
    best_code = [list(idx_to_word[i]) for i in initial_indices]
    best_len = len(initial_indices)
    
    # Local search: greedy removal with bitset checks
    # Maintain current code as a list of indices
    current = initial_indices[:]
    current_union = initial_union
    
    # For efficient removal checks, we maintain the union
    # When removing codeword c, new_union = current_union & ~coverage_bits[c]
    # This is valid iff new_union == full_mask
    
    # We'll do multiple passes with random ordering
    rng = random.Random(42)
    
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() >= deadline:
            break
        
        # Shuffle current order
        shuffled_order = current[:]
        rng.shuffle(shuffled_order)
        
        progress = False
        for c in shuffled_order:
            if time.time() >= deadline:
                break
            
            # Try removing c
            new_union = current_union & ~coverage_bits[c]
            if new_union == full_mask:
                # Valid removal
                current.remove(c)
                current_union = new_union
                progress = True
                if len(current) < best_len:
                    best_len = len(current)
                    best_code = [list(idx_to_word[i]) for i in current]
        
        if not progress:
            # Try swap-based local search: remove one, add another
            # For each codeword in current, try to swap with a non-current word
            swap_progress = False
            for c in current[:]:
                if time.time() >= deadline:
                    break
                
                # Try removing c
                union_without_c = current_union & ~coverage_bits[c]
                
                # Find words not covered by union_without_c
                uncovered = full_mask & ~union_without_c
                if uncovered == 0:
                    # Can just remove
                    current.remove(c)
                    current_union = union_without_c
                    swap_progress = True
                    if len(current) < best_len:
                        best_len = len(current)
                        best_code = [list(idx_to_word[i]) for i in current]
                    continue
                
                # Need to add a word that covers at least one uncovered word
                # Try adding words that cover uncovered words
                # For efficiency, pick a random uncovered word and find codewords covering it
                # Get an uncovered word index
                # Find lowest set bit
                uncovered_word_idx = (uncovered & -uncovered).bit_length() - 1
                uncovered_word_val = 1 << uncovered_word_idx
                
                # Find all codewords that cover this uncovered word
                # A codeword d covers uncovered_word_idx if coverage_bits[d] has bit uncovered_word_idx set
                # We can iterate over all possible codewords not in current
                # For efficiency, we'll try a random subset
                
                # Build set of current for fast lookup
                current_set = set(current)
                
                # Try random non-current codewords
                for _ in range(20):
                    if time.time() >= deadline:
                        break
                    d = rng.randint(0, total_words - 1)
                    if d in current_set:
                        continue
                    
                    # Check if adding d after removing c covers everything
                    new_union = (union_without_c | coverage_bits[d])
                    if new_union == full_mask:
                        # Valid swap
                        current.remove(c)
                        current.append(d)
                        current_union = new_union
                        swap_progress = True
                        if len(current) < best_len:
                            best_len = len(current)
                            best_code = [list(idx_to_word[i]) for i in current]
                        break
            
            if not swap_progress and not progress:
                # Try restart with different random seed
                if pass_num > 0:
                    # Occasionally restart from initial
                    if rng.random() < 0.3:
                        current = initial_indices[:]
                        current_union = initial_union
                else:
                    break
            elif not progress:
                # Only swap progress, continue
                continue
            else:
                continue
    
    return best_code
