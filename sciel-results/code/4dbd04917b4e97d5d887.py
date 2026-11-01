"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # If we have enough time, try to improve with a greedy set cover approach
    deadline = time.time() + 100  # Leave 20s safety margin for the 120s limit
    
    if n <= 7 and q == 3 and radius == 1:
        # Convert to numpy for vectorized operations
        total_words = q ** n
        # Precompute all words as indices
        all_indices = np.arange(total_words)
        
        # Convert codewords to indices
        def word_to_idx(word):
            idx = 0
            for i, d in enumerate(word):
                idx = idx * q + d
            return idx
        
        def idx_to_word(idx):
            word = []
            for _ in range(n):
                word.append(idx % q)
                idx //= q
            return word[::-1]
        
        # Compute Hamming distance between two words
        def hamming_dist(w1, w2):
            return sum(1 for a, b in zip(w1, w2) if a != b)
        
        # For each codeword, compute which words it covers (within radius)
        def get_covered_indices(codeword):
            covered = set()
            # Generate all words within Hamming distance radius
            for positions in itertools.combinations(range(n), radius):
                for changes in itertools.product(range(1, q), repeat=radius):
                    new_word = list(codeword)
                    for pos, change in zip(positions, changes):
                        old_val = new_word[pos]
                        new_val = (old_val + change) % q
                        new_word[pos] = new_val
                    covered.add(word_to_idx(new_word))
            # Also include the codeword itself
            covered.add(word_to_idx(codeword))
            return covered
        
        # Track which words are covered
        covered_words = set()
        for cw in initial:
            covered_words.update(get_covered_indices(cw))
        
        # If all words are already covered, return initial
        if len(covered_words) == total_words:
            return initial
        
        # Greedy improvement: try to find codewords that cover more uncovered words
        # We'll do a few passes
        best_code = list(initial)
        best_covered = covered_words
        
        for iteration in range(10):  # Limit iterations to stay within time
            if time.time() > deadline:
                break
            
            # Find the best codeword to add
            best_gain = 0
            best_new_codeword = None
            
            # Sample some candidate codewords from uncovered words
            uncovered = [idx for idx in range(total_words) if idx not in best_covered]
            if not uncovered:
                break
            
            # Try a subset of uncovered words as potential new codewords
            sample_size = min(50, len(uncovered))
            import random
            random.seed(42)
            candidates = random.sample(uncovered, sample_size)
            
            for idx in candidates:
                if time.time() > deadline:
                    break
                cand_word = idx_to_word(idx)
                covered = get_covered_indices(cand_word)
                gain = len(covered - best_covered)
                if gain > best_gain:
                    best_gain = gain
                    best_new_codeword = cand_word
            
            if best_new_codeword is not None:
                best_code.append(best_new_codeword)
                best_covered.update(get_covered_indices(best_new_codeword))
            else:
                break
        
        # Return the best code found
        return best_code
    
    return initial
# EVOLVE-BLOCK-END
