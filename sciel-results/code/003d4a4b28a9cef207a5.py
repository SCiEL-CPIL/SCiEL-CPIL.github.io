# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random

def _tetracode_words():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # For the specific task: q=3, n=7, radius=1
    # Use a vectorized greedy approach with criticality-based ordering
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct sum construction
        tetracode = _tetracode_words()
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n-4)]
        else:
            # Just return all words if n < 4 (trivial case)
            return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Vectorized approach for q=3, n=7, radius=1
    # Total space size: 3^7 = 2187
    # Each codeword covers 1 + 7*2 = 15 words (itself + 14 neighbors at distance 1)
    # Lower bound: ceil(2187/15) = 146
    
    # Represent words as integers 0..2186 (base-3 encoding)
    total_words = 3**n
    
    # Precompute: for each word, what's its ball (itself + all words at distance 1)
    # A word at distance 1 differs in exactly one position by ±1 (mod 3)
    
    # Convert word index to its 7 digits
    def idx_to_digits(idx):
        digits = []
        for _ in range(n):
            digits.append(idx % 3)
            idx //= 3
        return digits
    
    def digits_to_idx(digits):
        idx = 0
        for d in reversed(digits):
            idx = idx * 3 + d
        return idx
    
    # Precompute balls: ball[i] = set of words covered by codeword i
    # Actually, for efficiency, let's precompute for each word which codewords cover it
    # But that's expensive. Instead, let's use a different approach.
    
    # Greedy with criticality: 
    # 1. Start with empty code
    # 2. Repeatedly pick the codeword that covers the most uncovered words
    #    (or use criticality: prioritize words with fewest covering options)
    
    # For efficiency, track:
    # - covered: boolean array of size total_words
    # - For each word, count how many potential codewords cover it (for criticality)
    
    # A word w is covered by codeword c if dist(w,c) <= 1
    # dist(w,c) <= 1 means c == w or c differs from w in exactly one position
    
    # Precompute neighbors: for each word, the list of words at distance <= 1
    # That's the word itself + 14 neighbors (7 positions * 2 alternatives)
    
    neighbors = np.zeros((total_words, 15), dtype=np.int32)
    for w in range(total_words):
        digits = idx_to_digits(w)
        neighbors[w, 0] = w
        count = 1
        for pos in range(n):
            for delta in [-1, 1]:
                new_digits = digits.copy()
                new_digits[pos] = (new_digits[pos] + delta) % 3
                neighbors[w, count] = digits_to_idx(new_digits)
                count += 1
    
    # Greedy algorithm:
    # - covered: array of booleans
    # - At each step, find the codeword c that maximizes the number of uncovered words in its ball
    # - But computing this naively is O(2187 * 15) per step, and we need ~150 steps
    
    # Better: use a priority-based approach
    # For each word, track how many times it's "needed" (i.e., how many codewords could cover it)
    
    covered = np.zeros(total_words, dtype=bool)
    code = []
    
    # Count for each word how many words in the space are at distance <= 1 from it
    # This is always 15 (itself + 14 neighbors), so all words are symmetric initially
    
    # Use a simple greedy: repeatedly pick the word that covers the most uncovered words
    # To speed this up, maintain a count of uncovered words in each word's ball
    
    # uncovered_count[c] = number of uncovered words in the ball of word c
    # When we add a codeword, we need to update uncovered_count for all words
    
    # Initialize: uncovered_count[c] = 15 for all c (all words uncovered)
    uncovered_count = np.full(total_words, 15, dtype=np.int32)
    
    while True:
        if time.time() > deadline - 5.0:
            break
        
        if not np.any(~covered):
            break
        
        # Find the word with maximum uncovered_count
        best_idx = np.argmax(uncovered_count)
        best_count = uncovered_count[best_idx]
        
        if best_count <= 0:
            break
        
        # Add this codeword
        code.append(best_idx)
        covered[neighbors[best_idx]] = True
        
        # Update uncovered_count: for each word c, if any of its neighbors was just covered,
        # decrement uncovered_count[c]
        # For each newly covered word w, all words c such that w is in ball(c) need updating
        # w is in ball(c) iff dist(w,c) <= 1 iff c is in ball(w)
        # So for each newly covered word w, decrement uncovered_count[c] for c in neighbors[w]
        
        newly_covered = neighbors[best_idx]
        for w in newly_covered:
            if covered[w]:  # Only if it was just covered (was uncovered before)
                for c in neighbors[w]:
                    uncovered_count[c] -= 1
    
    # Convert indices to digit lists
    result = [idx_to_digits(c) for c in code]
    
    # Verify validity
    # Check that every word is within distance 1 of some codeword
    covered_check = np.zeros(total_words, dtype=bool)
    for c in code:
        covered_check[neighbors[c]] = True
    
    if not np.all(covered_check):
        # Fallback to direct sum
        tetracode = _tetracode_words()
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n-4)]
    
    return result
# EVOLVE-BLOCK-END
