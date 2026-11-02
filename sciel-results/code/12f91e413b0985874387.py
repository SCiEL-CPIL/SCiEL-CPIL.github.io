"""Geometric Hyperplane Projection approach for K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    Uses geometric hyperplane projection to guide greedy selection.
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[i] for i in range(q)]
    if n == 2:
        # All pairs needed for radius 0, but for radius 1 we can do better
        if radius >= 1:
            # Each codeword covers itself and all words at distance 1
            # For n=2, q=3, radius=1: each word covers 1 + 2*(q-1) = 7 words
            # Total 9 words, so 2 codewords might suffice? No, need to check.
            # Actually for n=2, q=3, radius=1: pick (0,0) covers {(0,0),(1,0),(2,0),(0,1),(0,2)}
            # Pick (1,1) covers {(1,1),(0,1),(2,1),(1,0),(1,2)}
            # Union: (0,0),(1,0),(2,0),(0,1),(0,2),(1,1),(2,1),(1,2) - missing (2,2)
            # So need a third. Let's just use greedy.
            pass
    
    # For the specific task: q=3, n=7, radius=1
    # We use a greedy algorithm guided by geometric projections.
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    
    # Create a mapping from word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # For efficiency, precompute neighbors for each word
    # neighbor[word] = set of indices within Hamming distance <= radius
    # This is expensive for large spaces, but for n=7, q=3, total=2187, it's manageable
    
    # Actually, let's think differently. Instead of precomputing all neighbors,
    # we can compute coverage incrementally.
    
    # Geometric projection approach:
    # Project onto 2D using two linear functionals over GF(3)
    # For ternary space, we can use parity-like checks
    
    # Define two projection functions
    # proj1: sum of coordinates mod 3
    # proj2: weighted sum with different weights mod 3
    
    weights1 = [1, 1, 1, 1, 1, 1, 1]  # all ones
    weights2 = [1, 2, 1, 2, 1, 2, 1]  # alternating
    
    # For each word, compute its 2D projection
    # proj = (sum(w_i * weights1_i) mod q, sum(w_i * weights2_i) mod q)
    
    # But this is a static projection. The key insight is to use the projection
    # to identify which regions of the space are under-covered.
    
    # Alternative approach: Use a greedy algorithm where at each step,
    # we pick the codeword that maximizes the number of newly covered words.
    # To speed this up, we track which words are still uncovered.
    
    # For n=7, q=3, the space is 2187 words. Each codeword covers
    # 1 + n*(q-1) = 1 + 7*2 = 15 words.
    # So we need at least ceil(2187/15) = 146 codewords.
    
    # Let's implement a greedy algorithm with efficient coverage tracking.
    
    # Represent words as integers for efficient hashing
    # Base-q representation
    
    def word_to_int(word):
        val = 0
        for i, v in enumerate(word):
            val = val * q + v
        return val
    
    def int_to_word(val, n):
        word = []
        for i in range(n):
            word.append(val % q)
            val //= q
        return word
    
    # Precompute neighbors for each word index
    # neighbors[i] = list of word indices within Hamming distance <= radius
    
    # For radius=1, neighbors of word w are:
    # w itself, and for each position i, change w[i] to each other value
    
    # Let's precompute this
    neighbors = []
    for w in all_words:
        w_int = word_to_int(w)
        nbs = [w_int]
        w_list = list(w)
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val == old_val:
                    continue
                w_list[pos] = new_val
                nbs.append(word_to_int(tuple(w_list)))
            w_list[pos] = old_val
        neighbors.append(nbs)
    
    # Now greedy algorithm
    uncovered = set(range(total_words))
    codewords = []
    
    # To speed up, precompute for each potential codeword which words it covers
    # and use a priority queue or just iterate
    
    # For each step, find the codeword that covers the most uncovered words
    # This is O(total_words * avg_neighbors) per step, which is ~2187*15 = 32805
    # With ~186 steps, that's ~6 million operations, which is fine.
    
    # But we can be smarter: maintain a count of how many uncovered words each
    # candidate codeword would cover.
    
    # Actually, let's just do a simple greedy: at each step, try all remaining
    # uncovered words as potential codewords and pick the one that covers the
    # most currently uncovered words.
    
    # Wait, the optimal codeword doesn't have to be uncovered itself. It could
    # be a word that's already covered but covers many other uncovered words.
    # So we should consider ALL words as potential codewords, not just uncovered ones.
    
    # But that's 2187 candidates per step. Let's do it.
    
    covered = [False] * total_words
    remaining_uncovered = total_words
    
    for step in range(300):  # cap at 300 steps
        if remaining_uncovered == 0:
            break
        
        best_codeword = -1
        best_count = -1
        
        # Try all words as potential codewords
        for w_idx in range(total_words):
            # Count how many currently uncovered words this codeword would cover
            count = 0
            for nb in neighbors[w_idx]:
                if not covered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_codeword = w_idx
        
        if best_count <= 0:
            break
        
        # Add the best codeword
        codewords.append(all_words[best_codeword])
        for nb in neighbors[best_codeword]:
            if not covered[nb]:
                covered[nb] = True
                remaining_uncovered -= 1
    
    # Convert back to list of lists
    return [list(w) for w in codewords]
# EVOLVE-BLOCK-END
