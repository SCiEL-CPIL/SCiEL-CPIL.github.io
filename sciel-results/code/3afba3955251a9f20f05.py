"""Football pool problem: ternary covering code K_3(7,1).

Strategy:
1. Use the known best construction for K_3(7,1) with 186 codewords (van Laarhoven et al., 1989)
   as a starting point if we can construct it.
2. Since explicitly listing all 186 codewords is tedious, we use a greedy set-cover approach
   that is fast enough for n=7, q=3, radius=1 (2187 words, each covered by 1 + 2*7 = 15 words).
3. The greedy algorithm picks the codeword that covers the most uncovered words at each step.
   With 2187 words and ~186 steps, each step scans 2187 candidates * 15 neighbors = ~32k ops,
   total ~6M ops, well within 120s.

We also try to improve upon the greedy result with a local search (swap/replacement) if time allows,
but the primary goal is to match or beat 186.

Actually, the greedy set-cover for covering codes often gets close to optimal for small instances.
For K_3(7,1), the optimal is 186. Greedy might get slightly worse (e.g., 190-200).
To do better, we can use a more sophisticated approach: start from a good initial code and do local search.

Let's use a hybrid:
- Generate a good initial code using a structured method or greedy.
- Then do local search: try replacing one codeword with another to reduce the total size while maintaining coverage.

But given time constraints, let's first implement a fast greedy and see.

Actually, a better approach for this specific small instance: use integer programming? No, too slow.

Let's use a greedy algorithm with a twist: process words in a smart order, and use bitsets for speed.

We represent the set of all 2187 words as a bitset (or a boolean array).
For each candidate codeword, we precompute which words it covers (itself + 14 neighbors).
Greedy: repeatedly pick the codeword that covers the most currently-uncovered words.

We can speed this up by using numpy or by using Python sets, but 2187 is small.

Let's implement greedy with efficient counting.
"""

import itertools
import numpy as np
import time


def construct(q, n, radius):
    # Only handle the specific case q=3, n=7, radius=1 efficiently.
    # For other parameters, fall back to a simple method.
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1()
    
    # Fallback: use the direct sum method from the seed if possible, else greedy.
    # For generality, do a greedy set cover.
    return _greedy_cover(q, n, radius)


def _construct_3_7_1():
    """Construct a covering code for q=3, n=7, radius=1.
    
    We use a greedy set-cover approach followed by local search refinement.
    1. Greedy: Pick codewords that cover the most uncovered words.
    2. Local Search: Try to remove redundant codewords. Repeat until no more can be removed.
    """
    # Generate all words and map to indices
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)  # 2187
    
    # Precompute coverage masks as bitsets (using Python integers for efficiency)
    # Each integer represents a set of covered words. Bit i is 1 if word i is covered.
    covers = [0] * N
    for i, w in enumerate(all_words):
        mask = 0
        # Cover itself
        mask |= (1 << i)
        # Cover neighbors (distance 1)
        for pos in range(7):
            for delta in [1, 2]:
                new_w = list(w)
                new_w[pos] = (w[pos] + delta) % 3
                idx = word_to_idx[tuple(new_w)]
                mask |= (1 << idx)
        covers[i] = mask
    
    # Greedy Set Cover
    uncovered_mask = (1 << N) - 1
    code_indices = []
    
    while uncovered_mask:
        best_idx = -1
        best_count = -1
        # Iterate through all possible codewords
        for i in range(N):
            # Count how many uncovered words this codeword covers
            # We can optimize by skipping if the word is already covered by previous choices?
            # No, a covered word might still cover other uncovered words.
            count = bin(uncovered_mask & covers[i]).count("1")
            if count > best_count:
                best_count = count
                best_idx = i
                if count == 15: # Max possible is 15 (1 self + 14 neighbors). If we find 15, it's optimal for this step.
                    break
        
        code_indices.append(best_idx)
        uncovered_mask &= ~covers[best_idx]
        
        # Early exit if we've exceeded a known good bound? No, we want to minimize.
        # But if we are already worse than 186, we might want to switch strategies?
        # For now, just finish greedy.
        
    # Local Search: Try to remove redundant codewords
    # We maintain the union of covers of the current code.
    # If removing a codeword doesn't change the union, it's redundant.
    
    current_code = list(code_indices)
    
    # Optimization: Precompute covers as a list of integers.
    # We can try to remove codewords in a random order or specific order to find more redundancies.
    
    # Simple local search: repeatedly try to remove any codeword that is redundant.
    improved = True
    while improved:
        improved = False
        # Calculate total coverage
        total_cover = 0
        for idx in current_code:
            total_cover |= covers[idx]
            
        # Check if full coverage is maintained (should be, but good to verify)
        if total_cover != (1 << N) - 1:
            # This shouldn't happen if we started with a valid cover
            break
            
        # Try removing each codeword
        for i in range(len(current_code)):
            idx_to_remove = current_code[i]
            # Coverage without this codeword
            new_cover = 0
            for j in range(len(current_code)):
                if j != i:
                    new_cover |= covers[current_code[j]]
            
            if new_cover == (1 << N) - 1:
                # Redundant! Remove it.
                current_code.pop(i)
                improved = True
                break # Restart the loop as the list changed
                
    # Convert indices back to words
    return [list(all_words[i]) for i in current_code]


def _greedy_cover(q, n, radius):
    """General greedy set cover for covering codes."""
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    # Precompute covers for each word
    covers = []
    for w in all_words:
        covered = set()
        covered.add(word_to_idx[w])
        # Generate all words within Hamming distance radius
        # For small radius, enumerate all changes
        for r in range(1, radius + 1):
            for positions in itertools.combinations(range(n), r):
                for deltas in itertools.product(range(1, q), repeat=r):
                    new_w = list(w)
                    for pos, d in zip(positions, deltas):
                        new_w[pos] = (w[pos] + d) % q
                    covered.add(word_to_idx[tuple(new_w)])
        covers.append(covered)
    
    uncovered = set(range(N))
    code = []
    while uncovered:
        best = None
        best_count = -1
        for i in range(N):
            count = len(covers[i] & uncovered)
            if count > best_count:
                best_count = count
                best = i
        code.append(list(all_words[best]))
        uncovered -= covers[best]
    
    return code
