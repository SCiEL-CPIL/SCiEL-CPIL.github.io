# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    # Deadline guard: evaluator kills at 150s; use 115s budget for safety.
    deadline = time.time() + 115.0

    # Build the ternary tetracode (perfect code of length 4, covering radius 1).
    # 9 codewords; each covers 1 + 4*2 = 9 words; 9*9 = 81 = 3^4.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    # Direct-sum construction: 9 * 3^(n-4) codewords.
    # For n=8: 9 * 81 = 729 codewords.
    if n <= 4:
        # Fallback for small n (not the main case)
        if n == 0:
            return [[]]
        # Brute force for tiny n
        all_words = list(itertools.product(range(q), repeat=n))
        return [list(w) for w in all_words]

    # Generate the direct-sum code
    tails = list(itertools.product(range(q), repeat=n - 4))
    code = [h + list(tail) for h in tetracode for tail in tails]

    # Now try to prune redundant codewords.
    # A codeword is redundant if removing it still leaves every word within radius.
    # We use a greedy removal approach: repeatedly remove a codeword whose
    # "exclusive" covered words (words covered only by it) are zero or can be
    # covered by remaining codewords.
    #
    # For efficiency, we compute for each codeword the set of words it uniquely covers.
    # If a codeword has no uniquely covered words, it can be removed.
    # We iterate this process.

    # For n=8, q=3, radius=1: total space = 6561 words.
    # Each codeword covers 1 + 8*2 = 17 words.
    # 729 codewords cover 729*17 = 12393 "slots" but only 6561 words, so lots of overlap.

    # Precompute neighbors for each codeword
    def get_neighbors(cw):
        """Get all words within Hamming distance radius of cw."""
        neighbors = [tuple(cw)]
        for i in range(n):
            for val in range(q):
                if val != cw[i]:
                    new_word = list(cw)
                    new_word[i] = val
                    neighbors.append(tuple(new_word))
        return neighbors

    # For pruning, we need to track coverage counts.
    # Given the large number of codewords (729) and words (6561), we can use arrays.
    # Map each word to an index: word index = sum(cw[i] * q^(n-1-i))
    
    total_words = q ** n
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx
    
    # Build coverage count array
    coverage = [0] * total_words
    for cw in code:
        for nb in get_neighbors(cw):
            coverage[word_to_idx(nb)] += 1

    # Now iteratively remove codewords with no exclusive coverage.
    # A codeword has exclusive coverage if some word in its neighborhood has coverage == 1.
    # We need to track which codewords are still present.
    
    active = [True] * len(code)
    active_count = len(code)
    
    # Precompute neighbor lists for all codewords
    all_neighbors = [get_neighbors(cw) for cw in code]
    
    # Iterate removal
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for i in range(len(code)):
            if not active[i]:
                continue
            # Check if any word in neighbors[i] has coverage == 1 (and that word is still "needed")
            # Actually, we need: is there a word w in neighbors[i] such that coverage[w] == 1?
            # If yes, codeword i is essential.
            # If no, codeword i is redundant.
            essential = False
            for nb in all_neighbors[i]:
                if coverage[word_to_idx(nb)] == 1:
                    essential = True
                    break
            if not essential:
                # Remove codeword i
                active[i] = False
                active_count -= 1
                changed = True
                for nb in all_neighbors[i]:
                    coverage[word_to_idx(nb)] -= 1
    
    # Build final code
    result = [code[i] for i in range(len(code)) if active[i]]
    
    # Verify validity (quick check)
    # Build a set of covered words
    covered = set()
    for cw in result:
        for nb in get_neighbors(cw):
            covered.add(nb)
    if len(covered) < total_words:
        # Fallback to full code
        return code
    
    return result
# EVOLVE-BLOCK-END
