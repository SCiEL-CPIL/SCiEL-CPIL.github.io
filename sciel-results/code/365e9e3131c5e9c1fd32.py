"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    # Deadline guard: evaluator kills at 150s wall-clock; return best found with margin.
    t_start = time.time()
    deadline = t_start + 145.0

    if n <= 0:
        return [[]]
    if radius >= n:
        return [[0] * n]

    # For the specific task (q=3, n=7, radius=1), we use a decomposition strategy.
    # The space is {0,1,2}^7. We can view each word as (prefix of length 4, suffix of length 3).
    # The Hamming distance is the sum of distances in prefix and suffix.
    # We need to cover all 2187 words with radius 1.
    #
    # Strategy:
    # 1. Precompute the best radius-1 covering code for length 3 ternary words (the "suffix" part).
    #    A perfect code of length 3 over q=3 with radius 1 is the ternary Hamming code, which has
    #    3^(3-1)/3 = 3 words? No, Hamming code length 4.
    #    For length 3, radius 1: The space has 27 words. A ball of radius 1 has 1 + 3*2 = 7 words.
    #    27/7 ≈ 3.86, so we need at least 4 words.
    #    Can we cover 27 words with 4 codewords of radius 1?
    #    Let's check if a (3, 4, 1) ternary covering code exists.
    #    Actually, let's just compute the minimum size greedily or by search for the suffix part.
    #
    # 2. For each possible prefix (3^4 = 81 prefixes), we decide which suffix codewords to use.
    #    This is complex because the prefix and suffix choices interact.
    #
    # Alternative simpler strategy for n=7, q=3, r=1:
    # The direct sum construction in the seed gives 9 * 3^3 = 243 words.
    # We can try to improve this by using a better code for the full length 7.
    #
    # Let's try a greedy set cover approach on the full space, but that's 2187 elements.
    # A greedy set cover for 2187 elements with balls of size 1 + 7*2 = 15 is feasible.
    #
    # However, greedy set cover often gives poor results. Let's try a more structured approach.
    #
    # Known result: K_3(7,1) = 186.
    # The seed gives 243. We need to get closer to 186.
    #
    # Let's use a two-stage approach:
    # Stage 1: Find a good covering code for length 4 (prefix).
    # Stage 2: For each prefix codeword, choose suffix codewords to cover the remaining space.
    #
    # Actually, a better decomposition:
    # Split into 4 and 3.
    # Let C4 be a code for length 4, radius 1.
    # Let C3 be a code for length 3, radius 1.
    # The product C4 x C3 is a code for length 7, radius 2 (since distances add).
    # We need radius 1. So we can't just take the product.
    #
    # Instead, we can use the following:
    # For each prefix p in {0,1,2}^4, we need to cover the set of words (p, s) for s in {0,1,2}^3.
    # A codeword (c, d) covers (p, s) if dist(c, p) + dist(d, s) <= 1.
    # This means either:
    #   dist(c, p) = 0 and dist(d, s) <= 1, OR
    #   dist(c, p) = 1 and dist(d, s) = 0.
    #
    # So for a fixed prefix c, the codeword (c, d) covers:
    #   - All suffixes s with dist(d, s) <= 1 (i.e., the ball B(d, 1)) for prefix c.
    #   - For each prefix p with dist(c, p) = 1, the codeword covers only the suffix s = d (dist=0).
    #
    # This structure is complex to optimize directly.
    #
    # Let's try a different approach: use the known structure of the ternary Hamming code.
    # The ternary Hamming code of length 4 is a perfect code of radius 1 with 9 codewords.
    # The seed uses this for the first 4 coordinates and all 27 words for the last 3.
    # This gives 9 * 27 = 243 codewords.
    #
    # Can we do better?
    # The lower bound is 156, best known is 186.
    #
    # Let's try to use a code for the last 3 coordinates that is smaller than 27.
    # If we use a radius-1 covering code for length 3 with k words, then for each of the 9 prefixes,
    # we need to cover the 27 suffixes. But the interaction with the prefix distance is the issue.
    #
    # Let's implement a greedy set cover on the full 2187-word space.
    # It might not reach 186, but it might beat 243.
    #
    # Greedy set cover:
    # 1. Start with an empty set of codewords.
    # 2. At each step, choose the codeword that covers the most uncovered words.
    # 3. Repeat until all words are covered.
    #
    # This is O(2187^2 * 15) which is too slow in Python if done naively.
    # We can use NumPy to speed it up.
    #
    # Actually, let's precompute the balls for all 2187 possible codewords.
    # Each ball has 15 words.
    # Then we can use a greedy algorithm with NumPy operations.

    if q == 3 and n == 7 and radius == 1:
        # Special case optimization for the known task
        return _construct_ternary_7_1(deadline)
    
    # Fallback: greedy set cover for general case
    return _greedy_cover(q, n, radius, deadline)


def _construct_ternary_7_1(deadline):
    """Optimized construction for q=3, n=7, radius=1."""
    q, n, radius = 3, 7, 1
    total_words = q ** n  # 2187
    
    # Precompute all words as integers 0..2186
    # word[i] = w0 * 3^6 + w1 * 3^5 + ... + w6
    # We'll represent words as integers for fast bit manipulation (well, ternary manipulation)
    
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute Hamming distances from each word to all others
    # This is a 2187x2187 matrix, which is 4.8M entries. Feasible.
    # But we only need to know which words are within distance 1.
    
    # For each word, compute the set of words within distance 1
    # We'll represent this as a boolean matrix or as lists.
    
    # Let's precompute the balls.
    # ball[i] = set of words within distance 1 of word i
    # We'll use a 2187x2187 boolean matrix for fast operations.
    
    # Actually, let's use a different approach.
    # Greedy set cover with bitsets.
    
    # Represent the set of uncovered words as a bitset (Python int with 2187 bits)
    # But Python ints are arbitrary precision, so this works.
    
    # Precompute for each word i, the bitset of words covered by choosing i as codeword.
    # cover[i] = bitset of words within distance 1 of word i
    
    # First, compute all words as integers
    word_to_int = {}
    int_to_word = {}
    for w in all_words:
        val = 0
        for digit in w:
            val = val * q + digit
        word_to_int[tuple(w)] = val
        int_to_word[val] = list(w)
    
    # Precompute the cover bitsets
    # For each word i, find all words j with dist(i, j) <= 1
    covers = [0] * total_words
    for i in range(total_words):
        wi = int_to_word[i]
        # Distance 0: itself
        covers[i] |= (1 << i)
        # Distance 1: change one position
        for pos in range(n):
            for new_val in range(q):
                if new_val == wi[pos]:
                    continue
                wj = wi[:]
                wj[pos] = new_val
                j = word_to_int[tuple(wj)]
                covers[i] |= (1 << j)
    
    # Greedy set cover
    uncovered = (1 << total_words) - 1  # All bits set
    selected = []
    
    # To speed up, we can use a priority queue or just iterate
    # For each step, find the word that covers the most uncovered words
    # This is O(2187) per step, and we might need ~200 steps, so ~437K operations. Feasible.
    
    while uncovered != 0:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_count = -1
        for i in range(total_words):
            if (1 << i) & uncovered == 0:
                # i is already covered, but we can still select it as a codeword
                # Actually, we should consider all i, not just uncovered ones
                pass
            count = bin(covers[i] & uncovered).count('1')
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1 or best_count <= 0:
            break
        
        selected.append(best_idx)
        uncovered &= ~covers[best_idx]
    
    # Convert back to word lists
    result = [int_to_word[i] for i in selected]
    
    # Verify coverage
    if _verify_coverage(result, q, n, radius):
        return result
    
    # Fallback to seed construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _greedy_cover(q, n, radius, deadline):
    """General greedy set cover for covering codes."""
    total_words = q ** n
    if total_words > 100000:
        # Too large for greedy
        # Use direct sum construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Fallback: all zeros
        return [[0] * n]
    
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    word_to_int = {}
    int_to_word = {}
    for w in all_words:
        val = 0
        for digit in w:
            val = val * q + digit
        word_to_int[tuple(w)] = val
        int_to_word[val] = list(w)
    
    # Precompute covers
    covers = [0] * total_words
    for i in range(total_words):
        wi = int_to_word[i]
        covers[i] |= (1 << i)
        # Generate all words within distance radius
        # For radius 1, just change one position
        if radius == 1:
            for pos in range(n):
                for new_val in range(q):
                    if new_val == wi[pos]:
                        continue
                    wj = wi[:]
                    wj[pos] = new_val
                    j = word_to_int[tuple(wj)]
                    covers[i] |= (1 << j)
        else:
            # For larger radius, we need to generate all words within distance radius
            # This is more expensive
            from itertools import combinations
            for k in range(1, radius + 1):
                for positions in combinations(range(n), k):
                    for changes in itertools.product(range(q), repeat=k):
                        if all(changes[i] == wi[positions[i]] for i in range(k)):
                            continue
                        wj = wi[:]
                        for idx, pos in enumerate(positions):
                            wj[pos] = changes[idx]
                        j = word_to_int[tuple(wj)]
                        covers[i] |= (1 << j)
    
    uncovered = (1 << total_words) - 1
    selected = []
    
    while uncovered != 0:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_count = -1
        for i in range(total_words):
            count = bin(covers[i] & uncovered).count('1')
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1 or best_count <= 0:
            break
        
        selected.append(best_idx)
        uncovered &= ~covers[best_idx]
    
    result = [int_to_word[i] for i in selected]
    
    if _verify_coverage(result, q, n, radius):
        return result
    
    return [[0] * n]


def _verify_coverage(codewords, q, n, radius):
    """Verify that all words are within distance radius of some codeword."""
    if not codewords:
        return n == 0
    
    # Create a set of codeword tuples for fast lookup
    cw_set = set(tuple(cw) for cw in codewords)
    
    # Check all words
    for word in itertools.product(range(q), repeat=n):
        found = False
        for cw in cw_set:
            dist = sum(1 for a, b in zip(word, cw) if a != b)
            if dist <= radius:
                found = True
                break
        if not found:
            return False
    return True
# EVOLVE-BLOCK-END
