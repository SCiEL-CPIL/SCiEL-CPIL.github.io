"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import sys


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # For the specific task: q=3, n=8, radius=1
    # But write general code that handles these parameters.
    
    total_words = q ** n
    # Precompute all words as integers 0..total_words-1
    # We represent each word as an integer in base-q.
    
    # For radius=1, a word covers itself and all words at Hamming distance 1.
    # Number of words covered by one codeword: 1 + n*(q-1)
    # For q=3, n=8: 1 + 8*2 = 17 words per codeword.
    # Lower bound: 6561 / 17 ≈ 386. Best known: 486.
    
    # Strategy: Greedy set cover with vectorized marginal coverage,
    # followed by local search (swap/removal) to reduce count.
    
    # Precompute the coverage sets for all possible words.
    # Each word i covers a set of words. We need to represent this efficiently.
    
    # For q=3, n=8, radius=1:
    # Word i covers: i itself, and for each position p, the q-1 variants.
    # Total 17 words covered per codeword.
    
    # Precompute neighbors for each word.
    # neighbors[i] = list of words covered by codeword i (including i itself)
    
    if q == 3 and n == 8 and radius == 1:
        # Specialized fast path
        return _construct_ternary_8_1(deadline, start_time)
    else:
        # General greedy set cover
        return _construct_general(q, n, radius, deadline, start_time)


def _construct_ternary_8_1(deadline, start_time):
    q, n, radius = 3, 8, 1
    total = 3 ** 8  # 6561
    
    # Precompute coverage: for each word i (0..6560), which words does it cover?
    # A word covers itself and all words at Hamming distance <= 1.
    # For radius=1: itself + n*(q-1) = 1 + 16 = 17 words.
    
    # Represent each word as a tuple of digits, or as an integer.
    # Let's use integer representation: word = d0 + d1*3 + d2*9 + ... + d7*3^7
    
    # Precompute the list of covered words for each possible codeword.
    # covered[i] = set of word indices covered by codeword i
    
    # For efficiency, precompute as a boolean matrix or as lists.
    # 6561 words, each covers 17 words. Total entries: 6561 * 17 = 111,537.
    
    # Build coverage lists
    powers = [3**i for i in range(n)]
    
    # For each word i, compute the 17 words it covers.
    # Word i has digits: i // powers[j] % 3 for j=0..7
    # For each position j, the variants are: replace digit at j with (digit+1)%3 and (digit+2)%3
    
    # Precompute all covered sets
    covered = []
    for i in range(total):
        # Get digits of i
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        
        cov = [i]
        for j in range(n):
            d = digits[j]
            for nd in range(3):
                if nd != d:
                    # Compute the word with digit j changed to nd
                    new_val = i + (nd - d) * powers[j]
                    cov.append(new_val)
        covered.append(cov)
    
    # Greedy set cover:
    # Maintain: uncovered[i] = 1 if word i is not yet covered, 0 otherwise.
    # At each step, pick the codeword that covers the most uncovered words.
    # Then mark those words as covered.
    
    # Use numpy for speed.
    # uncovered_mask: boolean array of length total, True if uncovered.
    uncovered = np.ones(total, dtype=bool)
    selected = []
    
    # Precompute covered as a 2D array for fast indexing
    # covered_arr[i] = array of 17 word indices
    covered_arr = np.array(covered, dtype=np.int32)
    
    # Greedy: at each iteration, for each candidate codeword, count how many
    # of its 17 covered words are still uncovered. Pick the max.
    
    # This is O(total * 17) per iteration in the worst case, but we can vectorize.
    # For 6561 candidates, each with 17 words, that's 6561*17 = 111k operations per iteration.
    # We need about 486 iterations. Total: 486 * 111k ≈ 54M operations. Should be fine in numpy.
    
    # Actually, let's be smarter. We can compute marginal coverage incrementally.
    # But for simplicity and speed, let's just do the full scan each time with numpy.
    
    # Actually, a better approach: maintain a count array.
    # For each word w, count[w] = number of selected codewords that cover w.
    # When we select a new codeword c, for each w in covered[c], increment count[w].
    # A word is covered if count[w] > 0.
    # Marginal gain of selecting c = number of w in covered[c] with count[w] == 0.
    
    count = np.zeros(total, dtype=np.int32)
    selected = []
    
    # Precompute covered_arr as int32 array of shape (total, 17)
    # covered_arr is already built above.
    
    # For speed, let's use a different approach:
    # At each step, compute marginal gain for all candidates.
    # marginal[i] = sum(uncovered[covered_arr[i]])
    # This is: for each i, count how many of its 17 covered words are uncovered.
    
    # We can do this with numpy:
    # uncovered[covered_arr] gives a (total, 17) boolean array.
    # Then sum along axis 1.
    
    # But this creates a 6561*17 = 111k element array each iteration, which is fine.
    
    while True:
        if time.time() > deadline - 5.0:
            break
        
        # Check if all covered
        if not uncovered.any():
            break
        
        # Compute marginal gains
        # uncovered[covered_arr] -> (6561, 17) boolean array
        # This might be slow due to fancy indexing. Let's try it.
        
        # Alternative: precompute a matrix M where M[i, w] = 1 if w in covered[i].
        # Then marginal = M @ uncovered
        # M is 6561 x 6561, which is 43M entries. Too large for dense, but sparse.
        # Each row has 17 ones. Sparse matrix would work.
        
        # Let's use the fancy indexing approach and see if it's fast enough.
        cov_mask = uncovered[covered_arr]  # (6561, 17)
        marginal = cov_mask.sum(axis=1)  # (6561,)
        
        best = np.argmax(marginal)
        if marginal[best] == 0:
            # No improvement possible, but uncovered words remain.
            # This shouldn't happen if the code is valid.
            break
        
        selected.append(int(best))
        # Update uncovered
        uncovered[covered_arr[best]] = False
        # Update count
        count[covered_arr[best]] += 1
    
    # Now try to reduce the code size via local search.
    # Try removing each codeword: if after removal, all words are still covered, keep it removed.
    # Also try swapping: remove one, add another if it helps.
    
    # First, try simple removal.
    # Rebuild count from selected.
    count = np.zeros(total, dtype=np.int32)
    for c in selected:
        count[covered_arr[c]] += 1
    
    changed = True
    while changed:
        if time.time() > deadline - 3.0:
            break
        changed = False
        new_selected = []
        for c in selected:
            # Try removing c
            count[covered_arr[c]] -= 1
            if (count > 0).all():
                # c is redundant, don't add it back
                changed = True
            else:
                count[covered_arr[c]] += 1
                new_selected.append(c)
        if changed:
            selected = new_selected
            # Rebuild count
            count = np.zeros(total, dtype=np.int32)
            for c in selected:
                count[covered_arr[c]] += 1
    
    # Convert selected indices back to codewords (lists of digits)
    codewords = []
    for i in selected:
        digits = []
        temp = i
        for j in range(n):
            digits.append(temp % 3)
            temp //= 3
        codewords.append(digits)
    
    return codewords


def _construct_general(q, n, radius, deadline, start_time):
    """General greedy set cover for arbitrary q, n, radius."""
    total = q ** n
    
    # Precompute coverage for each word
    # For radius r, a word covers all words within Hamming distance r.
    # For small radius, this is manageable.
    
    powers = [q**i for i in range(n)]
    
    covered = []
    for i in range(total):
        # Get digits
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % q
            temp //= q
        
        # Generate all words within Hamming distance radius
        cov = set()
        cov.add(i)
        
        # For radius=1, just add the n*(q-1) neighbors
        if radius == 1:
            for j in range(n):
                d = digits[j]
                for nd in range(q):
                    if nd != d:
                        new_val = i + (nd - d) * powers[j]
                        cov.add(new_val)
        else:
            # General case: generate all combinations
            # For small radius and n, this is feasible
            from itertools import combinations
            for r in range(1, radius + 1):
                for positions in combinations(range(n), r):
                    for changes in itertools.product(range(q), repeat=r):
                        new_val = i
                        for pos, ch in zip(positions, changes):
                            d = digits[pos]
                            if ch != d:
                                new_val += (ch - d) * powers[pos]
                        cov.add(new_val)
        
        covered.append(list(cov))
    
    covered_arr = np.array(covered, dtype=np.int32)
    
    uncovered = np.ones(total, dtype=bool)
    selected = []
    
    while True:
        if time.time() > deadline - 5.0:
            break
        if not uncovered.any():
            break
        
        cov_mask = uncovered[covered_arr]
        marginal = cov_mask.sum(axis=1)
        best = np.argmax(marginal)
        if marginal[best] == 0:
            break
        
        selected.append(int(best))
        uncovered[covered_arr[best]] = False
    
    # Local search: removal
    count = np.zeros(total, dtype=np.int32)
    for c in selected:
        count[covered_arr[c]] += 1
    
    changed = True
    while changed:
        if time.time() > deadline - 3.0:
            break
        changed = False
        new_selected = []
        for c in selected:
            count[covered_arr[c]] -= 1
            if (count > 0).all():
                changed = True
            else:
                count[covered_arr[c]] += 1
                new_selected.append(c)
        if changed:
            selected = new_selected
            count = np.zeros(total, dtype=np.int32)
            for c in selected:
                count[covered_arr[c]] += 1
    
    codewords = []
    for i in selected:
        digits = []
        temp = i
        for j in range(n):
            digits.append(temp % q)
            temp //= q
        codewords.append(digits)
    
    return codewords

# EVOLVE-BLOCK-END
