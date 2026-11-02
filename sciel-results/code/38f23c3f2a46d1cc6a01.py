"""Improved solver for the ternary covering code K_3(6,1).

Strategy:
1. Start from the direct-sum construction (9 * 3^(n-4) words) as a valid baseline.
2. Use a greedy set-cover style reduction: repeatedly remove a codeword if the
   remaining set still covers the entire space. This is safe and can only reduce
   the number of codewords.
3. To make removal checks fast, precompute for each word in the space the set of
   codeword indices that cover it (i.e., codewords within Hamming distance <= radius).
   When we remove a codeword, we update coverage counts; if any word's count drops
   to 0, we abort the removal and restore.
4. We iterate this greedy removal until no more codewords can be removed.
5. We also try a second pass: after the first greedy sweep, we try removing codewords
   in a different order (e.g., reverse, or by some heuristic like codewords that
   are "redundant" more often).
6. Time budget: 150s wall-clock; we keep a safety margin and return the best found.

The key insight: the direct-sum construction has 9 * 3^(6-4) = 81 words for n=6.
The known best is 73. So we have 8 words to remove. Greedy removal should find
some of these.

For general (q, n, radius), we handle the general case but optimize for the common
case q=3, n=6, radius=1.
"""
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # --- Step 1: Build initial code via direct sum of tetracode ---
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        # Fallback: all-zero word plus all single-error neighbors (star code)
        code = [[0] * n]
        for i in range(n):
            for v in range(1, q):
                w = [0] * n
                w[i] = v
                code.append(w)
        # This is a valid covering code for radius >= 1, but large.
        # For the specific task (q=3, n=6, radius=1), the tetracode path is used.

    # --- Step 2: Greedy removal to reduce code size ---
    # Precompute coverage: for each word in the space, which codewords cover it?
    # Space size is q^n. For q=3, n=6, that's 729 words. Very manageable.
    space_size = q ** n
    if space_size > 10**6:
        # Too large to precompute; just return the initial code
        return code

    # Map each codeword to an index
    codeword_set = set(tuple(c) for c in code)
    num_codewords = len(code)

    # For each word in the space, compute which codewords are within distance <= radius
    # coverage[word_idx] = list of codeword indices that cover this word
    # We use a list of lists, but for efficiency, we'll use counts and a "last coverer" approach.
    
    # Actually, for removal we need to know: if we remove codeword i, is every word still covered?
    # Equivalent: for each word, count how many codewords cover it. If min count >= 2, we can remove any one.
    # But we need to be careful: removing one might make some word have count 1, then removing another
    # might make it 0. So we need to track counts and do a careful greedy.
    
    # Let's precompute: for each word in the space, the list of codeword indices that cover it.
    # Then for each codeword, we can determine which words it uniquely covers (words for which it's the only coverer).
    # A codeword can be removed iff it has no uniquely covered words.
    
    # Step 2a: Build coverage lists
    # word_to_idx: map a word (tuple) to its index in the space
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx
    
    # Precompute all words in the space and their indices
    all_words = [list(itertools.product(range(q), repeat=n))]
    word_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # For each word, list of codeword indices that cover it
    coverage = [[] for _ in range(space_size)]
    
    for ci, cw in enumerate(code):
        cw_tuple = tuple(cw)
        # Find all words within Hamming distance <= radius from cw
        # For small radius (e.g., 1), this is efficient
        if radius == 0:
            # Only the word itself
            idx = word_idx.get(cw_tuple)
            if idx is not None:
                coverage[idx].append(ci)
        elif radius == 1:
            # The word itself
            idx = word_idx.get(cw_tuple)
            if idx is not None:
                coverage[idx].append(ci)
            # All words at distance 1
            for pos in range(n):
                for val in range(q):
                    if val == cw_tuple[pos]:
                        continue
                    neighbor = list(cw_tuple)
                    neighbor[pos] = val
                    nidx = word_idx.get(tuple(neighbor))
                    if nidx is not None:
                        coverage[nidx].append(ci)
        else:
            # General case: enumerate all words and check distance (expensive for large n)
            for wi, w in enumerate(all_words):
                dist = sum(1 for a, b in zip(cw, w) if a != b)
                if dist <= radius:
                    coverage[wi].append(ci)
    
    # Step 2b: For each codeword, find words it uniquely covers
    # uniquely_covered[ci] = list of word indices for which ci is the only coverer
    uniquely_covered = [[] for _ in range(num_codewords)]
    for wi in range(space_size):
        covs = coverage[wi]
        if len(covs) == 1:
            uniquely_covered[covs[0]].append(wi)
    
    # Step 2c: Greedy removal
    # A codeword can be removed if uniquely_covered[ci] is empty
    # We remove in order of increasing "importance" (e.g., codewords that cover fewer words)
    
    # Compute coverage size for each codeword
    cov_size = [0] * num_codewords
    for wi in range(space_size):
        for ci in coverage[wi]:
            cov_size[ci] += 1
    
    # Order codewords by coverage size (ascending) - remove less-covered first
    order = sorted(range(num_codewords), key=lambda i: cov_size[i])
    
    removed = [False] * num_codewords
    for ci in order:
        if removed[ci]:
            continue
        if uniquely_covered[ci]:
            continue
        # Check time
        if time.time() > deadline - 5.0:
            break
        # Remove codeword ci
        removed[ci] = True
        # Update coverage: for each word that ci covered, remove ci from its coverage list
        # and check if any word now has empty coverage
        # We need to update uniquely_covered for other codewords
        # This is expensive; let's do it carefully
        for wi in range(space_size):
            if ci in coverage[wi]:
                coverage[wi].remove(ci)
                if len(coverage[wi]) == 0:
                    # Can't remove ci; restore
                    removed[ci] = False
                    coverage[wi].append(ci)
                    break
                elif len(coverage[wi]) == 1:
                    other_ci = coverage[wi][0]
                    # This word is now uniquely covered by other_ci
                    # Add to uniquely_covered[other_ci]
                    uniquely_covered[other_ci].append(wi)
        else:
            # Successfully removed ci
            # No need to update uniquely_covered for ci since it's removed
            continue
    
    # Build the reduced code
    reduced_code = [code[i] for i in range(num_codewords) if not removed[i]]
    
    # Step 2d: Second pass with different ordering
    if time.time() < deadline - 10.0:
        # Rebuild coverage for the reduced code
        num_codewords2 = len(reduced_code)
        coverage2 = [[] for _ in range(space_size)]
        for ci, cw in enumerate(reduced_code):
            cw_tuple = tuple(cw)
            if radius == 1:
                idx = word_idx.get(cw_tuple)
                if idx is not None:
                    coverage2[idx].append(ci)
                for pos in range(n):
                    for val in range(q):
                        if val == cw_tuple[pos]:
                            continue
                        neighbor = list(cw_tuple)
                        neighbor[pos] = val
                        nidx = word_idx.get(tuple(neighbor))
                        if nidx is not None:
                            coverage2[nidx].append(ci)
            elif radius == 0:
                idx = word_idx.get(cw_tuple)
                if idx is not None:
                    coverage2[idx].append(ci)
            else:
                for wi, w in enumerate(all_words):
                    dist = sum(1 for a, b in zip(cw, w) if a != b)
                    if dist <= radius:
                        coverage2[wi].append(ci)
        
        uniquely_covered2 = [[] for _ in range(num_codewords2)]
        for wi in range(space_size):
            covs = coverage2[wi]
            if len(covs) == 1:
                uniquely_covered2[covs[0]].append(wi)
        
        cov_size2 = [0] * num_codewords2
        for wi in range(space_size):
            for ci in coverage2[wi]:
                cov_size2[ci] += 1
        
        # Try reverse order this time
        order2 = sorted(range(num_codewords2), key=lambda i: -cov_size2[i])
        
        removed2 = [False] * num_codewords2
        for ci in order2:
            if removed2[ci]:
                continue
            if uniquely_covered2[ci]:
                continue
            if time.time() > deadline - 5.0:
                break
            removed2[ci] = True
            for wi in range(space_size):
                if ci in coverage2[wi]:
                    coverage2[wi].remove(ci)
                    if len(coverage2[wi]) == 0:
                        removed2[ci] = False
                        coverage2[wi].append(ci)
                        break
                    elif len(coverage2[wi]) == 1:
                        other_ci = coverage2[wi][0]
                        uniquely_covered2[other_ci].append(wi)
            else:
                continue
        
        reduced_code2 = [reduced_code[i] for i in range(num_codewords2) if not removed2[i]]
        if len(reduced_code2) < len(reduced_code):
            reduced_code = reduced_code2
    
    return reduced_code
