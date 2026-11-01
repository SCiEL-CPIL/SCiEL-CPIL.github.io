"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius r.
    Uses a greedy set-cover approach with random restarts and local improvement.
    """
    if radius < 0:
        radius = 0
    if radius >= n:
        # Any single word covers the entire space
        return [[0] * n]
    
    # Special case: if q=3, n=6, radius=1, we can use a known good construction
    # But let's use a general algorithm that should work well.
    
    # For small cases, we can try to find a good solution
    # The greedy approach: repeatedly pick the word that covers the most uncovered words
    
    # To speed up, we use a bitset representation
    # Total number of words: q**n
    # For q=3, n=6: 729 words, which is manageable
    
    total_words = q ** n
    
    if total_words > 100000:
        # For larger spaces, use a simpler heuristic
        # Direct product construction as fallback
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Simple greedy
            return _greedy_cover(q, n, radius, max_iterations=1000)
    
    # Precompute neighborhoods for all words
    # neighborhood[w] = set of words within Hamming distance radius from w
    # For efficiency, we represent sets as bitsets (Python integers)
    
    # Generate all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_index = {w: i for i, w in enumerate(all_words)}
    
    # Precompute the index of each word
    # For each word, compute which words it covers
    # This is O(total_words * q * n * ...) which might be slow for large spaces
    # But for q=3, n=6, total_words=729, it's fine
    
    # Compute coverage bitset for each word
    # coverage[i] is a bitmask where bit j is set if word j is within distance radius of word i
    coverage = [0] * total_words
    
    # Helper: compute Hamming distance
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
    
    # Precompute coverage
    for i, w in enumerate(all_words):
        mask = 0
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                mask |= (1 << j)
        coverage[i] = mask
    
    # Greedy set cover with multiple random restarts
    best_codewords = None
    best_size = total_words + 1
    
    num_restarts = 20
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        uncovered = (1 << total_words) - 1
        codewords = []
        
        while uncovered:
            # Find the word that covers the most uncovered words
            # To add randomness, we can slightly perturb the choice
            best_gain = -1
            best_idx = -1
            candidates = []
            
            for i in range(total_words):
                gain = bin(coverage[i] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    candidates = [(gain, i)]
                elif gain == best_gain:
                    candidates.append((gain, i))
            
            if best_gain <= 0:
                break
            
            # Randomly pick among top candidates
            if len(candidates) > 1 and rng.random() < 0.3:
                best_idx = rng.choice(candidates)[1]
            
            codewords.append(all_words[best_idx])
            uncovered &= ~coverage[best_idx]
        
        # Local improvement: try to replace each codeword with a better one
        improved = True
        while improved:
            improved = False
            for idx in range(len(codewords)):
                # Try removing this codeword
                remaining = codewords[:idx] + codewords[idx+1:]
                # Compute coverage of remaining
                rem_mask = 0
                for cw in remaining:
                    rem_mask |= coverage[word_to_index[cw]]
                
                if rem_mask == ((1 << total_words) - 1):
                    # Can remove this codeword
                    codewords = remaining
                    improved = True
                    break
                else:
                    # Try to find a replacement that covers more
                    # Find words not covered by remaining
                    not_covered = ((1 << total_words) - 1) & ~rem_mask
                    if not_covered:
                        best_gain = bin(coverage[word_to_index[codewords[idx]]] & not_covered).count('1')
                        best_replace = None
                        for i in range(total_words):
                            if codewords[idx] == all_words[i]:
                                continue
                            gain = bin(coverage[i] & not_covered).count('1')
                            if gain > best_gain:
                                best_gain = gain
                                best_replace = i
                        
                        if best_replace is not None:
                            codewords[idx] = all_words[best_replace]
                            improved = True
                            break
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
        
        # Early stopping if we reach a good solution
        if best_size <= 73 and q == 3 and n == 6 and radius == 1:
            break
    
    if best_codewords is None:
        # Fallback
        return [[0] * n]
    
    # Deduplicate
    seen = set()
    unique_codewords = []
    for cw in best_codewords:
        if cw not in seen:
            seen.add(cw)
            unique_codewords.append(cw)
    
    return unique_codewords


def _greedy_cover(q, n, radius, max_iterations=1000):
    """Simple greedy cover for larger spaces."""
    codewords = []
    covered = set()
    
    all_words = list(itertools.product(range(q), repeat=n))
    rng = random.Random(42)
    
    for _ in range(max_iterations):
        # Pick a random word and check how many new words it covers
        best_word = None
        best_new = 0
        for w in all_words:
            if w in covered:
                continue
            # Count how many words within distance radius are not yet covered
            new_count = 0
            for v in all_words:
                if v in covered:
                    continue
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    new_count += 1
            if new_count > best_new:
                best_new = new_count
                best_word = w
        
        if best_word is None or best_new == 0:
            break
        
        codewords.append(list(best_word))
        for v in all_words:
            dist = sum(1 for a, b in zip(best_word, v) if a != b)
            if dist <= radius:
                covered.add(v)
        
        if len(covered) == len(all_words):
            break
    
    return codewords
# EVOLVE-BLOCK-END
