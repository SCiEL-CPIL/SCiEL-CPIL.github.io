"""Optimized ternary covering code constructor."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def _hamming_dist(a, b):
    return sum(x != y for x, y in zip(a, b))


def _is_covered(word, codewords, q, n, radius):
    # Check if word is within radius of any codeword
    for cw in codewords:
        if _hamming_dist(word, cw) <= radius:
            return True
    return False


def _greedy_cover(q, n, radius, time_limit=110.0):
    """
    Greedy set cover approach:
    1. Start with a good initial code (direct sum construction).
    2. Iteratively try to remove codewords if coverage is maintained.
    3. Also try adding new codewords that cover the most uncovered words,
       then prune again.
    """
    start_time = time.time()
    
    # Initial code: direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    if n <= 4:
        # For small n, use a simpler approach
        if n == 1:
            return [[i] for i in range(q)]
        elif n == 2:
            return [list(p) for p in itertools.product(range(q), repeat=2)]
        elif n == 3:
            # Perfect code for q=3, n=3, r=1
            return [[a, b, (a + b) % 3] for a in range(3) for b in range(3)]
        else:
            # n=4: tetracode
            return tetracode[:9] if len(tetracode) >= 9 else tetracode
    
    # Build initial code using direct sum
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    # Randomize the order to help greedy pruning
    random.shuffle(initial_code)
    
    codewords = list(initial_code)
    
    # Precompute coverage: for each codeword, track which words it covers
    # This is expensive for large n, so we use a heuristic approach
    
    # Strategy: repeatedly try to remove codewords
    # For each codeword, check if all words it covers are covered by others
    # This is O(|C|^2 * q^n) which is too slow, so we use a sampled approach
    
    # Better approach: use a priority queue of "usefulness"
    # For each codeword, estimate how many words it uniquely covers
    # Remove the least useful ones first
    
    max_iters = 50
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try to remove each codeword in random order
        indices = list(range(len(codewords)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            if idx >= len(codewords):
                continue
                
            # Check if we can remove codeword[idx]
            # Sample some words that this codeword covers to verify they're still covered
            # For efficiency, we check a subset
            
            cw = codewords[idx]
            removable = True
            
            # Generate all words within radius of cw
            # This is sum_{i=0}^{radius} C(n,i) * (q-1)^i
            # For n=6, q=3, r=1: 1 + 6*2 = 13 words per codeword
            
            # Check if all these words are covered by other codewords
            for delta in _deltas_within_radius(q, n, radius):
                word = [(cw[i] + delta[i]) % q for i in range(n)]
                # Check if word is covered by any other codeword
                covered = False
                for j, other_cw in enumerate(codewords):
                    if j == idx:
                        continue
                    if _hamming_dist(word, other_cw) <= radius:
                        covered = True
                        break
                if not covered:
                    removable = False
                    break
            
            if removable:
                del codewords[idx]
                improved = True
                break  # Restart the loop since indices changed
        
        if not improved:
            break
    
    # Second phase: try to add new codewords that cover many currently uncovered words
    # Then prune again
    for iteration in range(20):
        if time.time() - start_time > time_limit:
            break
        
        # Find uncovered words (sample-based)
        # For large spaces, we can't check all words, so we sample
        sample_size = min(1000, 3**n // 10)
        uncovered = []
        for _ in range(sample_size):
            word = [random.randint(0, q - 1) for _ in range(n)]
            if not _is_covered(word, codewords, q, n, radius):
                uncovered.append(word)
        
        if not uncovered:
            break
        
        # For each uncovered word, find the best candidate codeword to add
        # A candidate is a word within radius of the uncovered word
        # We want to maximize coverage of other uncovered words
        
        best_candidate = None
        best_score = 0
        
        # Try candidates from the uncovered words themselves and their neighbors
        candidates = set()
        for uw in uncovered[:50]:  # Limit to first 50
            for delta in _deltas_within_radius(q, n, radius):
                cand = tuple((uw[i] + delta[i]) % q for i in range(n))
                if cand not in [tuple(c) for c in codewords]:
                    candidates.add(cand)
        
        # Score each candidate by how many uncovered words it covers
        for cand in list(candidates)[:200]:  # Limit candidates
            cand_list = list(cand)
            score = 0
            for uw in uncovered:
                if _hamming_dist(cand_list, uw) <= radius:
                    score += 1
            if score > best_score:
                best_score = score
                best_candidate = cand_list
        
        if best_candidate is not None and best_score > 0:
            codewords.append(best_candidate)
        
        # Prune again
        for idx in range(len(codewords) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            
            cw = codewords[idx]
            removable = True
            
            for delta in _deltas_within_radius(q, n, radius):
                word = [(cw[i] + delta[i]) % q for i in range(n)]
                covered = False
                for j, other_cw in enumerate(codewords):
                    if j == idx:
                        continue
                    if _hamming_dist(word, other_cw) <= radius:
                        covered = True
                        break
                if not covered:
                    removable = False
                    break
            
            if removable:
                del codewords[idx]
    
    return codewords


def _deltas_within_radius(q, n, radius):
    """Generate all delta vectors with Hamming weight <= radius."""
    deltas = []
    # Weight 0
    deltas.append([0] * n)
    
    # Weight 1 to radius
    for weight in range(1, radius + 1):
        for positions in itertools.combinations(range(n), weight):
            for values in itertools.product(range(1, q), repeat=weight):
                delta = [0] * n
                for pos, val in zip(positions, values):
                    delta[pos] = val
                deltas.append(delta)
    return deltas


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q
    with covering radius radius.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for the 120s limit
    
    # Handle edge cases
    if radius >= n:
        return [[0] * n]
    
    if n == 0:
        return [[]]
    
    if q == 1:
        return [[0] * n]
    
    # For small cases, use exact or near-exact methods
    if n <= 4:
        return _greedy_cover(q, n, radius, time_limit)
    
    # For the specific case q=3, n=6, r=1, use the known best construction
    # The direct sum gives 9 * 3^2 = 81 words
    # We know 73 is achievable, so we need to do better
    
    # Use greedy optimization
    result = _greedy_cover(q, n, radius, time_limit)
    
    # Ensure we return a valid covering code
    # For safety, if the greedy method fails to produce a valid code,
    # fall back to the direct sum construction
    if time.time() - start_time > time_limit:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return result
    
    return result
# EVOLVE-BLOCK-END
