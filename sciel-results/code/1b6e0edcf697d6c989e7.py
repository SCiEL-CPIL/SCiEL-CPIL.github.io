"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

# Deadline guard: evaluator kills at 150s; we use 110s as our internal deadline to be safe.
DEADLINE = 110.0


def _all_words(q, n):
    """Generate all q^n words as tuples of ints in 0..q-1."""
    return list(itertools.product(range(q), repeat=n))


def _hamming_distance(a, b):
    """Compute Hamming distance between two tuples/lists of same length."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _neighbors_within_radius(word, q, n, radius):
    """
    Generate all words within Hamming distance <= radius from the given word.
    Returns a set of tuples.
    """
    neighbors = set()
    neighbors.add(tuple(word))
    if radius == 0:
        return neighbors
    
    # For radius 1: word itself + all single-position changes
    if radius == 1:
        for i in range(n):
            orig = word[i]
            for val in range(q):
                if val == orig:
                    continue
                lst = list(word)
                lst[i] = val
                neighbors.add(tuple(lst))
        return neighbors
    
    # General radius: use recursive generation
    # For small n and radius, this is fine
    from itertools import combinations_with_replacement, product
    
    # Generate all positions to change (up to 'radius' positions)
    # For each subset of positions of size <= radius, and each assignment of non-original values
    # This can explode, but for n=7, radius=1 it's fine. For larger radii we need care.
    
    # Actually for the task at hand (n=7, radius=1), the radius-1 case above suffices.
    # Let's handle general case more carefully for robustness.
    
    # For radius >= 2, we need a more general approach.
    # But given the task is specifically n=7, radius=1, let's optimize for that.
    # Still, let's make it general enough.
    
    # Use a BFS-like approach or recursive
    def _gen(pos, changed, current):
        if pos == n:
            if changed <= radius:
                neighbors.add(tuple(current))
            return
        # Don't change position pos
        _gen(pos + 1, changed, current)
        # Change position pos
        if changed < radius:
            orig = current[pos]
            for val in range(q):
                if val != orig:
                    current[pos] = val
                    _gen(pos + 1, changed + 1, current)
                    current[pos] = orig
    
    _gen(0, 0, list(word))
    return neighbors


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius 'radius'.
    Uses a greedy set-cover approach with random restarts.
    """
    start_time = time.time()
    
    # For the specific task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Each codeword covers 1 + n*(q-1) = 1 + 7*2 = 15 words for radius=1, q=3
    
    # Precompute all words
    all_words = _all_words(q, n)
    num_words = len(all_words)
    
    # For efficiency with n=7, q=3, radius=1, we can represent words as integers 0..2186
    # and precompute coverage
    
    if q == 3 and n == 7 and radius == 1:
        return _construct_ternary_7_1(start_time)
    
    # General case: greedy set cover with random restarts
    best_code = None
    best_size = num_words  # worst case: all words
    
    # Precompute coverage sets for each word
    # coverage[word_index] = set of word indices covered by choosing word_index as codeword
    word_to_index = {w: i for i, w in enumerate(all_words)}
    coverage = []
    for w in all_words:
        cov_set = set()
        for neighbor in _neighbors_within_radius(w, q, n, radius):
            if neighbor in word_to_index:
                cov_set.add(word_to_index[neighbor])
        coverage.append(cov_set)
    
    # Greedy set cover with multiple random restarts
    for restart in range(50):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        random.seed(restart * 12345 + 7)
        
        # Randomly shuffle the order of candidate words
        candidates = list(range(num_words))
        random.shuffle(candidates)
        
        # Greedy selection
        covered = set()
        code = []
        
        for cand_idx in candidates:
            if len(covered) == num_words:
                break
            
            # Compute how many new words this candidate covers
            new_cover = coverage[cand_idx] - covered
            if not new_cover:
                continue
            
            # Greedy: pick the one that covers the most new words
            # But we're iterating in random order; let's do proper greedy
            break
        
        # Proper greedy: at each step, pick the candidate that covers the most uncovered words
        covered = set()
        code = []
        remaining_candidates = set(range(num_words))
        
        while len(covered) < num_words:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 5:
                break
            
            best_cand = -1
            best_new_count = 0
            
            # Sample a subset of candidates for speed if there are many
            if len(remaining_candidates) > 200:
                sampled = random.sample(list(remaining_candidates), 200)
            else:
                sampled = list(remaining_candidates)
            
            for cand in sampled:
                new_count = len(coverage[cand] - covered)
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_cand = cand
            
            if best_cand == -1 or best_new_count == 0:
                break
            
            covered.update(coverage[best_cand])
            code.append(all_words[best_cand])
            remaining_candidates.discard(best_cand)
        
        if len(covered) == num_words and len(code) < best_size:
            best_size = len(code)
            best_code = [list(w) for w in code]
    
    if best_code is None:
        # Fallback: return all words (valid but large)
        best_code = [list(w) for w in all_words]
    
    return best_code


def _construct_ternary_7_1(start_time):
    """
    Specialized fast path for q=3, n=7, radius=1.
    Uses bitset-based greedy set cover with random restarts.
    """
    q = 3
    n = 7
    radius = 1
    num_words = q ** n  # 2187
    
    # Represent each word as an integer 0..2186 (base-3 representation)
    # word_to_int: tuple -> int
    word_to_int = {}
    int_to_word = {}
    idx = 0
    for w in itertools.product(range(q), repeat=n):
        val = 0
        for digit in w:
            val = val * q + digit
        word_to_int[w] = val
        int_to_word[val] = w
        idx += 1
    
    # Precompute coverage bitsets
    # coverage[i] is a Python int whose bit j is set if word j is within distance 1 of word i
    # We use Python ints as bitsets (arbitrary precision)
    coverage = [0] * num_words
    
    for i in range(num_words):
        word = int_to_word[i]
        bitset = 1 << i  # self
        # All neighbors at distance 1
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_word = list(word)
                new_word[pos] = val
                j = word_to_int[tuple(new_word)]
                bitset |= (1 << j)
        coverage[i] = bitset
    
    # Greedy set cover with random restarts
    best_code = None
    best_size = num_words
    
    all_indices = list(range(num_words))
    
    for restart in range(200):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        random.seed(restart * 98765 + 42)
        
        # Random perturbation: shuffle and use a slightly randomized greedy
        # At each step, among the top-k candidates by coverage, pick one randomly
        covered_bits = 0
        code_indices = []
        remaining = set(all_indices)
        
        # To speed up, we'll track uncovered count
        uncovered_count = num_words
        
        while uncovered_count > 0:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 5:
                break
            
            # Find candidates that cover the most uncovered words
            # For speed, sample if too many
            if len(remaining) > 300:
                sampled = random.sample(list(remaining), 300)
            else:
                sampled = list(remaining)
            
            best_new = 0
            top_candidates = []
            
            for cand in sampled:
                # Count uncovered bits in coverage[cand]
                new_bits = coverage[cand] & (~covered_bits)
                # Count set bits
                new_count = bin(new_bits).count('1')
                if new_count > best_new:
                    best_new = new_count
                    top_candidates = [cand]
                elif new_count == best_new and new_count > 0:
                    top_candidates.append(cand)
            
            if not top_candidates:
                break
            
            # Pick randomly from top candidates for diversity
            chosen = random.choice(top_candidates)
            covered_bits |= coverage[chosen]
            code_indices.append(chosen)
            remaining.discard(chosen)
            
            # Update uncovered count
            uncovered_count = num_words - bin(covered_bits).count('1')
        
        if uncovered_count == 0 and len(code_indices) < best_size:
            best_size = len(code_indices)
            best_code = [list(int_to_word[i]) for i in code_indices]
    
    if best_code is None:
        # Fallback: use the direct sum construction from the seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return best_code
# EVOLVE-BLOCK-END
