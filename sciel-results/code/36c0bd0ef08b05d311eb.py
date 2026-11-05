"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill

    # Fallback: direct sum of tetracode
    def _fallback():
        if q == 3 and n == 4:
            return [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # General fallback: all-zero + greedy
        return _greedy_construct(q, n, radius, deadline)

    if q == 3 and n == 7 and radius == 1:
        # Try to find a better code via simulated annealing / local search
        # Start from the tetracode-based code (243 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Convert to a set for fast lookup
        code_set = set(tuple(c) for c in initial)
        
        # Try local search: remove codewords that are redundant
        # A codeword is redundant if all words within distance 1 of it are also within distance 1 of another codeword
        # This is expensive, so do it iteratively
        
        def _is_redundant(cw, code_set, q, n, radius):
            """Check if removing cw still covers all words within distance radius of cw."""
            # For each word within distance radius of cw, check if it's within distance radius of some other codeword
            # This is O(3^7 * |code|) which is too slow, so we use a heuristic:
            # Check a random subset of words within distance 1 of cw
            import random
            rng = random.Random(42)
            # Generate all words within distance 1 of cw
            neighbors = []
            for pos in range(n):
                for val in range(q):
                    if val != cw[pos]:
                        w = list(cw)
                        w[pos] = val
                        neighbors.append(tuple(w))
            # Also include cw itself
            neighbors.append(tuple(cw))
            
            # For each neighbor, check if covered by another codeword
            # To speed up, only check a sample
            sample_size = min(len(neighbors), 20)
            rng.shuffle(neighbors)
            for w in neighbors[:sample_size]:
                covered = False
                for other in code_set:
                    if other == tuple(cw):
                        continue
                    # Check Hamming distance
                    dist = sum(1 for i in range(n) if other[i] != w[i])
                    if dist <= radius:
                        covered = True
                        break
                if not covered:
                    return False
            return True

        # Iteratively remove redundant codewords
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            code_list = list(code_set)
            random.shuffle(code_list)
            for cw in code_list:
                if time.time() > deadline - 10.0:
                    break
                if _is_redundant(cw, code_set, q, n, radius):
                    code_set.remove(tuple(cw))
                    improved = True
                    break  # restart scan after modification
        
        result = list(code_set)
        
        # If we haven't improved much, try a different initial code
        if len(result) >= 240 and time.time() < deadline - 30.0:
            # Try greedy construction
            greedy_result = _greedy_construct(q, n, radius, deadline)
            if len(greedy_result) < len(result):
                result = greedy_result
        
        return result
    else:
        return _fallback()


def _greedy_construct(q, n, radius, deadline):
    """Greedy set cover: repeatedly pick the word that covers the most uncovered words."""
    start_time = time.time()
    
    # For q=3, n=7, radius=1, the space is 2187 words
    # Each codeword covers 1 + n*(q-1) = 1 + 7*2 = 15 words
    
    # Represent each word as an integer for fast hashing
    def word_to_int(w):
        result = 0
        for i, v in enumerate(w):
            result = result * q + v
        return result
    
    def int_to_word(x):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = x % q
            x //= q
        return w
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    all_words_int = set(word_to_int(w) for w in all_words)
    
    # Precompute coverage: for each possible codeword, which words does it cover?
    # This is 3^7 * 15 = 32805 entries, manageable
    
    def get_coverage(cw_int):
        """Return set of word ints covered by codeword cw_int."""
        cw = int_to_word(cw_int)
        covered = {cw_int}
        for pos in range(n):
            for val in range(q):
                if val != cw[pos]:
                    w = list(cw)
                    w[pos] = val
                    covered.add(word_to_int(w))
        return covered
    
    # Greedy: maintain uncovered set
    uncovered = set(all_words_int)
    codewords = []
    
    # Precompute coverage for all possible codewords
    # There are 3^7 = 2187 possible codewords
    all_cw_ints = list(all_words_int)
    coverage_cache = {}
    for cw_int in all_cw_ints:
        coverage_cache[cw_int] = get_coverage(cw_int)
    
    while uncovered and time.time() < deadline - 5.0:
        # Find the codeword that covers the most uncovered words
        best_cw = None
        best_count = -1
        for cw_int in all_cw_ints:
            if cw_int in set(tuple(c) for c in codewords) if False else False:
                pass
            # Skip if already selected
            # We need a set of selected codewords
            pass
        
        # Use a set for selected
        selected_set = set(word_to_int(tuple(c)) for c in codewords)
        
        best_cw = None
        best_count = -1
        for cw_int in all_cw_ints:
            if cw_int in selected_set:
                continue
            count = len(coverage_cache[cw_int] & uncovered)
            if count > best_count:
                best_count = count
                best_cw = cw_int
        
        if best_cw is None:
            break
        
        codewords.append(int_to_word(best_cw))
        selected_set.add(best_cw)
        uncovered -= coverage_cache[best_cw]
    
    return codewords
# EVOLVE-BLOCK-END
