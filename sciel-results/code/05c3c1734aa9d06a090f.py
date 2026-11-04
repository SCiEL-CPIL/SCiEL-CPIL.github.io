"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict


def _int_to_base(q, n, x):
    """Convert integer x to a list of n digits in base q."""
    digits = []
    for _ in range(n):
        digits.append(x % q)
        x //= q
    return digits


def _base_to_int(q, n, digits):
    """Convert list of n digits in base q to integer."""
    result = 0
    for i, d in enumerate(reversed(digits)):
        result += d * (q ** i)
    return result


def _hamming_distance(a, b):
    """Compute Hamming distance between two lists."""
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def _ball(q, n, center, radius):
    """Generate all words within Hamming distance radius of center."""
    if radius == 0:
        return [center[:]]
    
    # Generate all positions to change
    positions = range(n)
    if radius == 1:
        result = [center[:]]
        for pos in positions:
            for val in range(q):
                if val != center[pos]:
                    word = center[:]
                    word[pos] = val
                    result.append(word)
        return result
    else:
        # For general radius, use recursive generation
        result = []
        def gen(idx, current, changes_left):
            if changes_left == 0:
                result.append(current[:])
                return
            # Option: don't change this position
            gen(idx + 1, current, changes_left)
            # Option: change this position to a different value
            for val in range(q):
                if val != current[idx]:
                    current[idx] = val
                    gen(idx + 1, current, changes_left - 1)
                    current[idx] = center[idx]
        
        gen(0, center[:], radius)
        return result


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Total number of words in the space
    total_words = q ** n
    
    # Precompute all words as integers
    all_words_int = list(range(total_words))
    
    # For each word, precompute the set of words within distance radius
    # This is expensive for large n, so we'll compute on the fly or cache
    
    # Greedy set cover: repeatedly pick the word that covers the most uncovered words
    # First, initialize: start with a reasonable initial solution
    
    # Start with the tetracode-based construction as a warm start
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # For small n, start with all words
        initial_code = [_int_to_base(q, n, i) for i in range(total_words)]
    
    # Convert to set of integers for faster lookup
    code_set = set(_base_to_int(q, n, cw) for cw in initial_code)
    
    # Compute which words are covered
    covered = [False] * total_words
    
    def update_coverage(add_word_int, remove_word_int=None):
        """Update coverage when adding/removing a codeword."""
        if add_word_int is not None:
            center = _int_to_base(q, n, add_word_int)
            for word in _ball(q, n, center, radius):
                w_int = _base_to_int(q, n, word)
                covered[w_int] = True
        if remove_word_int is not None:
            center = _int_to_base(q, n, remove_word_int)
            for word in _ball(q, n, center, radius):
                w_int = _base_to_int(q, n, word)
                # Only set to False if no other codeword covers it
                # This is expensive; we'll handle it differently
                pass
    
    # Simpler approach: recompute coverage from scratch each time (expensive but correct)
    # Better: maintain a count of how many codewords cover each word
    
    cover_count = [0] * total_words
    
    def recompute_cover_counts():
        """Recompute cover_count from current code_set."""
        for i in range(total_words):
            cover_count[i] = 0
        for cw_int in code_set:
            center = _int_to_base(q, n, cw_int)
            for word in _ball(q, n, center, radius):
                w_int = _base_to_int(q, n, word)
                cover_count[w_int] += 1
    
    def is_fully_covered():
        """Check if all words are covered."""
        for i in range(total_words):
            if cover_count[i] == 0:
                return False
        return True
    
    # Initialize cover counts
    recompute_cover_counts()
    
    best_code = list(code_set)
    best_size = len(code_set)
    
    # Local search: try to remove redundant codewords
    # A codeword is redundant if removing it still leaves all words covered
    
    def try_remove(cw_int):
        """Try to remove cw_int; return True if successful."""
        # Check if any word would become uncovered
        center = _int_to_base(q, n, cw_int)
        for word in _ball(q, n, center, radius):
            w_int = _base_to_int(q, n, word)
            if cover_count[w_int] == 1:
                return False
        # Safe to remove
        for word in _ball(q, n, center, radius):
            w_int = _base_to_int(q, n, word)
            cover_count[w_int] -= 1
        code_set.remove(cw_int)
        return True
    
    # Try to remove redundant codewords iteratively
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cw_int in list(code_set):
            if try_remove(cw_int):
                improved = True
                best_size = len(code_set)
                best_code = list(code_set)
                if time.time() > deadline:
                    break
    
    # Now try greedy addition to cover any uncovered words
    # (shouldn't happen if we started from a valid code, but just in case)
    
    # Try to add codewords to reduce the count via swapping
    # For each codeword in the code, try replacing it with another that covers more uncovered words
    
    # A more effective local search: for each codeword, try all possible replacements
    # and keep the one that maximizes coverage of currently uncovered words
    
    # First, let's do a simple greedy to see if we can do better
    # Start from scratch with greedy
    
    if time.time() < deadline:
        # Greedy from scratch
        greedy_covered = [False] * total_words
        greedy_code = []
        
        # Precompute balls for all words (expensive but we'll do it once)
        # Actually, let's compute on the fly
        
        while time.time() < deadline:
            # Find uncovered words
            uncovered = [i for i in range(total_words) if not greedy_covered[i]]
            if not uncovered:
                break
            
            # Pick the word that covers the most uncovered words
            best_word = -1
            best_count = 0
            
            # Sample a subset of candidate words to speed up
            candidates = list(range(total_words))
            random.shuffle(candidates)
            candidates = candidates[:min(500, total_words)]
            
            for cw_int in candidates:
                center = _int_to_base(q, n, cw_int)
                count = 0
                for word in _ball(q, n, center, radius):
                    w_int = _base_to_int(q, n, word)
                    if not greedy_covered[w_int]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_word = cw_int
            
            if best_word == -1 or best_count == 0:
                break
            
            # Add this word
            greedy_code.append(best_word)
            center = _int_to_base(q, n, best_word)
            for word in _ball(q, n, center, radius):
                w_int = _base_to_int(q, n, word)
                greedy_covered[w_int] = True
        
        # Now do local search on the greedy solution
        greedy_cover_count = [0] * total_words
        for cw_int in greedy_code:
            center = _int_to_base(q, n, cw_int)
            for word in _ball(q, n, center, radius):
                w_int = _base_to_int(q, n, word)
                greedy_cover_count[w_int] += 1
        
        greedy_set = set(greedy_code)
        
        def greedy_try_remove(cw_int):
            center = _int_to_base(q, n, cw_int)
            for word in _ball(q, n, center, radius):
                w_int = _base_to_int(q, n, word)
                if greedy_cover_count[w_int] == 1:
                    return False
            for word in _ball(q, n, center, radius):
                w_int = _base_to_int(q, n, word)
                greedy_cover_count[w_int] -= 1
            greedy_set.remove(cw_int)
            return True
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for cw_int in list(greedy_set):
                if greedy_try_remove(cw_int):
                    improved = True
        
        if len(greedy_set) < best_size:
            best_size = len(greedy_set)
            best_code = list(greedy_set)
    
    # Convert best_code back to list of lists
    result = [_int_to_base(q, n, cw) for cw in best_code]
    
    return result
# EVOLVE-BLOCK-END
