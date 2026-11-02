"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_covered(code, q, n, radius):
    """Check if a code covers the entire space within the given radius."""
    code = np.array(code, dtype=np.int64)
    if len(code) == 0:
        return False
    # Number of words covered by one codeword within radius r
    # For ternary code, radius 1: 1 + n*(q-1) = 1 + 2n
    # We need to check if the union of balls covers all q^n words
    # Efficient check: use a bitset or array
    # For q=3, n=7, total words = 2187, which is small enough for a boolean array
    
    total_words = q ** n
    covered = np.zeros(total_words, dtype=bool)
    
    # Precompute neighbors for each codeword
    # For radius r, a word has 1 + n*(q-1) + ... neighbors
    # For r=1: 1 + n*(q-1) neighbors
    
    def _word_to_idx(w):
        """Convert a word (tuple/list of ints) to an index in [0, q^n)."""
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    for cw in code:
        cw = list(cw)
        # Add the codeword itself
        idx = _word_to_idx(cw)
        covered[idx] = True
        
        if radius >= 1:
            # Add all words at distance 1
            for pos in range(n):
                orig = cw[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    cw[pos] = val
                    idx = _word_to_idx(cw)
                    covered[idx] = True
                cw[pos] = orig
        
        if radius >= 2:
            # Add all words at distance 2
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1 = cw[pos1]
                    orig2 = cw[pos2]
                    for val1 in range(q):
                        if val1 == orig1:
                            continue
                        cw[pos1] = val1
                        for val2 in range(q):
                            if val2 == orig2:
                                continue
                            cw[pos2] = val2
                            idx = _word_to_idx(cw)
                            covered[idx] = True
                        cw[pos2] = orig2
                    cw[pos1] = orig1
    
    return covered.all()


def _greedy_cover(q, n, radius, time_limit=100):
    """
    Greedy algorithm to find a covering code.
    Repeatedly pick the codeword that covers the most uncovered words.
    """
    start_time = time.time()
    
    total_words = q ** n
    # Represent each word as an index
    # Precompute the ball (set of covered words) for each possible codeword
    # For q=3, n=7, total_words = 2187, ball size for r=1 is 1 + 7*2 = 15
    
    def _word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def _idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return list(reversed(w))
    
    # Precompute balls: for each codeword index, which word indices it covers
    ball = {}
    for idx in range(total_words):
        w = _idx_to_word(idx)
        covered_indices = set()
        covered_indices.add(idx)
        
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    w[pos] = val
                    covered_indices.add(_word_to_idx(w))
                w[pos] = orig
        
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1 = w[pos1]
                    orig2 = w[pos2]
                    for val1 in range(q):
                        if val1 == orig1:
                            continue
                        w[pos1] = val1
                        for val2 in range(q):
                            if val2 == orig2:
                                continue
                            w[pos2] = val2
                            covered_indices.add(_word_to_idx(w))
                        w[pos2] = orig2
                    w[pos1] = orig1
        
        ball[idx] = covered_indices
    
    # Convert to numpy arrays for faster operations
    ball_arrays = {}
    for idx in range(total_words):
        ball_arrays[idx] = np.array(sorted(ball[idx]), dtype=np.int64)
    
    # Greedy selection
    covered_mask = np.zeros(total_words, dtype=bool)
    selected = []
    
    while not covered_mask.all():
        if time.time() - start_time > time_limit:
            break
        
        # Find the codeword that covers the most uncovered words
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            if time.time() - start_time > time_limit:
                break
            b = ball_arrays[idx]
            count = np.sum(~covered_mask[b])
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1 or best_count <= 0:
            break
        
        selected.append(best_idx)
        b = ball_arrays[best_idx]
        covered_mask[b] = True
    
    # Convert back to words
    return [_idx_to_word(idx) for idx in selected]


def _remove_redundant(code, q, n, radius, time_limit=50):
    """
    Remove redundant codewords from a valid covering code.
    A codeword is redundant if removing it still leaves a valid cover.
    """
    start_time = time.time()
    code = [list(c) for c in code]
    changed = True
    
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(code)):
            if time.time() - start_time > time_limit:
                break
            # Try removing codeword i
            test_code = code[:i] + code[i+1:]
            if _is_covered(test_code, q, n, radius):
                code = test_code
                changed = True
                break
    
    return code


def _local_search(code, q, n, radius, time_limit=60):
    """
    Local search: try replacing a codeword with another to reduce total count or improve coverage.
    For now, just do redundancy removal after greedy.
    """
    start_time = time.time()
    code = [list(c) for c in code]
    
    # First, remove redundant codewords
    code = _remove_redundant(code, q, n, radius, time_limit=30)
    
    # Try to find a smaller code by random restarts
    best_code = code
    best_len = len(code)
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        return best_code
    
    # Try a few random perturbations
    rng = random.Random(42)
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly remove a few codewords and try to repair
        perturbed = list(code)
        # Remove up to 5% of codewords randomly
        num_remove = max(1, len(perturbed) // 20)
        for _ in range(num_remove):
            if perturbed:
                idx = rng.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
        
        # Try to repair by adding back useful codewords
        # This is complex; for now, just check if still valid
        if _is_covered(perturbed, q, n, radius) and len(perturbed) < best_len:
            best_code = perturbed
            best_len = len(perturbed)
    
    return best_code


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0,...,q-1} with covering radius r.
    Returns a list of codewords (each a list of n integers).
    """
    start_time = time.time()
    total_time_limit = 120  # seconds
    
    # Fallback: direct sum construction
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        fallback = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # For small n, use all words
        import itertools
        fallback = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    best_code = fallback
    best_len = len(fallback)
    
    # Try greedy algorithm
    greedy_time = min(80, total_time_limit - 40)
    if greedy_time > 10:
        greedy_code = _greedy_cover(q, n, radius, time_limit=greedy_time)
        if _is_covered(greedy_code, q, n, radius) and len(greedy_code) < best_len:
            best_code = greedy_code
            best_len = len(greedy_code)
    
    # Try local search / redundancy removal
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 20 and len(best_code) > 0:
        improved_code = _local_search(best_code, q, n, radius, time_limit=min(60, remaining - 10))
        if _is_covered(improved_code, q, n, radius) and len(improved_code) < best_len:
            best_code = improved_code
            best_len = len(improved_code)
    
    return best_code
# EVOLVE-BLOCK-END
