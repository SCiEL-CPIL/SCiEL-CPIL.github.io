"""Improved program: greedy set cover with bitsets and local search for K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math


def _word_to_int(word, q):
    """Convert a list of digits to an integer in base q."""
    val = 0
    for d in word:
        val = val * q + d
    return val


def _int_to_word(w_int, q, n):
    """Convert an integer to a list of n digits in base q."""
    word = []
    for i in range(n - 1, -1, -1):
        word.append(w_int % q)
        w_int //= q
    return word


def _ball_mask(q, n, radius, center_int):
    """Return an int bitmask of all words within Hamming distance radius of center_int."""
    center = []
    for i in range(n):
        center.append((center_int // (q ** (n - 1 - i))) % q)
    
    mask = 1 << center_int
    
    if radius == 1:
        for pos in range(n):
            orig = center[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                word = center.copy()
                word[pos] = new_val
                w_int = 0
                for j in range(n):
                    w_int = w_int * q + word[j]
                mask |= (1 << w_int)
    else:
        for d in range(1, radius + 1):
            for positions in itertools.combinations(range(n), d):
                for new_vals in itertools.product(range(q), repeat=d):
                    if all(new_vals[i] == center[positions[i]] for i in range(d)):
                        continue
                    word = center.copy()
                    for i, pos in enumerate(positions):
                        word[pos] = new_vals[i]
                    w_int = 0
                    for j in range(n):
                        w_int = w_int * q + word[j]
                    mask |= (1 << w_int)
    
    return mask


def _local_search(code, balls, all_mask, total_words, rng, deadline, max_iters=500):
    """Try to improve the code by removing redundant codewords and swapping."""
    if not code:
        return code
    
    code_set = set(code)
    covered = 0
    for c in code:
        covered |= balls[c]
    
    if covered != all_mask:
        return code
    
    current_size = len(code)
    best_code = code[:]
    best_size = current_size
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try removing a random codeword
        idx = rng.randint(0, len(code) - 1)
        candidate = code.pop(idx)
        
        new_covered = 0
        for c in code:
            new_covered |= balls[c]
        
        if new_covered == all_mask:
            # Successfully removed, keep it
            best_code = code[:]
            best_size = len(code)
        else:
            # Put it back
            code.insert(idx, candidate)
            
            # Try a swap: remove this one and add another random one
            if len(code) > 1:
                other_idx = rng.randint(0, total_words - 1)
                if other_idx not in code_set:
                    test_code = code[:]
                    test_code.append(other_idx)
                    test_covered = 0
                    for c in test_code:
                        test_covered |= balls[c]
                    
                    if test_covered == all_mask and len(test_code) < best_size:
                        code = test_code
                        best_code = code[:]
                        best_size = len(code)
    
    return best_code


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin from 120s limit
    
    total_words = q ** n
    
    # Precompute ball masks for all words
    balls = [0] * total_words
    for i in range(total_words):
        balls[i] = _ball_mask(q, n, radius, i)
    
    # All bits set initially
    all_mask = (1 << total_words) - 1
    best_code = None
    best_size = float('inf')
    
    # Multiple restarts with different random orderings
    rng = random.Random(42)
    
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        uncovered = all_mask
        code = []
        
        # Create a shuffled list of candidates
        candidates = list(range(total_words))
        rng.shuffle(candidates)
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_c = -1
            best_count = 0
            
            for c in candidates:
                count = bin(uncovered & balls[c]).count('1')
                if count > best_count:
                    best_count = count
                    best_c = c
                    if best_count == total_words:
                        break
            
            if best_c == -1 or best_count == 0:
                break
            
            code.append(best_c)
            uncovered &= ~balls[best_c]
        
        if uncovered == 0 and len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
        
        # Apply local search to improve
        if uncovered == 0 and len(code) <= best_size + 2:
            improved = _local_search(code, balls, all_mask, total_words, rng, deadline, max_iters=200)
            if len(improved) < best_size:
                best_size = len(improved)
                best_code = improved[:]
        
        # If we found a very good code, we can break early
        if best_size <= 73:
            break
    
    # If we didn't find any valid code, fall back to the tetracode construction
    if best_code is None:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Convert integer codes back to word lists
    result = [_int_to_word(c, q, n) for c in best_code]
    return result
# EVOLVE-BLOCK-END
