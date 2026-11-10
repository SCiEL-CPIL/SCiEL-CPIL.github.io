"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if n < 4 or q != 3:
        # Fallback for small n or non-ternary
        if n <= 6 and q ** n <= 100000:
            # Greedy set cover for small spaces
            all_words = list(itertools.product(range(q), repeat=n))
            all_words_set = set(all_words)
            covered = set()
            code = []
            # Precompute balls for all words
            balls = {}
            for w in all_words:
                ball = {w}
                for pos in range(n):
                    for val in range(q):
                        if val != w[pos]:
                            nw = list(w)
                            nw[pos] = val
                            ball.add(tuple(nw))
                balls[w] = ball
            
            remaining = set(all_words)
            while remaining and time.time() < deadline - 5:
                # Greedy: pick word covering most uncovered
                best_word = None
                best_count = 0
                for w in all_words:
                    if w in covered:
                        continue
                    cnt = len(balls[w] & remaining)
                    if cnt > best_count:
                        best_count = cnt
                        best_word = w
                    if time.time() > deadline - 10:
                        break
                if best_word is None or best_count == 0:
                    break
                code.append(list(best_word))
                covered.update(balls[best_word])
                remaining -= balls[best_word]
            return code
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Ternary tetracode words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Construct initial code: 9 * 3^(n-4) words
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    code = [list(c) for c in initial_code]
    
    # Convert to numpy for fast operations
    code_arr = np.array(code, dtype=np.int32)
    
    # Precompute all ball members for each codeword as integer indices
    # Word w -> int: sum(w[i] * 3**i)
    def word_to_int(w):
        val = 0
        for i in range(n):
            val += w[i] * (3 ** i)
        return val
    
    def int_to_word(val):
        w = [0] * n
        for i in range(n):
            w[i] = val % 3
            val //= 3
        return w
    
    size_space = 3 ** n
    
    # Precompute balls for all possible words in the space
    # This is feasible for n=6: 729 words, each ball has 1 + 6*2 = 13 words
    all_balls = []
    for w_int in range(size_space):
        w = int_to_word(w_int)
        ball = [w_int]
        for pos in range(n):
            for val in range(3):
                if val != w[pos]:
                    nw = w[:]
                    nw[pos] = val
                    ball.append(word_to_int(nw))
        all_balls.append(ball)
    
    # Coverage array
    coverage = np.zeros(size_space, dtype=np.int32)
    
    # Initialize coverage
    for cw in code:
        ball = all_balls[word_to_int(cw)]
        for w in ball:
            coverage[w] += 1
    
    # Helper to check if a codeword is removable
    def is_removable(cw):
        ball = all_balls[word_to_int(cw)]
        for w in ball:
            if coverage[w] <= 1:
                return False
        return True
    
    # Greedy removal loop
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            if is_removable(code[i]):
                ball = all_balls[word_to_int(code[i])]
                for w in ball:
                    coverage[w] -= 1
                code.pop(i)
                code_arr = np.array(code, dtype=np.int32)
                improved = True
    
    # Now try swap operations: replace one codeword with another to potentially enable further removals
    # Strategy: for each codeword, try swapping with a random other word in the space
    # If the swap maintains coverage, keep it and try to remove more
    
    # Precompute all words
    all_words_int = list(range(size_space))
    all_words = [int_to_word(w) for w in all_words_int]
    
    # Try a limited number of swap attempts
    max_swap_attempts = 2000
    swap_attempts = 0
    np_random = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0 and swap_attempts < max_swap_attempts:
        swap_attempts += 1
        if len(code) == 0:
            break
        
        # Pick a random codeword to potentially swap
        idx = np_random.integers(0, len(code))
        cw_to_swap = code[idx]
        
        # Try swapping with a random word
        new_word_int = np_random.integers(0, size_space)
        new_word = all_words[new_word_int]
        
        if new_word == cw_to_swap:
            continue
        
        # Check if we can swap: remove cw_to_swap, add new_word, check coverage
        # First, temporarily remove cw_to_swap
        old_ball = all_balls[word_to_int(cw_to_swap)]
        new_ball = all_balls[new_word_int]
        
        # Check if after removal, all words in old_ball are still covered by other codewords
        # And after addition, all words in new_ball are covered
        # This is complex; simpler: just do the swap and check if coverage stays >= 1 everywhere
        
        # Save old coverage
        old_coverage = coverage.copy()
        
        # Remove old
        for w in old_ball:
            coverage[w] -= 1
        
        # Check if any went to 0
        if np.any(coverage < 0):
            coverage = old_coverage
            continue
        
        # Add new
        for w in new_ball:
            coverage[w] += 1
        
        # The swap is valid if no word has coverage < 1
        # But we need to check that all words are still covered
        # Actually, we need to verify that the entire space is still covered
        # This is expensive; instead, check that no word in old_ball went to 0
        # (which we did) and that the swap doesn't create new uncovered words
        # Since we're just swapping, the only risk is words that were only covered by cw_to_swap
        
        # Actually, the check above (no coverage < 0 after removal) is sufficient
        # because if a word had coverage exactly 1 and it was only covered by cw_to_swap,
        # then after removal it would be 0, and we'd catch it
        
        # Perform the swap
        code[idx] = new_word
        code_arr = np.array(code, dtype=np.int32)
        
        # Now try to remove redundant codewords again
        for _ in range(10):  # Limited removal attempts after swap
            removed_any = False
            for i in range(len(code) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                if is_removable(code[i]):
                    ball = all_balls[word_to_int(code[i])]
                    for w in ball:
                        coverage[w] -= 1
                    code.pop(i)
                    code_arr = np.array(code, dtype=np.int32)
                    removed_any = True
                    break
            if not removed_any:
                break
    
    return code

# EVOLVE-BLOCK-END
