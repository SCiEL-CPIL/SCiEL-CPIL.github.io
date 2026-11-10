"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    if len(code) <= 73:
        return code
    
    # Build coverage structure using numpy
    # Convert code to numpy array
    code_arr = np.array(code, dtype=np.int8)
    m = len(code_arr)
    
    # Precompute ball members for each codeword (all words within Hamming distance radius)
    # For radius=1, q=3, n=6: each ball has 1 + 6*2 = 13 words
    # Total space: 3^6 = 729 words
    
    # Encode words as integers for fast lookup
    def encode(word):
        """Encode a word as an integer in base-q"""
        val = 0
        for sym in word:
            val = val * q + sym
        return val
    
    def decode(val, length):
        """Decode an integer to a word"""
        word = [0] * length
        for i in range(length - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word
    
    # Precompute all ball members as encoded integers
    # For each codeword, generate all words within Hamming distance radius
    def get_ball_members(word, q, n, radius):
        """Generate all encoded words within Hamming distance radius of word"""
        members = [encode(word)]
        if radius >= 1:
            for pos in range(n):
                for sym in range(q):
                    if sym != word[pos]:
                        new_word = list(word)
                        new_word[pos] = sym
                        members.append(encode(new_word))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for sym1 in range(q):
                        if sym1 == word[pos1]:
                            continue
                        for sym2 in range(q):
                            if sym2 == word[pos2]:
                                continue
                            new_word = list(word)
                            new_word[pos1] = sym1
                            new_word[pos2] = sym2
                            members.append(encode(new_word))
        return members
    
    # Build coverage count array
    space_size = q ** n
    coverage = np.zeros(space_size, dtype=np.int32)
    
    # For each codeword, add 1 to all words in its ball
    balls = []
    for i in range(m):
        ball = get_ball_members(code[i], q, n, radius)
        balls.append(ball)
        for enc in ball:
            coverage[enc] += 1
    
    # Greedy removal: try to remove codewords that don't make any word uncovered
    # A codeword can be removed if all words in its ball are covered by at least 2 codewords
    removed = set()
    order = list(range(m))
    
    # Sort by ball size (smaller balls first - easier to remove)
    # Actually, for removal we want codewords whose unique coverage is minimal
    # Let's try multiple passes
    
    max_passes = 50
    for p in range(max_passes):
        if time.time() > deadline:
            break
        
        any_removed = False
        # Try removing in order
        for idx in order:
            if time.time() > deadline:
                break
            if idx in removed:
                continue
            
            # Check if all words in this codeword's ball are covered by >= 2
            ball = balls[idx]
            removable = True
            for enc in ball:
                if coverage[enc] <= 1:
                    removable = False
                    break
            
            if removable:
                # Remove this codeword
                removed.add(idx)
                any_removed = True
                for enc in ball:
                    coverage[enc] -= 1
    
        if not any_removed:
            break
    
    # Build result
    result = [code[i] for i in range(m) if i not in removed]
    
    # If we still have too many, try a more aggressive approach
    # Random restart with different removal orders
    if len(result) > 73 and time.time() < deadline:
        best_result = result
        best_len = len(result)
        
        # Try random permutations for removal
        rng = np.random.default_rng(42)
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Reset coverage
            coverage = np.zeros(space_size, dtype=np.int32)
            for i in range(m):
                for enc in balls[i]:
                    coverage[enc] += 1
            
            removed = set()
            # Random order
            order = list(range(m))
            rng.shuffle(order)
            
            for _ in range(max_passes):
                if time.time() > deadline:
                    break
                any_removed = False
                for idx in order:
                    if time.time() > deadline:
                        break
                    if idx in removed:
                        continue
                    
                    ball = balls[idx]
                    removable = True
                    for enc in ball:
                        if coverage[enc] <= 1:
                            removable = False
                            break
                    
                    if removable:
                        removed.add(idx)
                        any_removed = True
                        for enc in ball:
                            coverage[enc] -= 1
                
                if not any_removed:
                    break
            
            current_result = [code[i] for i in range(m) if i not in removed]
            if len(current_result) < best_len:
                best_len = len(current_result)
                best_result = current_result
        
        result = best_result
    
    return result
# EVOLVE-BLOCK-END
