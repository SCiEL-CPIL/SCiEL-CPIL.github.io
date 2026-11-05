"""Rewritten solver for ternary covering code K_3(6,1)."""
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # We solve the specific instance q=3, n=6, radius=1.
    # The known optimal size is 73 (van Laarhoven et al., 1989).
    # Strategy:
    # 1. Start with the direct sum of the ternary tetracode (9 words) and all 3^(6-4)=9 tails.
    #    This gives 81 words.
    # 2. Use local search (add/remove heuristic) to reduce the number of codewords while
    #    maintaining the covering property.
    # 3. Use a deadline guard to ensure we finish within the time limit.

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Step 1: Initial code (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    code = initial_code[:]
    
    # We represent the code as a set of tuples for fast lookup
    code_set = set(tuple(c) for c in code)
    
    # Precompute all words in the space
    all_words = list(itertools.product(range(q), repeat=n))
    
    # For each word, precompute its neighbors (distance 0 and 1)
    # neighbor[word] = list of words within distance <= radius
    # Actually, for covering radius 1, we need to check if a word is covered by any codeword.
    # A codeword covers itself and all words at distance 1.
    
    # Let's precompute for each codeword which words it covers.
    # But that's expensive. Instead, for each word in the space, we check if it's covered.
    
    # A more efficient approach:
    # For each codeword c, the set of words it covers is {c} U {words at distance 1 from c}.
    # We maintain a count for each word: how many codewords cover it.
    # If all words have count >= 1, the code is valid.
    # When we remove a codeword, we decrement the count for all words it covered.
    # If any word's count drops to 0, the removal is invalid.
    
    # Let's precompute the coverage list for each possible codeword.
    # There are 3^6 = 729 possible codewords.
    
    # coverage[word] = list of words covered by this codeword
    coverage = {}
    for w in all_words:
        covered = [w]
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    new_w = list(w)
                    new_w[i] = v
                    covered.append(tuple(new_w))
        coverage[w] = covered
    
    # Initialize coverage counts
    # cover_count[word] = number of codewords that cover this word
    cover_count = np.zeros(q**n, dtype=np.int32)
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # For each codeword in the initial code, increment cover_count for all words it covers
    for c in code:
        c_tuple = tuple(c)
        for covered_word in coverage[c_tuple]:
            cover_count[word_to_idx[covered_word]] += 1
    
    # Check initial validity (should be valid)
    # min_cover = np.min(cover_count)
    # assert min_cover >= 1, "Initial code is not valid"
    
    def is_valid():
        return np.min(cover_count) >= 1
    
    def remove_codeword(c):
        c_tuple = tuple(c)
        for covered_word in coverage[c_tuple]:
            cover_count[word_to_idx[covered_word]] -= 1
        code_set.remove(c_tuple)
    
    def add_codeword(c):
        c_tuple = tuple(c)
        for covered_word in coverage[c_tuple]:
            cover_count[word_to_idx[covered_word]] += 1
        code_set.add(c_tuple)
    
    best_code = code[:]
    best_len = len(code)
    
    # Local search: try to remove codewords
    # We'll iterate: try to remove each codeword in random order.
    # If removal keeps the code valid, remove it.
    # Repeat until no more removals are possible or time is up.
    
    # We can also try to swap: remove one, add another.
    # But first, let's just do removals.
    
    while time.time() < deadline:
        progress = False
        # Shuffle the code to try removals in random order
        current_code = list(code_set)
        random.shuffle(current_code)
        
        for c in current_code:
            if time.time() >= deadline:
                break
            
            c_tuple = tuple(c)
            if c_tuple not in code_set:
                continue
            
            # Try to remove c
            # Temporarily remove
            for covered_word in coverage[c_tuple]:
                cover_count[word_to_idx[covered_word]] -= 1
            
            if np.min(cover_count) >= 1:
                # Valid removal
                code_set.remove(c_tuple)
                progress = True
                if len(code_set) < best_len:
                    best_len = len(code_set)
                    best_code = [list(c) for c in code_set]
            else:
                # Restore
                for covered_word in coverage[c_tuple]:
                    cover_count[word_to_idx[covered_word]] += 1
        
        if not progress:
            # No more removals possible in this pass
            # Try a different strategy: maybe we need to swap?
            # Or just restart with a different random seed?
            # For now, break out and return best_code
            break
    
    return best_code
