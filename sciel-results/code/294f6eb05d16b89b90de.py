import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Start time
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Generate the seed code: Direct sum of tetracode (n=4) and identity (n-4)
    # Tetracode: [a, b, a+b, a+2b] mod 3
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # For n=6, we append 2 coordinates.
    # The seed program does: h + tail where h is tetracode word, tail is product(0..2, repeat=n-4)
    # This creates 9 * 3^2 = 81 words.
    
    if n != 6:
        # Fallback for other n if needed, though task is specific to n=6
        # Use simple product code or just return a large valid code
        # For safety, if n != 6, we can use a simple greedy construction or return seed logic adapted
        # But task is fixed to n=6. We will handle n=6 specifically for speed.
        pass

    # Map words to indices for fast coverage counting
    # Index = sum(w[i] * 3**i)
    base = 3
    total_words = base ** n
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Precompute neighbors for each possible word?
    # Or just compute on the fly. 
    # Since we only check neighbors of codewords, and there are ~80 codewords, 
    # computing neighbors on the fly is cheap.
    
    def get_neighbors(word):
        # Returns list of indices of words within Hamming distance 1
        idx = 0
        powers = [1]
        for i in range(n-1):
            powers.append(powers[-1] * base)
        
        # Calculate index of the word itself
        for i in range(n):
            idx += word[i] * powers[i]
        
        neighbors = [idx]
        for i in range(n):
            for val in range(base):
                if val != word[i]:
                    new_word = word.copy()
                    new_word[i] = val
                    # Calculate index incrementally or from scratch
                    # Incremental: idx + (val - word[i]) * powers[i]
                    neighbors.append(idx + (val - word[i]) * powers[i])
        return neighbors

    # Initial code
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(base), repeat=n-4):
            initial_code.append(h + list(tail))
            
    best_code = None
    best_size = float('inf')
    
    # Multiple restarts with random ordering
    max_restarts = 1000
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        code = [list(w) for w in initial_code]
        random.shuffle(code)
        
        # Initialize cover count
        cover_count[:] = 0
        for w in code:
            for idx in get_neighbors(w):
                cover_count[idx] += 1
                
        # Greedy removal
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle to break symmetry/local minima
            random.shuffle(code)
            
            i = 0
            while i < len(code):
                w = code[i]
                # Check if w is redundant
                is_redundant = True
                for idx in get_neighbors(w):
                    if cover_count[idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove w
                    for idx in get_neighbors(w):
                        cover_count[idx] -= 1
                    code.pop(i)
                    changed = True
                    # Do not increment i, check next word at same index
                else:
                    i += 1
                    
        if len(code) < best_size:
            best_size = len(code)
            best_code = [list(w) for w in code]
            
        # Early exit if we hit the theoretical lower bound or known best
        if best_size <= 73:
            break

    if best_code is None:
        # Fallback to initial code if something went wrong
        return initial_code
        
    return best_code
