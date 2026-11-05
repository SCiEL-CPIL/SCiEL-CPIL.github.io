import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters if needed, though task specifies 3,6,1
        # For safety, return a simple valid code if params differ, or raise error?
        # Task says "evaluator passes this task's alphabet size...", implying fixed.
        # But robustness is good.
        pass

    # 1. Precompute balls for all 729 words
    balls = []
    for w in range(729):
        # Convert to base 3 digits
        digits = []
        tmp = w
        for _ in range(6):
            digits.append(tmp % 3)
            tmp //= 3
        
        neighbors = [w]
        for i in range(6):
            orig = digits[i]
            # +1 mod 3
            new_digit = (orig + 1) % 3
            new_digits = digits.copy()
            new_digits[i] = new_digit
            new_w = 0
            for j in range(6):
                new_w = new_w * 3 + new_digits[j]
            neighbors.append(new_w)
            
            # -1 mod 3
            new_digit = (orig - 1) % 3
            new_digits[i] = new_digit
            new_w = 0
            for j in range(6):
                new_w = new_w * 3 + new_digits[j]
            neighbors.append(new_w)
        balls.append(neighbors)
    
    # 2. Generate initial code: Tetracode sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    import itertools
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = h + list(tail)
            # Convert to int
            w_int = 0
            for d in word:
                w_int = w_int * 3 + d
            initial_code.append(w_int)
    
    best_code = initial_code[:]
    best_size = len(best_code)
    
    # 3. Pruning Loop
    code = initial_code[:]
    cover = np.zeros(729, dtype=np.int8)
    
    # Initialize coverage
    for c in code:
        for v in balls[c]:
            cover[v] += 1
            
    # Try to prune
    for restart in range(1000): # Max restarts
        if time.time() > deadline:
            break
            
        # Shuffle code to break symmetry
        random.shuffle(code)
        
        removed_any = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
                
            c = code[i]
            # Tentatively remove
            for v in balls[c]:
                cover[v] -= 1
                
            # Check if all covered
            # balls[c] is a list of 13 ints
            if np.min(cover[balls[c]]) > 0:
                # Safe to remove
                code.pop(i)
                removed_any = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
            else:
                # Restore
                for v in balls[c]:
                    cover[v] += 1
                
        if not removed_any:
            # Stuck, try another restart
            continue
            
        if best_size <= 73:
            break

    # Convert best_code back to lists of ints
    result = []
    for w_int in best_code:
        digits = []
        tmp = w_int
        for _ in range(6):
            digits.append(tmp % 3)
            tmp //= 3
        digits.reverse() # Since we built w_int as MSB first in generation? 
        # Wait, in generation: w_int = w_int * 3 + d. So first digit is MSB.
        # In precompute: w_int = sum(d_j * 3^(5-j))? No.
        # Let's check consistency.
        # Generation:
        # word = [d0, d1, d2, d3, d4, d5]
        # w_int = ((d0*3 + d1)*3 + d2)*3 + d3)*3 + d4)*3 + d5
        # This is standard base-3 representation where d0 is the most significant.
        
        # Precompute:
        # tmp = w
        # digits.append(tmp % 3) -> d5 (LSB)
        # ...
        # digits = [d5, d4, d3, d2, d1, d0]
        # Then new_w calculation:
        # new_w = 0
        # for j in range(6): new_w = new_w * 3 + new_digits[j]
        # This treats new_digits[0] as MSB.
        # But new_digits is a copy of `digits` which was [d5, d4, ... d0].
        # So `digits` list in precompute is [LSB, ..., MSB].
        # So `new_w` calculation effectively computes:
        # new_w = d5*3^5 + d4*3^4 + ... + d0*3^0 ?
        # NO.
        # If digits = [d5, d4, d3, d2, d1, d0]
        # Loop j=0: val = d5
        # Loop j=1: val = d5*3 + d4
        # ...
        # Loop j=5: val = d5*3^5 + ... + d0
        # So the integer representation in `balls` corresponds to d5 being the most significant power (3^5).
        
        # In Generation:
        # w_int = d0*3^5 + d1*3^4 + ... + d5*3^0.
        
        # MISMATCH!
        # We must ensure the integer mapping is consistent.
        
        # Let's fix the mapping to be consistent.
        # Let's define `int_to_word` and `word_to_int` clearly.
        
        pass # Placeholder, need to rewrite carefully.

    # Let's restart the logic with consistent mapping.
    
    def word_to_int(word):
        w = 0
        for d in word:
            w = w * 3 + d
        return w

    def int_to_word(w):
        digits = []
        for _ in range(6):
            digits.append(w % 3)
            w //= 3
        digits.reverse()
        return digits

    # Precompute balls with consistent mapping
    balls = []
    for w in range(729):
        word = int_to_word(w)
        neighbors = [w]
        for i in range(6):
            orig = word[i]
            for delta in [1, -1]:
                new_digit = (orig + delta) % 3
                new_word = word[:]
                new_word[i] = new_digit
                neighbors.append(word_to_int(new_word))
        balls.append(neighbors)

    # Regenerate initial code with consistent mapping
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = h + list(tail)
            initial_code.append(word_to_int(word))
            
    # Re-run pruning with correct balls
    best_code = initial_code[:]
    best_size = len(best_code)
    
    code = initial_code[:]
    cover = np.zeros(729, dtype=np.int8)
    for c in code:
        for v in balls[c]:
            cover[v] += 1
            
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        random.shuffle(code)
        removed_any = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
            c = code[i]
            for v in balls[c]:
                cover[v] -= 1
            if np.min(cover[balls[c]]) > 0:
                code.pop(i)
                removed_any = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
            else:
                for v in balls[c]:
                    cover[v] += 1
                
        if not removed_any:
            continue
        if best_size <= 73:
            break

    # Convert best_code to list of lists
    result = [int_to_word(w) for w in best_code]
    return result
