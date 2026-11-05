import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 140.0
    
    total_words = 3**n
    neighbors = np.zeros((total_words, 15), dtype=np.int32)
    for i in range(total_words):
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits = digits[::-1]
        neighbors[i, 0] = i
        idx = 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_digits = digits[:]
                new_digits[pos] = new_val
                val = 0
                for d in new_digits:
                    val = val * 3 + d
                neighbors[i, idx] = val
                idx += 1

    def to_int(word):
        val = 0
        for d in word:
            val = val * 3 + d
        return val

    def to_list(c):
        digits = []
        temp = c
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]

    # Initial set
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(to_int(word))
    
    best_size = len(initial_codewords)
    best_codewords = list(initial_codewords)
    
    # Local search function
    def local_search(initial_set, deadline):
        current_codewords = set(initial_set)
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in current_codewords:
            coverage[neighbors[c]] += 1
            
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline:
            improved = False
            if current_codewords:
                c = random.choice(list(current_codewords))
                coverage[neighbors[c]] -= 1
                if np.all(coverage >= 1):
                    current_codewords.remove(c)
                    improved = True
                    no_improve = 0
                else:
                    coverage[neighbors[c]] += 1
                    no_improve += 1
            
            if no_improve > max_no_improve:
                # Swap
                c1 = random.choice(list(current_codewords))
                while True:
                    c2 = random.randint(0, total_words - 1)
                    if c2 not in current_codewords:
                        break
                coverage[neighbors[c1]] -= 1
                coverage[neighbors[c2]] += 1
                if np.all(coverage >= 1):
                    current_codewords.remove(c1)
                    current_codewords.add(c2)
                    no_improve = 0
                else:
                    coverage[neighbors[c1]] += 1
                    coverage[neighbors[c2]] -= 1
                    no_improve += 1
            
            if len(current_codewords) < best_size:
                best_size = len(current_codewords)
                best_codewords = list(current_codewords)
                
            if best_size <= 186:
                break
                
        return best_size, best_codewords

    # Run local search
    size, codewords = local_search(initial_codewords, deadline)
    
    # If not optimal, try another start?
    # A random valid set is hard to generate.
    # We can try to perturb the best found set.
    
    # Convert result
    result = [to_list(c) for c in codewords]
    return result
