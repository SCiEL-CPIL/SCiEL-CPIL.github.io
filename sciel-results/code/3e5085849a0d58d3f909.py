import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate initial code
    tetracode = []
    for a in range(q):
        for b in range(q):
            tetracode.append([a, b, (a + b) % q, (a + 2 * b) % q])
    
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    initial_code_template = []
    for h in tetracode:
        for tail in tails:
            initial_code_template.append(h + list(tail))
            
    # Pre-compute balls
    size_space = q**n
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
        
    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w
        
    balls = {}
    for i in range(size_space):
        w = int_to_word(i)
        neighbors = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                w[pos] = new_val
                neighbors.append(word_to_int(w))
            w[pos] = original_val
        balls[i] = neighbors
        
    best_code = None
    best_size = size_space # Upper bound
    
    # Multiple restarts
    while time.time() < deadline:
        # Create a shuffled copy
        current_code = list(initial_code_template)
        random.shuffle(current_code)
        
        cov = [0] * size_space
        
        # Initial coverage
        for c in current_code:
            ci = word_to_int(c)
            for nb in balls[ci]:
                cov[nb] += 1
                
        # Greedy deletion
        changed = True
        while changed:
            changed = False
            # Iterate through a snapshot or index carefully
            # We'll iterate through indices 0..len-1
            # If we remove one, we break and restart the inner while loop (which restarts the for loop)
            for i in range(len(current_code)):
                c = current_code[i]
                ci = word_to_int(c)
                redundant = True
                for nb in balls[ci]:
                    if cov[nb] == 1:
                        redundant = False
                        break
                
                if redundant:
                    current_code.pop(i)
                    for nb in balls[ci]:
                        cov[nb] -= 1
                    changed = True
                    break
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
            
        # If we reached the theoretical lower bound or a very low number, we might stop early?
        # Lower bound is 403. Known is 486.
        if best_size <= 486:
            break

    if best_code is None:
        # Fallback to initial
        return initial_code_template
        
    return best_code
