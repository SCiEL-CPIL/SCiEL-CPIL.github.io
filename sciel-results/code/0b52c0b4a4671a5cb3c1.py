import itertools
import random
import time

def construct(q, n, radius):
    # Assume q=3, n=6, radius=1 based on task description
    # But handle generally if possible.
    
    # Specific optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1()
    
    # Fallback generic greedy
    return _construct_generic(q, n, radius)

def _construct_ternary_6_1():
    # Generate initial code: Direct sum of ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(h + list(tail))
            
    def to_int(word):
        val = 0
        for digit in word:
            val = val * 3 + digit
        return val

    def to_word(val):
        word = []
        for _ in range(6):
            word.append(val % 3)
            val //= 3
        return word[::-1]

    code_set = set(to_int(w) for w in initial_code)
    
    all_words = list(range(729))
    
    # Precompute neighbors
    neighbors = {}
    for w in all_words:
        word = to_word(w)
        nbs = [w]
        for i in range(6):
            for new_val in range(3):
                if new_val == word[i]:
                    continue
                new_word = word.copy()
                new_word[i] = new_val
                nbs.append(to_int(new_word))
        neighbors[w] = nbs

    coverage = [0] * 729
    
    def add_codeword(c):
        for nb in neighbors[c]:
            coverage[nb] += 1
            
    def remove_codeword(c):
        for nb in neighbors[c]:
            coverage[nb] -= 1

    for c in code_set:
        add_codeword(c)

    current_code = list(code_set)
    
    start_time = time.time()
    deadline = 148.0
    
    best_size = len(current_code)
    best_code = set(current_code)
    
    # Simulated annealing parameters
    initial_temp = 1.0
    cooling_rate = 0.9995
    min_temp = 0.01
    max_iterations = 100000
    
    def try_remove(c):
        """Try to remove codeword c. Returns True if successful."""
        remove_codeword(c)
        if min(coverage) > 0:
            current_code.remove(c)
            return True
        else:
            add_codeword(c)
            return False
    
    def try_swap(c_old, c_new):
        """Try to swap c_old with c_new. Returns True if successful."""
        if c_new in current_code:
            return False
        remove_codeword(c_old)
        add_codeword(c_new)
        if min(coverage) > 0:
            current_code.remove(c_old)
            current_code.append(c_new)
            return True
        else:
            remove_codeword(c_new)
            add_codeword(c_old)
            return False
    
    temp = initial_temp
    iteration = 0
    
    while time.time() < deadline and temp > min_temp:
        iteration += 1
        
        # Phase 1: Try to remove redundant codewords
        removed_any = False
        # Shuffle for randomness
        code_list = list(current_code)
        random.shuffle(code_list)
        
        for c in code_list:
            if time.time() > deadline:
                break
            if c not in current_code:
                continue
            if try_remove(c):
                removed_any = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
        
        # Phase 2: Try random swaps (simulated annealing)
        if not removed_any and current_code:
            # Try multiple swaps per iteration
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                c_old = random.choice(current_code)
                c_new = random.randint(0, 728)
                
                if c_new in current_code:
                    continue
                
                # Calculate coverage change for this swap
                # Remove c_old, add c_new
                remove_codeword(c_old)
                add_codeword(c_new)
                
                # Check if valid
                if min(coverage) > 0:
                    # Valid swap, always accept
                    current_code.remove(c_old)
                    current_code.append(c_new)
                    # Don't update best_code unless size decreased (it didn't)
                else:
                    # Invalid swap, reject
                    remove_codeword(c_new)
                    add_codeword(c_old)
        
        # Cool down
        temp *= cooling_rate
        
        # Periodic restart with small perturbation if stuck
        if iteration % 500 == 0 and len(current_code) > best_size:
            # Try to recover by removing more aggressively
            code_list = list(current_code)
            random.shuffle(code_list)
            for c in code_list:
                if time.time() > deadline:
                    break
                if c not in current_code:
                    continue
                if try_remove(c):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = set(current_code)

    # Final cleanup: try to remove as many as possible
    code_list = list(current_code)
    random.shuffle(code_list)
    for c in code_list:
        if time.time() > deadline:
            break
        if c not in current_code:
            continue
        if try_remove(c):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = set(current_code)

    # Use best found
    if len(best_code) < len(current_code):
        current_code = list(best_code)
        # Rebuild coverage
        coverage = [0] * 729
        for c in current_code:
            add_codeword(c)

    return [to_word(c) for c in current_code]

def _construct_generic(q, n, radius):
    # Simple greedy for fallback
    # This is not optimized for the specific task but ensures correctness for other params
    total = q ** n
    words = [list(map(int, itertools.product(range(q), repeat=n))[i]) for i in range(total)]
    
    def dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    covered = [False] * total
    code = []
    
    # Precompute coverage sets for all words? Too big if n is large.
    # For small n, it's fine.
    # But this is a fallback, so keep it simple.
    
    remaining = set(range(total))
    
    while remaining:
        # Find word that covers most remaining
        best_w = None
        best_count = -1
        for w_idx in remaining:
            w = words[w_idx]
            count = 0
            for r_idx in remaining:
                if dist(w, words[r_idx]) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_w = w_idx
                
        code.append(words[best_w])
        # Update remaining
        w = words[best_w]
        new_remaining = set()
        for r_idx in remaining:
            if dist(w, words[r_idx]) > radius:
                new_remaining.add(r_idx)
        remaining = new_remaining
        
    return code
