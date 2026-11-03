import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Precompute ball neighbors for each word
    # ball[w] = list of words within Hamming distance 1 of w
    balls = []
    for w in range(total_words):
        w_list = []
        tmp = w
        for i in range(n):
            w_list.insert(0, tmp % q)
            tmp //= q
        
        neighbors = [w]
        for i in range(n):
            for d in range(q):
                if d != w_list[i]:
                    new_word = w_list.copy()
                    new_word[i] = d
                    val = 0
                    for digit in new_word:
                        val = val * q + digit
                    neighbors.append(val)
        balls.append(neighbors)
    
    # Precompute ball as numpy array for fast indexing
    balls_np = [np.array(b, dtype=np.int64) for b in balls]
    
    # Build covers matrix: covers[w, c] = 1 if c covers w (i.e., w is in ball[c])
    # For memory efficiency, we'll use a sparse-like approach
    # Actually, let's precompute for each codeword c, the set of words it covers
    # covers_by_codeword[c] = set/array of words covered by c
    
    # For the tetracode seed, we'll have at most ~81 codewords initially
    # Let's precompute the ball for each possible word index
    
    # Generate initial seed: Direct sum of tetracode
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            val = 0
            for digit in word:
                val = val * q + digit
            initial_codewords.append(val)
    
    code = set(initial_codewords)
    
    # Fast coverage computation using precomputed balls
    def compute_coverage(code_set):
        """Compute coverage counts for all words."""
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            coverage[balls_np[c]] += 1
        return coverage
    
    # Precompute for each word w, which codewords cover it (for fast essential detection)
    # This is the transpose of the covers matrix
    # covers_word[w] = list of codeword indices c such that w in ball[c]
    
    def get_essential_and_removable(coverage, code_set):
        """Find essential codewords (uniquely cover some word) and removable ones."""
        # Find words with coverage == 1
        unique_mask = (coverage == 1)
        unique_words = np.where(unique_mask)[0]
        
        if len(unique_words) == 0:
            # No uniquely covered words, any codeword can be removed
            return set(), code_set.copy()
        
        # For each uniquely covered word, find which codeword covers it
        essential = set()
        for w in unique_words:
            w_int = int(w)
            # Find which codeword in code covers w
            for c in code_set:
                if w_int in balls_np[c]:
                    essential.add(c)
                    break
        
        removable = code_set - essential
        return essential, removable
    
    def try_find_replacement(c_to_remove, coverage, code_set):
        """Try to find a replacement for c_to_remove that maintains coverage."""
        # Find words uniquely covered by c_to_remove
        unique_words = []
        for w in balls_np[c_to_remove]:
            if coverage[w] == 1:
                unique_words.append(int(w))
        
        if not unique_words:
            return None  # Can just remove
        
        # Find candidate replacements: words that cover all unique_words
        # A candidate x must be within distance 1 of each unique_word
        # i.e., x must be in the intersection of balls[unique_words[i]] for all i
        if len(unique_words) > 3:
            # Too many constraints, likely no replacement
            return None
        
        # Start with ball of first unique word
        candidates = set(balls_np[unique_words[0]])
        for w in unique_words[1:]:
            candidates &= set(balls_np[w])
        
        # Filter: must not be in code
        valid = [w for w in candidates if w not in code_set]
        
        if not valid:
            return None
        
        return random.choice(valid)
    
    def greedy_remove_pass(coverage, code_set):
        """Remove all removable codewords in one pass."""
        removed = 0
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            essential, removable = get_essential_and_removable(coverage, code_set)
            if removable:
                # Remove one at a time to update coverage
                c = random.choice(list(removable))
                code_set.discard(c)
                coverage[balls_np[c]] -= 1
                removed += 1
                changed = True
        return removed
    
    def swap_phase(coverage, code_set):
        """Try swaps to enable further removals."""
        made_swap = False
        for c in list(code_set):
            if time.time() > deadline - 10:
                break
            replacement = try_find_replacement(c, coverage, code_set)
            if replacement is not None:
                code_set.discard(c)
                code_set.add(replacement)
                coverage[balls_np[c]] -= 1
                coverage[balls_np[replacement]] += 1
                made_swap = True
                break  # Restart after a swap
        return made_swap
    
    def simulated_annealing(coverage, code_set):
        """SA to escape local minima."""
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        best_code = set(code_set)
        best_size = len(code_set)
        
        while time.time() < deadline - 15 and T > T_min:
            move_type = random.random()
            
            if move_type < 0.5:
                # Try removal
                essential, removable = get_essential_and_removable(coverage, code_set)
                if removable:
                    c = random.choice(list(removable))
                    code_set.discard(c)
                    coverage[balls_np[c]] -= 1
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = set(code_set)
                else:
                    # Try swap
                    c = random.choice(list(code_set))
                    replacement = try_find_replacement(c, coverage, code_set)
                    if replacement is not None:
                        code_set.discard(c)
                        code_set.add(replacement)
                        coverage[balls_np[c]] -= 1
                        coverage[balls_np[replacement]] += 1
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = set(code_set)
            else:
                # Try swap
                c = random.choice(list(code_set))
                replacement = try_find_replacement(c, coverage, code_set)
                if replacement is not None:
                    code_set.discard(c)
                    code_set.add(replacement)
                    coverage[balls_np[c]] -= 1
                    coverage[balls_np[replacement]] += 1
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = set(code_set)
                else:
                    # Random move: add a random word
                    w_new = random.randint(0, total_words - 1)
                    if w_new not in code_set:
                        code_set.add(w_new)
                        coverage[balls_np[w_new]] += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Restore best
        code_set.clear()
        code_set.update(best_code)
        # Recompute coverage
        coverage_new = compute_coverage(code_set)
        coverage[:] = coverage_new
        return code_set
    
    # Main loop
    coverage = compute_coverage(code)
    
    # Phase 1: Aggressive greedy removal
    while time.time() < deadline - 30:
        removed = greedy_remove_pass(coverage, code)
        if removed == 0:
            break
    
    # Phase 2: Swap and remove iterations
    while time.time() < deadline - 20:
        # Try swaps
        if swap_phase(coverage, code):
            # After a swap, try to remove more
            greedy_remove_pass(coverage, code)
        else:
            # No swaps available, try SA
            if time.time() < deadline - 15:
                simulated_annealing(coverage, code)
                # After SA, try greedy removal again
                greedy_remove_pass(coverage, code)
            else:
                break
    
    # Final cleanup: remove any remaining redundancies
    while time.time() < deadline - 2:
        essential, removable = get_essential_and_removable(coverage, code)
        if removable:
            c = random.choice(list(removable))
            code.discard(c)
            coverage[balls_np[c]] -= 1
        else:
            break
    
    # Convert back to list of lists
    def int_to_list(val, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word
    
    result = [int_to_list(c, n) for c in code]
    return result
