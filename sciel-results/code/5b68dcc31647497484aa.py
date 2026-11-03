import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    # Helper functions
    def list_to_int(word):
        val = 0
        for digit in word:
            val = val * q + digit
        return val

    def int_to_list(val, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Generate initial seed: Direct sum of tetracode
    # Tetracode: 9 words of length 4 over {0,1,2}
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(list_to_int(word))
    
    code = set(initial_codewords)
    
    # Precompute balls for all q^n words
    all_words = list(range(q**n))
    balls = {}
    for w in all_words:
        b = [w]
        w_list = int_to_list(w, n)
        for i in range(n):
            for d in range(q):
                if d != w_list[i]:
                    new_word = w_list.copy()
                    new_word[i] = d
                    b.append(list_to_int(new_word))
        balls[w] = set(b)
    
    # Coverage tracking
    def compute_coverage():
        """Compute coverage counts and unique coverers."""
        coverage = {w: 0 for w in all_words}
        unique_coverer = {w: None for w in all_words}
        
        for c in code:
            for w in balls[c]:
                coverage[w] += 1
                if coverage[w] == 1:
                    unique_coverer[w] = c
                else:
                    unique_coverer[w] = None
        
        return coverage, unique_coverer
    
    def get_removable():
        """Find codewords that can be removed without breaking coverage."""
        if time.time() > deadline - 5:
            return None
        
        coverage, unique_coverer = compute_coverage()
        
        essential = set()
        for w in all_words:
            if unique_coverer[w] is not None:
                essential.add(unique_coverer[w])
        
        removable = code - essential
        if not removable:
            return None
        return removable
    
    def try_swap(c_to_remove):
        """Try to find a replacement for c_to_remove."""
        if time.time() > deadline - 5:
            return None
        
        coverage, unique_coverer = compute_coverage()
        
        # Find words uniquely covered by c_to_remove
        unique_words = []
        for w in all_words:
            if unique_coverer[w] == c_to_remove:
                unique_words.append(w)
        
        if not unique_words:
            # Can just remove
            return None
        
        # Find candidate replacements that cover all unique_words
        candidate_ws = set()
        for w in unique_words:
            candidate_ws.add(w)
            for nb in balls[w]:
                candidate_ws.add(nb)
        
        # Filter candidates: must cover all unique_words and not be in code
        valid_candidates = []
        for w_new in candidate_ws:
            if w_new in code:
                continue
            if all(w in balls[w_new] for w in unique_words):
                # Check if removing c_to_remove and adding w_new keeps all covered
                # Words that lose coverage: those covered only by c_to_remove
                # We already know unique_words are covered by w_new
                # Other words covered by c_to_remove are also covered by others
                valid_candidates.append(w_new)
        
        if not valid_candidates:
            return None
        
        return random.choice(valid_candidates)
    
    def simulated_annealing():
        """Run simulated annealing to escape local minima."""
        if time.time() > deadline - 10:
            return
        
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        best_code = set(code)
        best_size = len(code)
        
        while time.time() < deadline - 10 and T > T_min:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Try removal
                removable = get_removable()
                if removable:
                    c = random.choice(list(removable))
                    code.remove(c)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = set(code)
                else:
                    # No removal possible, try swap
                    c = random.choice(list(code))
                    w_new = try_swap(c)
                    if w_new is not None:
                        code.remove(c)
                        code.add(w_new)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = set(code)
            else:
                # Try swap
                c = random.choice(list(code))
                w_new = try_swap(c)
                if w_new is not None:
                    code.remove(c)
                    code.add(w_new)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = set(code)
                else:
                    # Try adding a random word (expansion)
                    w_new = random.choice(all_words)
                    if w_new not in code:
                        code.add(w_new)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Restore best code found
        code.clear()
        code.update(best_code)
    
    # Main Loop: First greedy removal, then simulated annealing
    while time.time() < deadline - 15:
        # Greedy removal phase
        removable = get_removable()
        if removable:
            c = random.choice(list(removable))
            code.remove(c)
            continue
        
        # If no more greedy removal, run simulated annealing
        if time.time() > deadline - 10:
            break
        
        simulated_annealing()
        
        # After SA, try greedy removal again
        removable = get_removable()
        if not removable:
            # Check if we can do any more swaps
            improved = False
            for c in list(code):
                w_new = try_swap(c)
                if w_new is not None:
                    code.remove(c)
                    code.add(w_new)
                    improved = True
                    break
            if not improved:
                break
    
    # Final greedy removal pass
    while time.time() < deadline - 2:
        removable = get_removable()
        if removable:
            c = random.choice(list(removable))
            code.remove(c)
        else:
            break
    
    # Convert back to list of lists
    result = [int_to_list(c, n) for c in code]
    return result
