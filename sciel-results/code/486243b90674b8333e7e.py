import itertools
import random
import time

def construct(q, n, radius):
    # The task specifies q=3, n=6, radius=1.
    # We assume the evaluator calls with these parameters.
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though the task is specific
        # For general case, a simple greedy might work, but we optimize for the specific case
        # Here we just return a trivial cover if not the specific case to avoid errors
        # But the prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
        # So we can assume q=3, n=6, radius=1.
        pass

    start_time = time.time()
    deadline = start_time + 110  # 110s limit, 10s buffer
    
    # Total number of words in the space
    total_words = q ** n
    
    # Map each word to an index 0..total_words-1
    # We use base-q representation
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute the balls (set of indices covered by each codeword)
    # Ball(c) = {c} U {c with 1 position changed}
    balls = [set() for _ in range(total_words)]
    
    # Generate all words and their indices
    all_words = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        all_words.append(w)
        balls[idx].add(idx)
        
        # Add neighbors at distance 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                nidx = word_to_idx(w)
                balls[idx].add(nidx)
            w[pos] = orig_val

    # Initial Code: Direct Sum Construction
    # Tetracode for n=4
    tetracode_indices = []
    for a in range(3):
        for b in range(3):
            w4 = [a, b, (a + b) % 3, (a + 2 * b) % 3]
            # Extend to n=6 by appending 0,0 (just one tail? No, the seed used all tails)
            # The seed used: [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            # This creates 9 * 9 = 81 words.
            # Let's start with this set.
            pass
            
    initial_code = set()
    for a in range(3):
        for b in range(3):
            w4 = [a, b, (a + b) % 3, (a + 2 * b) % 3]
            for tail in itertools.product(range(3), repeat=n - 4):
                w6 = w4 + list(tail)
                idx = word_to_idx(w6)
                initial_code.add(idx)
                
    current_code = list(initial_code)
    
    def is_cover(code_set):
        covered = set()
        for c in code_set:
            covered.update(balls[c])
        return len(covered) == total_words

    def prune(code_list):
        code_set = set(code_list)
        changed = True
        while changed:
            changed = False
            # Calculate coverage count for each word in the space
            # We need to know if a word is covered by >1 codeword in code_set
            # If a codeword c is removed, words covered ONLY by c will become uncovered.
            
            # Efficient check:
            # Count how many codewords in code_set cover each word in the space.
            # But we only care about words in the space.
            # Let's compute the set of words covered by code_set.
            # And for each c in code_set, check if removing c leaves any word uncovered.
            
            # Optimization:
            # Maintain a count of coverage for each word?
            # Since we are pruning, the set changes.
            
            # Simpler: 
            # For each c in code_set:
            #   Let U_c be the set of words covered by c.
            #   Let R_c be U_c \ (Union of balls of other codewords in code_set).
            #   If R_c is empty, c is redundant.
            
            # To make this fast, we can just check:
            # Is there any word w in balls[c] such that no other c' in code_set \ {c} covers w?
            
            for c in list(code_set):
                # Check if c is essential
                essential = False
                for w in balls[c]:
                    # Check if w is covered by any other codeword
                    covered_by_other = False
                    for c_prime in code_set:
                        if c_prime == c:
                            continue
                        if w in balls[c_prime]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        essential = True
                        break
                if not essential:
                    code_set.remove(c)
                    changed = True
                    if time.time() > deadline:
                        return list(code_set)
        return list(code_set)

    # Initial Prune
    current_code = prune(current_code)
    
    best_code = list(current_code)
    
    # Local Search
    # We try to swap words to find a configuration that allows further pruning.
    # Or simply try to reduce the size.
    
    # Heuristic: 
    # 1. Pick a random c in current_code.
    # 2. Remove it.
    # 3. If coverage is broken, find a c' not in current_code that covers the "gap".
    # 4. If found, swap.
    # 5. Prune.
    
    # To find c' efficiently:
    # When we remove c, the set of uncovered words U is balls[c] \ (Covered by others).
    # We need a c' such that U subset of balls[c'].
    # Note: c' might also cover other words, which is fine.
    
    # We can precompute for every pair of words? No, 729^2 is ~500k, which is fine.
    # But we need to check if U is subset of balls[c'].
    # U is small (<= 13).
    
    def find_replacement(c_removed, code_set):
        # Calculate uncovered words
        # Covered by others = Union(balls[c'] for c' in code_set if c' != c_removed)
        # This is expensive to compute from scratch every time.
        
        # Alternative:
        # We know that before removal, everything was covered.
        # The words that become uncovered are those in balls[c_removed] that are NOT in balls[c'] for any c' in code_set \ {c_removed}.
        
        # Let's compute the set of words covered by code_set \ {c_removed}
        # This is still O(|code| * 13).
        
        covered_by_others = set()
        for c_prime in code_set:
            if c_prime == c_removed:
                continue
            covered_by_others.update(balls[c_prime])
            
        uncovered = balls[c_removed] - covered_by_others
        
        if not uncovered:
            return None # No replacement needed, just remove
            
        # Find a c' not in code_set such that uncovered subset of balls[c']
        for c_prime in range(total_words):
            if c_prime in code_set:
                continue
            if uncovered.issubset(balls[c_prime]):
                return c_prime
        return None

    # Randomized Local Search Loop
    no_improve_count = 0
    max_no_improve = 500 # Restart logic
    
    while time.time() < deadline:
        if len(current_code) < len(best_code):
            best_code = list(current_code)
            no_improve_count = 0
            
        if no_improve_count > max_no_improve:
            # Restart with a perturbed version of best_code or random
            # Let's restart with a random set of size len(best_code) + 5
            # Or just restart the search from best_code
            current_code = list(best_code)
            # Add some random noise
            for _ in range(5):
                if current_code:
                    # Remove a random one
                    idx_to_remove = random.randint(0, len(current_code)-1)
                    current_code.pop(idx_to_remove)
                    # Add a random new one
                    new_c = random.randint(0, total_words-1)
                    if new_c not in current_code:
                        current_code.append(new_c)
            # Prune to fix coverage?
            # Actually, if we break coverage, we can't prune.
            # So we should only perturb if we maintain coverage or fix it.
            # Better: Just reset to best_code and try different random seeds for swaps.
            current_code = list(best_code)
            no_improve_count = 0
            continue

        # Pick a random codeword to try to remove
        if not current_code:
            break
            
        c_idx = random.randint(0, len(current_code) - 1)
        c_removed = current_code[c_idx]
        code_set = set(current_code)
        
        # Try to remove c_removed
        # Check if it's redundant
        # We can use the logic from prune, but for a single element.
        
        # Calculate uncovered if c_removed is removed
        covered_by_others = set()
        for c_prime in code_set:
            if c_prime == c_removed:
                continue
            covered_by_others.update(balls[c_prime])
            
        uncovered = balls[c_removed] - covered_by_others
        
        if not uncovered:
            # Safe to remove
            code_set.remove(c_removed)
            current_code = list(code_set)
            # Prune again to see if more can be removed
            current_code = prune(current_code)
            no_improve_count = 0 # Progress made
            continue
        else:
            # Need replacement
            replacement = find_replacement(c_removed, code_set)
            if replacement is not None:
                code_set.remove(c_removed)
                code_set.add(replacement)
                current_code = list(code_set)
                # Try to prune
                current_code = prune(current_code)
                # If size didn't change, we might be stuck in a local minimum
                # But the structure changed, which might allow future removals
                # So we don't increment no_improve_count if we made a valid swap?
                # Actually, if size is same, we are not closer to the goal (min size).
                # But we might be in a better basin.
                # Let's increment no_improve_count if size didn't decrease.
                if len(current_code) >= len(best_code):
                    no_improve_count += 1
            else:
                # No replacement found, try another
                no_improve_count += 1

    return [idx_to_word(idx) for idx in best_code]
